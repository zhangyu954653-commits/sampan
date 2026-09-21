"""主流程：音频 → 断句 → 识别 → 判语种 → 翻译 → 推送到网页。"""
from __future__ import annotations

import asyncio
import difflib
import itertools
import queue
import threading
import time

import numpy as np

from . import quick
from .asr import ASR
from .audio_capture import AudioCapture
from .echo import EchoGuard
from .translate import Translator
from .vad import Segmenter

CHANNEL_LABEL = {"me": "我方", "them": "对方"}


def friendly_error(e: Exception, backend: str = "anthropic") -> str:
    """把 API 报错翻成字幕上看得懂的一句话。

    报错里必须说对是哪一家。以前这里写死了 Anthropic——切到 OpenAI 之后
    出问题会让人跑去 Anthropic 控制台查，越查越糊涂。
    """
    openai = str(backend).lower() == "openai"
    who = "OpenAI" if openai else "Anthropic"
    key_field = "openai_api_key" if openai else "anthropic_api_key"
    url_field = "openai_base_url" if openai else "anthropic_base_url"
    model_field = "openai_translate_model" if openai else "model"

    name = type(e).__name__
    msg = str(e).lower()
    if "ratelimit" in name.lower() or "429" in msg or "rate_limit" in msg:
        return f"⚠ API 限流：请到 {who} 控制台把每分钟请求数调高"
    if "authentication" in name.lower() or "401" in msg \
            or "invalid x-api-key" in msg or "invalid_api_key" in msg:
        return f"⚠ {who} 的 API Key 无效，请检查 config.json 的 {key_field}"
    if "credit balance" in msg or "insufficient" in msg or "402" in msg \
            or "quota" in msg:
        return f"⚠ {who} 账户余额不足或配额用尽，请去充值"
    if "not_found" in msg or "404" in msg or "does not exist" in msg:
        return f"⚠ 模型名或中转地址不对，请检查 config.json 的 {model_field} / {url_field}"
    if "connect" in msg or "timeout" in msg or "ssl" in msg:
        return f"⚠ 连不上 {who}，检查网络或改用中转地址"
    return f"（翻译失败：{name}）"


class Pipeline:
    def __init__(self, cfg: dict, broadcast):
        self.cfg = cfg
        self.broadcast = broadcast          # async def broadcast(event: dict)

        self.audio_q: queue.Queue = queue.Queue(maxsize=400)
        # 队列必须短。识别在 CPU 上是串行的，一句 1~3 秒；队列开大了
        # 就会积压出几十秒的延迟，而实时字幕里过期的句子翻出来也没用。
        self.seg_q: queue.Queue = queue.Queue(maxsize=6)
        self.max_lag = float(cfg.get("max_lag_sec", 6.0))
        self.dropped = 0
        # 每一路各环节的进出账，出问题时一眼看出卡在哪
        self.stats = {ch: dict(断出=0, 回声丢弃=0, 排队超时丢弃=0, 队列挤掉=0,
                               识别为空=0, 语气词丢弃=0, 重复丢弃=0,
                               秒回=0, 出字幕=0)
                      for ch in ("me", "them")}
        self._last_report = time.time()
        # 同一段发言里沿用已判出的语种，隔这么久没说话就重新判
        self.lang_reuse_sec = float(cfg.get("lang_reuse_sec", 10.0))
        self._lang_of: dict[str, str] = {}
        self._lang_ts: dict[str, float] = {}

        self.capture = AudioCapture(cfg, self.audio_q)
        self.asr = ASR(cfg["asr"])
        # 没戴耳机时，对方的声音会从音箱漏回麦克风，被当成"我方"再翻一遍。
        # 这一道按声音本身判，比按识别出的文字比对可靠得多（见 echo.py 开头）。
        self.echo = EchoGuard(cfg.get("echo_cancel"))
        # 观察模式：判定照跑、日志照打，但不真的丢——
        # 用来在真实环境里核对它拦的到底是不是对方的话。
        self.echo_observe = str(
            (cfg.get("echo_cancel") or {}).get("mode", "on")).lower() == "observe"
        self.gender = str(cfg.get("speaker_gender", ""))
        self.drop_fillers = bool(cfg.get("drop_fillers", True))
        self.quick_replies = bool(cfg.get("quick_replies", True))

        from pathlib import Path

        from .transcript import Transcript

        self.transcript = Transcript(cfg, Path(__file__).resolve().parent.parent)

        # 没配 Key 也允许启动：可以先测麦克风和识别，字幕上会提示去配 Key
        try:
            # 翻译后端可切换。默认仍是 Anthropic——那条路实测跑过几场会，
            # 加选项不能顺手把能用的默认改掉。
            if str(cfg.get("translate_backend", "anthropic")).lower() == "openai":
                from .translate_openai import OpenAITranslator

                self.translator = OpenAITranslator(cfg)
                print(f"[翻译] 后端：OpenAI "
                      f"{cfg.get('openai_translate_model', 'gpt-4o-mini')}")
            else:
                self.translator = Translator(cfg)
            self.translate_error = ""
        except Exception as e:
            self.translator = None
            self.translate_error = str(e)
            print(f"[翻译] 未启用：{e}")

        self.segmenters = {
            "me": Segmenter(cfg["vad"]),
            "them": Segmenter(cfg["vad"]),
        }

        self._ids = itertools.count(1)
        self._stop = threading.Event()
        self._seg_thread: threading.Thread | None = None
        self._partial_thread: threading.Thread | None = None
        self._asr_busy = False
        self._loop: asyncio.AbstractEventLoop | None = None

        self.history: list[dict] = []        # 给翻译当上下文 + 新客户端补历史
        self.muted = {"me": False, "them": False}
        # 暂停：会开完了人还在屋里说话，照样被识别+翻译，白烧钱。
        # 暂停期间不断句、不识别、不翻译，也就不产生任何调用。
        self.paused = False
        self._paused_at = 0.0
        self.paused_total = 0.0
        # 流式识别（WebSocket 双向流）。默认关——它按音频秒数计费，
        # 而且境外调用要单独开通跨境版。开关在 config.json 的 stream_asr。
        self.stream = None
        self.stream_wait = float(cfg.get("stream_wait_sec", 2.5))
        self.fast_error = ""          # 极速模式开不了时，把原因告诉界面
        lock = cfg.get("language_lock") or {}
        self.locks = {"me": lock.get("me"), "them": lock.get("them")}
        self.status = {"asr": "", "devices": {}, "ready": False}

        # (时间, 声道, 文本, 字幕 id) 用于去重。带上 id 才能事后撤回。
        self._recent: list[tuple[float, str, str, int | None]] = []

    # ================================================================
    # 启动 / 停止
    # ================================================================
    async def start(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

        print("[识别] 正在加载语音模型（首次运行需要下载，请耐心等待）…")
        name = await loop.run_in_executor(None, self.asr.load)
        self.status["asr"] = name
        print(f"[识别] 就绪：{name}")

        self.status["devices"] = self.capture.start()
        for ch, dev in self.status["devices"].items():
            print(f"[音频] {CHANNEL_LABEL[ch]} ← {dev}")

        self._seg_thread = threading.Thread(target=self._segment_loop, daemon=True,
                                            name="segmenter")
        self._seg_thread.start()

        if self.asr.partial_on and self.asr._partial_model is not None:
            self._partial_thread = threading.Thread(target=self._partial_loop,
                                                    daemon=True, name="partial")
            self._partial_thread.start()
            print("[识别] 边说边显示：已开启")

        # 流式识别：能用才开，用不了就安静地继续用现在这条路。
        # 绝不能因为流式配置有问题就让整个程序起不来。
        if self.cfg.get("stream_asr"):
            from .stream_manager import StreamManager

            mgr = StreamManager(self.cfg, loop, self._stream_partial)
            ok, why = mgr.available()
            if ok:
                self.stream = mgr
                print("[识别] 流式识别：已开启（边说边识别，按音频秒数计费）")
            else:
                print(f"[识别] 流式识别用不了，继续用一句话识别：{why}")

        self.status["ready"] = True
        asyncio.create_task(self._worker())
        await self._push({"type": "status", **self.public_status()})

    def stop(self):
        self._stop.set()
        self.capture.stop()
        if self.stream is not None:
            self.stream.cancel_all()
            line = self.stream.report()
            if line:
                print("\n[用量] 流式识别")
                print(line)
        if self.paused and self._paused_at:
            self.paused_total += time.time() - self._paused_at
        tc = getattr(self.asr, "_tencent", None)
        if tc is not None and tc.calls:
            print(f"\n[用量] 本场会议调用腾讯云 {tc.calls} 次"
                  f"　（每月免费额度 {tc.free_quota} 次，1 号重置）")
        if self.paused_total > 30:
            print(f"[用量] 本场暂停了 {self.paused_total / 60:.1f} 分钟，"
                  f"这段时间没有任何识别和翻译调用")
        path = self.transcript.close()
        if path:
            print(f"[记录] 本场会议已保存：{path}")

    def public_status(self) -> dict:
        return {
            "asr": self.status["asr"],
            "devices": {CHANNEL_LABEL[k]: v for k, v in self.status["devices"].items()},
            "ready": self.status["ready"],
            "muted": dict(self.muted),
            "locks": dict(self.locks),
            "paused": self.paused,
            # 菜单里显不显示这两个入口。不影响行为，只影响看不看得见。
            "fast_enabled": bool(self.cfg.get("fast_enabled", True)),
            "share_enabled": bool(self.cfg.get("share_enabled", True)),
            "fast": self.stream is not None,
            "fast_error": self.fast_error,
            # 这场极速模式已经花了多少，实时显示——看得见才不会忘记关
            "fast_cost": (round(self.stream.asr.seconds / 3600 * 8.6, 2)
                          if self.stream is not None else 0.0),
            "fast_minutes": (round(self.stream.asr.seconds / 60, 1)
                             if self.stream is not None else 0.0),
            # 显示**当前真正在用的**那个模型。以前这里写死读 cfg["model"]，
            # 那是 Anthropic 的字段——切到 OpenAI 之后顶栏还显示 claude-opus-5，
            # 看着像没切成功。直接问翻译器自己是谁，就不会再对不上。
            "model": (getattr(self.translator, "model", None)
                      or self.cfg.get("model")) if self.translator
                     else "翻译未启用",
        }

    def set_lock(self, channel: str, lang):
        """锁定/解锁某一路的语言。lang 为 None 表示恢复自动判定。

        改动会写回 config.json。不写回的话，界面上的选择只活在内存里，
        下次启动又回到上一场会议的设置——实测就因为这个，
        和泰国同事开会时开场两句泰语被强行塞进中文引擎，出来一堆乱码。
        """
        if channel not in self.locks:
            return
        self.locks[channel] = lang if lang in ("zh", "th") else None
        self._lang_of.pop(channel, None)      # 清掉沿用的旧判定
        name = {"zh": "中文", "th": "泰语"}.get(self.locks[channel], "自动判定")
        print(f"[语种] {CHANNEL_LABEL[channel]} 已设为：{name}")

        self.cfg.setdefault("language_lock", {})[channel] = self.locks[channel]
        updates = {"language_lock": {channel: self.locks[channel]}}
        # 先验偏向跟着锁定一起走，**包括改回"自动"时要清掉**。
        #
        # 这条最初写成"改成自动时不动偏向"，理由是那算用户上一次的明确信号。
        # 实测证明是错的：用户把对方从"泰语"改回"自动"之后，偏向还留着 th，
        # 而那场会对方 87% 说中文。后果是每句话都拿一个错的先验去判语种：
        #   · 两个概率都低于噪声门限时会**无条件**返回偏向（asr.py 的 noise_floor）
        #   · 有把握时也会给偏向那一侧乘 2.5 倍
        # 判出来和猜测不符就要补打一次云端往返 —— 实测每句 1.34 次调用
        # （锁定时是 1.14 次），延迟中位数从 0.9s 涨到 1.9s。
        #
        # 偏向本来就是我按锁定推出来的，不是用户自己选的，所以解锁时
        # 一并清掉才诚实：不知道这一路说什么，就别装作知道。
        self.cfg.setdefault("language_bias", {})[channel] = self.locks[channel]
        updates["language_bias"] = {channel: self.locks[channel]}

        from .config import save_fields

        if save_fields(updates):
            print(f"       已记住，下次启动仍是「{name}」")

    # ================================================================
    # 线程：音频 → 句子
    # ================================================================
    def _segment_loop(self):
        while not self._stop.is_set():
            try:
                channel, pcm = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                channel = pcm = None

            now = time.time()
            if self.paused:
                # 照样把队列取空，不然采集线程那边会一直撞满队列丢帧；
                # 但不喂给断句器，也就不会产生任何句子。
                continue
            if channel is not None:
                if channel == "them":
                    # 先存参考再断句：麦克风那一句收尾时要回头比对这段时间
                    # 音箱在放什么。对方这一路即使被静音了也要存——
                    # 静音只是不出字幕，音箱照样在响，回声照样会漏进麦克风。
                    self.echo.push_reference(pcm, now)
                seg = self.segmenters[channel]
                was = seg.speaking
                finished = seg.push(pcm, now)
                # 流式：人一开口就把音频往云端送，说的过程中文字已经在回来。
                # 只在有人说话时送——实时识别按音频秒数计费，挂着空推流是烧钱。
                if self.stream is not None and not self.muted.get(channel) \
                        and (was or seg.speaking):
                    self.stream.feed(channel, pcm, self._stream_lang(channel))
                self._dispatch(channel, finished, now)

            # 每轮都检查一次断流：麦克风会持续供数据，队列基本不会空，
            # 所以不能只在 queue.Empty 的时候才收尾系统声音那一路。
            for ch, seg in self.segmenters.items():
                if ch != channel:
                    self._dispatch(ch, seg.tick(now), now)

    async def set_fast(self, value: bool) -> dict:
        """开/关极速模式（流式识别）。

        分成两档是用户自己提的，而且是对的：
          平时开会    → 一句话识别，¥1.4/小时，每月免费 5000 次
          重要会议    → 流式识别，快约 1.1 秒，¥5.9/小时，无免费额度

        所以这个开关必须在界面上，不能让人去改配置文件。

        **默认每次启动都是关的**：忘记关会一直烧钱，忘记开只是慢一点，
        两种遗忘的代价不对等，所以往安全的那边默认。
        """
        value = bool(value)
        # config.json 里关掉了就是真的关掉，不只是界面上看不见
        if value and not self.cfg.get("fast_enabled", True):
            print("[极速] config.json 里 fast_enabled 是 false，没开这个功能")
            return self.public_status()
        if value == (self.stream is not None):
            return self.public_status()

        if not value:
            if self.stream is not None:
                self.stream.cancel_all()
                line = self.stream.report()
                self.stream = None
                print("[极速] 已关闭，回到一句话识别" + (f"\n{line}" if line else ""))
            self.fast_error = ""
            return self.public_status()

        from .stream_manager import StreamManager

        mgr = StreamManager({**self.cfg, "stream_asr": True},
                            self._loop, self._stream_partial)
        # 先探一下再开。等到开会说第一句才发现用不了，那时候人已经在等字幕了。
        ok, why = await mgr.asr.probe()
        if not ok:
            self.fast_error = why
            print(f"[极速] 开不了：{why}")
            return self.public_status()
        self.stream = mgr
        self.fast_error = ""
        print("[极速] 已开启：边说边识别，按音频秒数计费（约 ¥8.6/小时音频）")
        return self.public_status()

    async def _stream_partial(self, channel: str, text: str):
        """流式过程中回来的中间结果，直接当"正在识别"推给界面。
        比本地小模型准得多，而且不占 CPU。"""
        if self.paused or self.muted.get(channel) or not text:
            return
        await self._push({"type": "partial", "channel": channel,
                          "speaker": CHANNEL_LABEL[channel], "text": text})

    def _stream_lang(self, channel: str) -> str | None:
        """流式会话要在开口那一刻就定好语种——没法像批量那样先判再识别。
        锁定了就用锁定的；没锁就沿用这一路上一句的；都没有就按偏向，再没有就中文。
        """
        locked = self.locks.get(channel)
        if locked in ("zh", "th"):
            return locked
        known = self._lang_of.get(channel)
        if known in ("zh", "th"):
            return known
        bias = (self.cfg.get("language_bias") or {}).get(channel)
        return bias if bias in ("zh", "th") else "zh"

    def _dispatch(self, channel: str, finished: list, now: float):
        if not finished or self.muted.get(channel):
            if finished and self.stream is not None:
                self.stream.cancel(channel)       # 静音的那一路不留会话
            return

        for audio, start_ts, partial_cut in finished:
            self.stats[channel]["断出"] += 1

            # 这里曾经有一道"回声抑制"：只要对方那一路最近出过声，就丢掉
            # 麦克风这一句。在真实通话里对方的声音几乎一直在响，结果是
            # 把自己说的话**全部**丢光，那一版等于废品。
            #
            # 现在换成比对声音本身：把这一句和同一时刻音箱在放的内容做
            # 包络互相关。"对方刚才出过声"说明不了任何问题，
            # "这一句的波形就是音箱那段波形"才说明是回声。
            note = ""
            if channel == "me":
                is_echo, why = self.echo.check(
                    audio, start_ts, start_ts + len(audio) / 16000.0)
                if is_echo:
                    if self.echo_observe:
                        # 观察模式：照常往下送，只在终端标出"本来会被丢掉"。
                        # 这样能连着识别出来的文字一起看——是对方的话才算判对。
                        note = why
                    else:
                        self.stats[channel]["回声丢弃"] += 1
                        print(f"[回声] 麦克风这一句是音箱漏回来的，丢弃（{why}）")
                        if self.stream is not None:
                            self.stream.cancel(channel)   # 回声不用翻，也别浪费识别
                        continue

            # 流式那边的最终文本用 future 交过来，_worker 去 await。
            # 拿不到就是 None，_worker 照老路走批量识别，不会丢句子。
            fut = self.stream.finish(channel) if self.stream is not None else None
            item = (channel, audio, now, partial_cut, note, fut)
            try:
                self.seg_q.put_nowait(item)
            except queue.Full:
                # 满了要丢最旧的那句，不是丢刚说的这句——
                # 新的话永远比旧的有用。
                try:
                    old_ch = self.seg_q.get_nowait()[0]
                    self.dropped += 1
                    self.stats[old_ch]["队列挤掉"] += 1
                except queue.Empty:
                    pass
                try:
                    self.seg_q.put_nowait(item)
                except queue.Full:
                    pass

    def _report(self, force: bool = False):
        """每分钟在终端打一份账：两路各断出多少句、各丢在哪一环。

        出问题时不用再猜——直接看哪一列的数字不对。
        """
        now = time.time()
        if not force and now - self._last_report < 60:
            return
        self._last_report = now
        # 极速模式开着时，把已经花掉的钱推给界面。看得见才不会忘记关。
        if self.stream is not None and self._loop is not None:
            asyncio.run_coroutine_threadsafe(
                self._push({"type": "status", **self.public_status()}),
                self._loop)
        if not any(s["断出"] for s in self.stats.values()):
            return
        print("\n──── 最近一分钟 ────")
        for ch, s in self.stats.items():
            if not s["断出"]:
                continue
            # 秒回不是损耗，是省下来的，别和丢弃的原因混在一起
            lost = "，".join(f"{k} {v}" for k, v in s.items()
                             if k not in ("断出", "出字幕", "秒回") and v)
            line = (f"  {CHANNEL_LABEL[ch]}：断出 {s['断出']} 句 → "
                    f"出字幕 {s['出字幕']} 句")
            if lost:
                line += f"（{lost}）"
            if s["秒回"]:
                line += f"　其中 {s['秒回']} 句查表秒回，没走模型"
            print(line)
        tc = getattr(self.asr, "_tencent", None)
        if tc is not None and tc.calls:
            print(f"  腾讯云本次已调用 {tc.calls} 次"
                  f"（每月免费 {tc.free_quota} 次，超出约 ¥3.2/千次）")
        # 声学回声判定的体检数据。它在真实环境里一直没触发过，
        # 而阈值是在合成音频上调出来的——把实际分数打出来，
        # 下次不用猜就知道差在哪一项、差多少。
        sc = self.echo.scores
        if sc:
            c = sorted(s[0] for s in sc)
            v = sorted(s[1] for s in sc)
            print(f"  回声判定：判了 {len(sc)} 句，"
                  f"相关度中位 {c[len(c)//2]:.2f}／最高 {c[-1]:.2f}（要 ≥{self.echo.corr_min}）"
                  f"　覆盖率中位 {v[len(v)//2]:.0%}／最高 {v[-1]:.0%}"
                  f"（要 ≥{self.echo.cover_min:.0%}）")
            if self.echo._lag_est is not None:
                print(f"　　　　　两路时间差已标定 {self.echo._lag_est*1000:+.0f}ms，"
                      f"拦下 {self.echo.dropped} 句")
            self.echo.scores = []
        print("────────────────\n")
        for s in self.stats.values():
            for k in s:
                s[k] = 0

    # ================================================================
    # 线程：说话过程中不断刷新"正在识别"的中间结果
    # ================================================================
    def _partial_loop(self):
        """用空闲算力跑中间结果，让人看到系统正在听。

        铁律：正式识别优先。只要正式识别在忙或队列里有活，就不跑中间结果——
        中间结果只是心理安慰，绝不能拖慢真正要用的那份译文。
        """
        min_audio = float(self.cfg.get("partial_min_sec", 1.0))
        every = float(self.cfg.get("partial_interval_sec", 1.1))
        last_len = {"me": 0, "them": 0}
        last_run = {"me": 0.0, "them": 0.0}
        shown = {"me": "", "them": ""}
        lang_of = {"me": None, "them": None}   # 每句话判一次语种，之后复用

        while not self._stop.is_set():
            time.sleep(0.25)
            if self.paused:
                continue

            for ch, seg in self.segmenters.items():
                # 说完了就把这一路的中间结果撤掉，等正式字幕顶上
                if not seg.speaking:
                    if shown[ch] or lang_of[ch]:
                        shown[ch] = ""
                        lang_of[ch] = None
                        last_len[ch] = 0
                        self._push_sync({"type": "partial", "channel": ch, "text": ""})
                    continue

                if self.muted.get(ch):
                    continue
                if self._asr_busy or not self.seg_q.empty():
                    continue        # 正式识别在忙，让路
                now = time.time()
                if now - last_run[ch] < every:
                    continue

                audio = seg.current_audio()
                if audio is None or len(audio) < 16000 * min_audio:
                    continue
                if len(audio) - last_len[ch] < 16000 * 0.8:
                    continue        # 没多说几个字，不用重跑

                last_run[ch] = now
                last_len[ch] = len(audio)
                if lang_of[ch] is None:
                    prefer = self.cfg.get("language_bias", {}).get(ch)
                    lang_of[ch] = self.asr.partial_language(audio, prefer)
                text = self.asr.partial(audio, lang_of[ch])
                if text and text != shown[ch]:
                    shown[ch] = text
                    self._push_sync({"type": "partial", "channel": ch,
                                     "speaker": CHANNEL_LABEL[ch], "text": text})

    def _push_sync(self, event: dict):
        """从工作线程往 asyncio 事件循环里投递消息。"""
        if self._loop is None or self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._push(event), self._loop)
        except Exception:
            pass

    # ================================================================
    # 协程：句子 → 识别 → 翻译 → 推送
    # ================================================================
    async def _worker(self):
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            try:
                channel, audio, ts, partial_cut, echo_note, stream_fut = \
                    await loop.run_in_executor(None, self.seg_q.get, True, 0.3)
            except queue.Empty:
                continue
            except Exception:
                continue

            # 排太久的直接丢掉：等它翻出来对话早过去了，
            # 而且只会让后面每一句都更迟。宁可漏一句，也不能整体落后。
            lag = time.time() - ts
            if lag > self.max_lag:
                self.dropped += 1
                self.stats[channel]["排队超时丢弃"] += 1
                print(f"[识别] 积压 {lag:.1f}s，丢弃 {CHANNEL_LABEL[channel]} 一句"
                      f"（累计 {self.dropped}）")
                continue

            t0 = time.time()
            self._asr_busy = True        # 让中间结果线程让路
            try:
                prefer = self.cfg.get("language_bias", {}).get(channel)
                # 锁定了就完全跳过判定。短音频的语种判定不可靠，
                # 而"对方全程说泰语"这种情况锁死才是 100% 可靠的。
                locked = self.locks.get(channel)
                if locked in ("zh", "th"):
                    known = locked          # 锁定时 known 不参与判定，只是占位
                else:
                    # 上一句的语种只作为"猜测"传下去：云端路径会同时重新判语种，
                    # 猜对了省一次往返，猜错了自动纠正。
                    # 不能当结论——多人会议里同一路混着中文和泰语，
                    # 沿用上一句会把说泰语的人judge成中文。
                    known = self._lang_of.get(channel)
                    if ts - self._lang_ts.get(channel, 0) > self.lang_reuse_sec:
                        known = None
                busy = not self.seg_q.empty()      # 后面还排着队就别做额外的重试
                # 流式那边说不定已经把文字识别好了——人说话的那几秒里就在识别，
                # 到这里往往只剩收尾。拿到了就直接用，省掉整段音频上传再等结果。
                text = lang = None
                if stream_fut is not None:
                    try:
                        got = await asyncio.wait_for(
                            self.stream.wrap(stream_fut),
                            timeout=self.stream_wait)
                        if got:
                            text, lang = got
                    except asyncio.TimeoutError:
                        print(f"[流式] {CHANNEL_LABEL[channel]} 等结果超时，"
                              f"退回一句话识别")
                    except Exception as e:
                        print(f"[流式] 取结果出错，退回一句话识别：{e}")
                if not text:
                    text, lang = await loop.run_in_executor(
                        None, self.asr.transcribe, audio, prefer, known, busy,
                        locked
                    )
                else:
                    text = self.asr.apply_fixes(text)
            except Exception as e:
                print(f"[识别] 出错：{e}")
                continue
            finally:
                self._asr_busy = False
            cost = time.time() - t0
            if cost > 2.5 or lag > 2.0:
                print(f"[识别] {CHANNEL_LABEL[channel]} {len(audio)/16000:.1f}s 音频 → "
                      f"识别耗时 {cost:.1f}s，排队 {lag:.1f}s，队列 {self.seg_q.qsize()}")

            # 这一句是暂停之前取出来的，识别已经跑完了，但人已经按下暂停——
            # 再让它冒出来会很突兀，而且翻译那一步的钱还没花，正好省下
            if self.paused:
                continue

            if echo_note:
                # 观察模式的核心：把"本来会被丢掉"和识别出来的文字摆在一起。
                # 这里印出来的如果是对方刚说过的话，就说明判对了；
                # 如果是你自己说的，说明它在你的环境里会误伤，该关掉。
                print(f"[回声·观察] 这一句本来会被当成回声丢掉（{echo_note}）\n"
                      f"           识别结果：{text or '（空）'}")

            if not text or lang not in ("zh", "th"):
                self.stats[channel]["识别为空"] += 1
                self._report()
                continue

            self._lang_of[channel] = lang
            self._lang_ts[channel] = ts

            # 纯语气词（嗯/啊/哦/อืม…）连字幕都不出。它们零信息量，
            # 却要占一行、等几秒、花一次翻译调用——实测"嗯。"一场会出现 13 次。
            if self.drop_fillers and quick.is_filler(text):
                self.stats[channel]["语气词丢弃"] += 1
                self._report()
                continue

            if self._is_duplicate(channel, text, ts):
                self.stats[channel]["重复丢弃"] += 1
                continue
            self.stats[channel]["出字幕"] += 1
            self._report()

            item = {
                "id": next(self._ids),
                "channel": channel,
                "speaker": CHANNEL_LABEL[channel],
                "src_lang": lang,
                "tgt_lang": "th" if lang == "zh" else "zh",
                "text": text,
                "translation": "",
                "ts": ts,
                "seconds": round(len(audio) / 16000, 2),
                "lag": round(time.time() - ts, 1),   # 说完到出字幕用了多久
                "cut": partial_cut,                  # 话太长被切开的半句
            }
            # "好的""收到""ครับ""ไม่มี"这类话查表直接给译文：
            # 0 毫秒、不占一次调用、而且同一句话每次译法都一样。
            # 这类短句实测占全部字幕的 24%，以前每条都要等 1~2 秒。
            fast = (quick.lookup(text, lang, item["tgt_lang"], self.gender)
                    if self.quick_replies else None)
            if fast:
                item["translation"] = fast
                item["quick"] = True
                self.stats[channel]["秒回"] += 1

            self.history.append(item)
            self.history[:] = self.history[-200:]
            self._remember(channel, text, item["id"], ts)

            # item 里已经带上译文时，网页那边会直接显示，不会闪一下"翻译中"
            await self._push({"type": "utterance", "item": item})

            # 对方这条到了，回头看我方刚才那条是不是它的回声
            gone = self.late_echo(item)
            if gone:
                self.stats["me"]["回声丢弃"] += 1
                self.history[:] = [h for h in self.history if h["id"] != gone["id"]]
                self._recent = [r for r in self._recent if r[3] != gone["id"]]
                await self._push({"type": "retract", "id": gone["id"],
                                  "reason": "回声"})
                asyncio.get_running_loop().run_in_executor(
                    None, self.transcript.remove, gone["id"])
            if fast:
                asyncio.get_running_loop().run_in_executor(
                    None, self.transcript.add, dict(item))
            else:
                asyncio.create_task(self._translate(item))

    async def _translate(self, item: dict):
        if self.translator is None:
            who = ("OpenAI" if str(self.cfg.get("translate_backend",
                                                "anthropic")).lower() == "openai"
                   else "Anthropic")
            item["translation"] = f"⚠ 未配置 {who} API Key，无法翻译"
            await self._push({
                "type": "translation", "id": item["id"],
                "translation": item["translation"], "tgt_lang": item["tgt_lang"],
            })
            return
        async def on_delta(chunk: str):
            # 边翻边推，字幕像打字一样长出来，不用干等整句翻完
            await self._push({"type": "delta", "id": item["id"], "text": chunk})

        try:
            ctx = [h for h in self.history if h["id"] != item["id"] and h["text"]]
            out = await self.translator.translate_stream(
                item["text"], item["src_lang"], item["tgt_lang"], ctx, on_delta,
                fragment=item.get("cut", False),
            )
        except Exception as e:
            print(f"[翻译] 出错：{e}")
            out = friendly_error(e, self.cfg.get("translate_backend",
                                                 "anthropic"))

        item["translation"] = out
        # 落盘放到线程里做。写文件虽然只有几毫秒，但它在翻译的必经之路上，
        # 而这条路上每一毫秒都直接加在用户看到字幕的时间里。
        asyncio.get_running_loop().run_in_executor(None, self.transcript.add, dict(item))
        # 收尾时把整句发一遍：流式过程中可能带了前缀或引号，这里统一修正
        await self._push({
            "type": "translation",
            "id": item["id"],
            "translation": out,
            "tgt_lang": item["tgt_lang"],
        })

    # ================================================================
    async def _push(self, event: dict):
        try:
            await self.broadcast(event)
        except Exception:
            pass

    def _is_duplicate(self, channel: str, text: str, now: float = 0.0) -> bool:
        """回声的兜底：这一句和对方刚说过的话几乎一样吗？

        主力是 echo.py 的声学判定，这里只补漏。两点和旧版不同：

        1) **只丢我方的。** 旧版丢"后到的那条"，但实测两路先后不固定，
           有 -0.8 秒的（麦克风那条先到）。丢后到的等于留下被音箱弄糊的
           那条、丢掉环回录到的清楚那条，还顺带把说话人标错了。
           对方那一路是从系统直接抓的，永远比麦克风干净，不能丢。

        2) 阈值 0.65、窗口 10 秒。真实会议里同一句话被两路各录一遍时，
           文字相似度只有 0.51~0.76——麦克风听到的是被音箱弄糊的残片，
           识别出来的字本来就对不齐。0.82 一条拦不住，0.72 只拦得住 2/8。
           再往下调到 0.58 能拦住 6/8，但会误伤真实对话里"两人互相复述"
           的句子（实测另一场会误伤 2 条），不划算。

        这里只处理"对方先到"的情况。"我方先到"（实测 8 条里占 4 条）
        结构上拦不住——判断时对方那条还没出现——由 late_echo() 事后撤回。
        """
        # 按**说话时刻**比，不能按处理时刻。两条字幕的处理间隔里含着各自的
        # 识别耗时（实测 0.5~7 秒不等），用 time.time() 会让时间窗忽宽忽窄。
        now = float(now if now else time.time())
        window = float(self.cfg.get("dedupe_window_sec", 10.0))
        ratio = float(self.cfg.get("dedupe_ratio", 0.65))
        self._recent = [r for r in self._recent if now - r[0] < window]
        if channel == "me":
            for _, ch, prev, _id in self._recent:
                if ch == "me":
                    continue
                if difflib.SequenceMatcher(None, prev, text).ratio() > ratio:
                    print(f"[去重] 这句和对方刚说的几乎一样，按回声丢弃：{text}")
                    return True
        return False

    def late_echo(self, item: dict) -> dict | None:
        """对方这条出来之后，回头看看我方刚才那条是不是它的回声。

        麦克风只听到对方一整句里最响的那几个字，所以那一段音频往往**更短、
        更早收尾**——实测 8 条漏音里有 4 条是麦克风这边先出字幕的。
        等对方那条正式到了，才有东西可比，这时候再撤回。

        只撤我方的，而且只在对方那条更长（= 更完整）时撤。
        """
        if item.get("channel") != "them":
            return None
        text = (item.get("text") or "").strip()
        if len(text) < 6:
            return None
        ratio = float(self.cfg.get("dedupe_ratio", 0.65))
        window = float(self.cfg.get("dedupe_window_sec", 10.0))
        now = float(item.get("ts") or time.time())
        for ts, ch, prev, item_id in reversed(self._recent):
            if ch != "me" or now - ts > window or item_id is None:
                continue
            if len(prev) >= len(text):
                continue            # 我方那条更长，不像是残片，不动它
            if difflib.SequenceMatcher(None, text, prev).ratio() > ratio:
                print(f"[去重] 对方这句到了才看出来，我方刚才那条是回声，撤回：{prev}")
                return {"id": item_id, "text": prev}
        return None

    def _remember(self, channel: str, text: str, item_id: int | None,
                  ts: float = 0.0):
        self._recent.append((float(ts or time.time()), channel, text, item_id))

    # ---- 界面控制 ----
    async def retranslate(self, item_id: int, new_text: str):
        """开会中当场改错字：改完重新翻译这一条。

        顺便在终端打印出「听错的 → 改成的」，方便把它加进 asr_corrections，
        下次就不用再手动改了。
        """
        item = next((h for h in self.history if h["id"] == item_id), None)
        new = (new_text or "").strip()

        # 任何情况下都必须回一条消息，否则界面会永远卡在"重新翻译中"。
        # 之前这里有两条静默 return，就是卡住的原因。
        if item is None:
            await self._push({"type": "translation", "id": item_id,
                              "translation": "⚠ 这条已经太旧，无法重新翻译"})
            return
        if not new:
            await self._push({"type": "translation", "id": item_id,
                              "translation": item.get("translation", "")})
            return

        old = item["text"]
        if old == new:      # 没改动，把原来的译文原样发回去，解除等待
            await self._push({"type": "translation", "id": item_id,
                              "translation": item.get("translation", ""),
                              "tgt_lang": item.get("tgt_lang")})
            return

        item["text"] = new
        item["translation"] = ""
        item["edited"] = True
        print(f"[纠错] {old}\n   →  {new}")
        self._log_correction(old, new)

        await self._push({"type": "edited", "id": item_id, "text": new})
        asyncio.create_task(self._translate(item))

    def _log_correction(self, old: str, new: str):
        """把会上手改的字集中记到一个文件里。

        术语表是长期活儿，每次开会都会冒出几个新的专业词。散在运行日志里
        不好找，集中到一个小文件，攒一阵子一次性补进 asr_corrections /
        glossary 就行。存的是"改之前 → 改之后"的原样，不做任何猜测。
        """
        try:
            from datetime import datetime
            from pathlib import Path

            d = getattr(self.transcript, "dir", None)
            if d is None:
                d = Path(__file__).resolve().parent.parent / str(
                    self.cfg.get("transcript_dir", "会议记录"))
            d.mkdir(parents=True, exist_ok=True)
            path = d / "纠错记录.txt"
            first = not path.exists()
            with open(path, "a", encoding="utf-8") as f:
                if first:
                    f.write("# 开会时手动改过的字。攒一批之后补进 config.json：\n"
                            "#   识别老听错的词 → asr_corrections（确定性替换）\n"
                            "#   译法要统一的词 → glossary（专业词、人名、公司名）\n\n")
                f.write(f"{datetime.now():%Y-%m-%d %H:%M}\n"
                        f"  听成： {old}\n"
                        f"  应为： {new}\n\n")
            print(f"       已记到 {path.name}，攒一批之后一起补进术语表")
        except Exception as e:
            print(f"       （纠错记录写入失败，不影响使用：{e}）")

    async def make_minutes(self, progress=None) -> dict:
        """把这场会的字幕整理成中泰双语纪要，写成 .md 和 .docx。

        用记录文件里的全部内容，而不是内存里的 history（后者只留最近 200 句）。
        """
        from pathlib import Path

        from .minutes import MinutesMaker, load_jsonl, save

        items = []
        jl = getattr(self.transcript, "jsonl", None)
        if jl and Path(jl).exists():
            self.transcript.flush(force=True)
            items = load_jsonl(jl)
        if not items:
            items = [dict(h) for h in self.history]
        items = [i for i in items if (i.get("text") or "").strip()]
        if not items:
            raise RuntimeError("这场会还没有字幕，没什么可整理的。")

        maker = MinutesMaker(self.cfg)
        md = await maker.make(items, progress)

        stem = self.transcript.started.strftime("%Y-%m-%d_%H%M")
        out = self.transcript.dir if self.transcript.enabled else \
            Path(__file__).resolve().parent.parent / str(
                self.cfg.get("transcript_dir", "会议记录"))
        paths = save(md, out, stem)
        print(f"[纪要] 已保存：{paths.get('md')}")
        if paths.get("docx"):
            print(f"[纪要] Word 版：{paths['docx']}")
        return {"markdown": md, **paths}

    def set_paused(self, value: bool) -> dict:
        """暂停/继续。暂停期间完全不产生云端调用。

        会议结束后人还在屋里聊天、放视频，麦克风照收，一句句识别加翻译，
        钱就这么流走了。暂停是最直接的止血办法。

        暂停时要把"在途"的东西全部清干净，否则一按继续，
        半分钟前的半句话会突然蹦出来：
          · 断句器里攒着的半句 → 丢掉
          · 排队等识别的句子   → 丢掉（它们本来也已经过期了）
          · 界面上的"正在识别" → 撤掉
        """
        value = bool(value)
        if value == self.paused:
            return self.public_status()
        self.paused = value

        if value:
            self._paused_at = time.time()
            if self.stream is not None:
                # 暂停还挂着连接就是在烧钱——实时识别按音频秒数计费
                self.stream.cancel_all()
            for seg in self.segmenters.values():
                seg.flush()                       # 丢掉攒了一半的句子
            while True:
                try:
                    self.seg_q.get_nowait()
                except queue.Empty:
                    break
            for ch in self.segmenters:
                self._push_sync({"type": "partial", "channel": ch, "text": ""})
            print("[暂停] 已暂停：不再识别、不再翻译，不产生任何调用")
        else:
            if self._paused_at:
                self.paused_total += time.time() - self._paused_at
                self._paused_at = 0.0
            # 暂停期间攒下的音频是旧的，清掉再开始，免得一上来就冒出半句陈年旧话
            for seg in self.segmenters.values():
                seg.flush()
            self._lang_of.clear()
            print(f"[暂停] 已继续（本场累计暂停 {self.paused_total / 60:.1f} 分钟）")
        return self.public_status()

    def set_muted(self, channel: str, value: bool):
        if channel in self.muted:
            self.muted[channel] = bool(value)

    def clear_history(self):
        self.history.clear()
        self._recent.clear()
