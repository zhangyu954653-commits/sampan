"""语音识别 + 语种判定（只在 中文 / 泰语 之间二选一）。

判定顺序：
  1) Whisper 的语种概率里，取 zh 和 th 谁高
  2) 识别出文字后，按字符所属文字系统（泰文字母 vs 汉字）二次校正 —— 这一步最准
"""
from __future__ import annotations

import io
import os
import re
import wave

import numpy as np

SR = 16000

_THAI = re.compile(r"[฀-๿]")
_CJK = re.compile(r"[一-鿿㐀-䶿]")

# Whisper（本地和云端都一样）经常输出繁体，统一转成简体
try:
    from opencc import OpenCC

    _T2S = OpenCC("t2s")
except Exception:
    _T2S = None


def to_simplified(text: str) -> str:
    if _T2S is None or not text:
        return text
    try:
        return _T2S.convert(text)
    except Exception:
        return text

# Whisper 在静音/噪音上常见的幻觉输出，直接丢掉。
# 注意要同时列繁体：Whisper 的中文输出经常是繁体，只写简体会漏掉。
_HALLUCINATION = {
    "谢谢观看", "谢谢大家", "感谢观看", "下次再见", "拜拜", "拜托", "字幕组",
    "字幕由amara.org社区提供", "字幕志愿者", "由 Amara.org 社区提供的字幕",
    "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目", "未经许可 不得翻唱或使用",
    "謝謝觀看", "謝謝大家", "感謝觀看", "下次再見", "拜託", "字幕組",
    "字幕由Amara.org社群提供", "請不吝點贊 訂閱 轉發 打賞支持明鏡與點點欄目",
    "ขอบคุณครับ", "ขอบคุณค่ะ", "สวัสดีครับ", "สวัสดีค่ะ",
    "Thank you.", "Thanks for watching!", "you", "Bye.",
}

# 注意：不要用 initial_prompt 去引导简体。静音片段上 Whisper 会直接把提示词
# 原样念出来，变成假字幕。繁体问题交给上面的 opencc 处理，确定性更强。


def detect_script(text: str) -> str | None:
    """按文字系统判断语种：th / zh / None（判断不了）。"""
    thai = len(_THAI.findall(text))
    cjk = len(_CJK.findall(text))
    if thai == 0 and cjk == 0:
        return None
    return "th" if thai >= cjk else "zh"


def _to_wav_bytes(audio: np.ndarray) -> bytes:
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return bio.getvalue()


# 识别偶尔会陷入复读：7 秒音频吐出 110 个"谢谢"。这种内容送去翻译，
# 模型要老老实实翻 110 遍，实测一条卡了 40.9 秒，还把相邻的句子拖到超时。
# 连续重复 4 次以上就压成 2 次——语义不丢，代价没了。
_REPEAT = re.compile(r"(.{1,8}?)\1{3,}")


def _collapse_repeats(text: str) -> str:
    prev = None
    while prev != text:                # 可能有嵌套重复，压到不动为止
        prev = text
        text = _REPEAT.sub(lambda m: m.group(1) * 2, text)
    return text


def _clean(text: str) -> str:
    text = to_simplified((text or "").strip())
    if not text:
        return ""
    stripped = text.strip(" 。.!！?？…~～,，")
    if stripped in _HALLUCINATION or text in _HALLUCINATION:
        return ""
    # 既没有泰文也没有汉字，且很短 —— 基本是噪音
    if detect_script(text) is None and len(stripped) < 6:
        return ""
    # "啊啊啊啊啊" 这种重复噪音。必须在压缩之前判——压完变成"啊啊"就
    # 够不着长度阈值了；而阈值又不能降到 2，否则"谢谢""拜拜"会被误杀。
    if len(set(stripped)) <= 1 and len(stripped) >= 3:
        return ""
    return _collapse_repeats(text)


class ASR:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.backend = cfg.get("backend", "local")
        self.beam_size = int(cfg.get("beam_size", 1))
        self.verify = bool(cfg.get("verify_language", True))
        self.bias = float(cfg.get("language_bias_strength", 2.5))
        # 两种语言的概率都低于这个值 = 检测器在噪声里比大小，结果不可信
        self.noise_floor = float(cfg.get("lang_noise_floor", 0.25))
        self.retry_below = float(cfg.get("verify_logprob", -0.95))
        self.switch_margin = float(cfg.get("verify_margin", 0.15))
        self.partial_on = bool(cfg.get("partial_results", True))
        self._tencent = None
        self._tencent_fail = 0
        self.both_engines = bool(cfg.get("both_engines", False))
        # 判语种（本地 CPU）和识别（云端网络）互不干扰，正好并行
        from concurrent.futures import ThreadPoolExecutor

        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="asr")
        # 纠错表的处理顺序：**保护条目优先，然后按长度倒序**。
        #
        # 把一个词映射到它自己（"香烟厂": "香烟厂"）表示"这个词别动"。
        # 这类条目必须排在所有替换规则前面，光靠长度排不住：
        # 实测「烟厂的→验厂的」和保护条目「香烟厂」都是 3 个字，
        # 谁先谁后取决于写进配置的顺序，结果「香烟厂的订单」
        # 被改成了「香验厂的订单」。
        fixes = cfg.get("asr_corrections") or {}
        self.fixes = sorted(fixes.items(),
                            key=lambda kv: (kv[0] != kv[1], -len(kv[0])))
        self._model = None
        self._detector = None
        self._partial_model = None
        self._openai = None

    # ------------------------------------------------------------------
    def load(self) -> str:
        # 腾讯云：只负责"把音频转成文字"这一步，语种判定和中间结果仍然用本地
        # tiny 模型（快、免费、实测够准）。云端出问题时自动退回本地 small。
        if self.backend == "tencent":
            from .tencent_asr import TencentASR

            self._tencent = TencentASR(self.cfg)
            try:
                tag = self._tencent.load()
            except Exception as e:
                print(f"[识别] 腾讯云不可用（{e}），改用本地模型。")
                self._tencent = None
            else:
                local = self._load_local(need_main=True)
                return f"{tag}　+ 本地兜底 {local}"

        if self.backend == "openai":
            key = self.cfg.get("openai_api_key") or ""
            try:
                if not key:
                    raise RuntimeError("没有配置 OpenAI API Key")
                from openai import OpenAI

                self._openai = OpenAI(api_key=key)
                return f"OpenAI {self.cfg.get('openai_model', 'whisper-1')}"
            except Exception as e:
                # 云端不可用时自动退回本地，保证程序还能用
                print(f"[识别] 云端 Whisper 不可用（{e}），自动改用本地模型。")
                self.backend = "local"

        return self._load_local()

    def _load_local(self, need_main: bool = True) -> str:
        from faster_whisper import WhisperModel

        device = self.cfg.get("device", "auto")
        compute = self.cfg.get("compute_type", "auto")
        size = self.cfg.get("model_size", "auto")

        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"
        if size == "auto":
            size = "medium" if device == "cuda" else "small"
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"

        # ctranslate2 在 CPU 上默认只开 4 线程，多核机器等于闲着一半。
        threads = int(self.cfg.get("cpu_threads", 0))
        if threads <= 0:
            threads = max(4, min(8, (os.cpu_count() or 4) - 4))

        try:
            self._model = WhisperModel(size, device=device, compute_type=compute,
                                       cpu_threads=threads)
        except Exception as e:
            print(f"[识别] {device}/{compute} 加载失败（{e}），改用 cpu/int8。")
            device, compute = "cpu", "int8"
            self._model = WhisperModel(size, device=device, compute_type=compute,
                                       cpu_threads=threads)

        # 专职判语种的小模型。实测：small 自己用 language=None 判语种要 7.0 秒，
        # 而 tiny 判完(0.5s)再让 small 按结果转写(3.9s)只要 4.4 秒，准确率一样。
        det = self.cfg.get("lang_detect_model", "tiny")
        if det and det != size:
            try:
                self._detector = WhisperModel(det, device=device,
                                              compute_type="int8", cpu_threads=threads)
            except Exception as e:
                print(f"[识别] 语种检测模型 {det} 加载失败（{e}），改用主模型自动判定。")
                self._detector = None

        # 边说边显示用的模型。单独一份实例，避免和语种检测抢同一个模型——
        # ctranslate2 的模型不能被两个线程同时调用。
        if self.partial_on:
            name = self.cfg.get("partial_model", "tiny")
            try:
                self._partial_model = WhisperModel(name, device=device,
                                                   compute_type="int8",
                                                   cpu_threads=max(2, threads // 2))
            except Exception as e:
                print(f"[识别] 中间结果模型 {name} 加载失败（{e}），已关闭边说边显示。")
                self._partial_model = None

        extra = f"/{threads}线程" if device == "cpu" else ""
        tag = f" + {det} 判语种" if self._detector is not None else ""
        return f"faster-whisper {size} ({device}/{compute}{extra}){tag}"

    # ------------------------------------------------------------------
    def transcribe(self, audio: np.ndarray, prefer: str | None = None,
                   known: str | None = None, fast: bool = False,
                   locked: str | None = None) -> tuple[str, str | None]:
        """返回 (文本, 语种)。语种为 'zh' / 'th' / None。

        prefer: 这一路更可能说的语言（'zh'/'th'/None），用于纠正语种误判。
        known:  同一段发言里已经判过的语种。人不会说到一半换语言，
                所以整段只判一次就够，后面每句省掉 0.5 秒的检测。
        fast:   后面还排着队，赶时间——跳过"换语言重试"（那一步要 4~7 秒）。
                宁可这一句语种判错，也不能让后面几句全部超时被丢掉。
        """
        try:
            # 锁定了就彻底跳过语种判定——用户明确说了这一路只说这种语言，
            # 再去判既浪费时间，判错了还会把锁定覆盖掉。
            if locked in ("zh", "th") and self._tencent is not None:
                got = self._cloud(audio, locked)
                if got is not None:
                    # 锁定就是锁定：_cloud 末尾会按字形再判一次语种，
                    # 那一步会把用户明确指定的语言覆盖掉（实测出现过）。
                    return got[0], locked
                got = None      # 云端挂了，往下走本地
            if self.backend == "openai":
                # 把已知语种传下去：省掉模型自己判语种的那一步，也更准
                got = self._transcribe_openai(
                    audio, locked or known or prefer)
            elif locked in ("zh", "th"):
                got = self._transcribe_local(audio, prefer, locked, fast)
            else:
                got = self._transcribe_local(audio, prefer, known, fast)
        except Exception as e:
            print(f"[识别] 出错：{e}")
            return "", None
        # 一定要返回元组。上层是 `text, lang = transcribe(...)`，
        # 返回 None 会在解包时抛异常，表现成"这句话凭空消失了"。
        return got if isinstance(got, tuple) else ("", None)

    def _run(self, audio: np.ndarray, lang: str | None) -> tuple[str, float, str, float]:
        """转写一次。lang=None 时让 whisper 自己判语种——它会复用同一次编码结果，
        比先单独调 detect_language 再转写省掉整整一次编码（实测省约 3.3 秒）。

        返回 (文本, 平均置信度, 判定的语种, 语种置信度)。
        """
        segments, info = self._model.transcribe(
            audio,
            language=lang,
            task="transcribe",
            beam_size=self.beam_size,
            temperature=0.0,
            condition_on_previous_text=False,  # 防止上一句污染下一句、陷入复读
            no_speech_threshold=0.6,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        # 逐段按置信度过滤：静音/噪音上模型会硬编出一句话，
        # 这种片段的 no_speech_prob 很高、avg_logprob 很低。
        texts, logps = [], []
        for s in segments:
            if getattr(s, "no_speech_prob", 0.0) >= 0.6:
                continue
            lp = float(getattr(s, "avg_logprob", -2.0))
            if lp <= -1.0:
                continue
            texts.append(s.text)
            logps.append(lp)

        text = _clean("".join(texts))
        score = sum(logps) / len(logps) if logps else -9.0
        got = lang or getattr(info, "language", None)
        prob = float(getattr(info, "language_probability", 1.0) or 1.0)
        return text, score, got, prob

    def _cloud(self, audio: np.ndarray, lang: str) -> tuple[str, str | None]:
        """把一段音频交给腾讯云。失败多次就永久退回本地。"""
        try:
            raw = self._tencent.transcribe(audio, lang)
        except Exception:
            self._tencent_fail += 1
            if self._tencent_fail >= 3:
                print("[识别] 腾讯云连续失败 3 次，本次会议改用本地模型。")
                self._tencent = None
            return None                  # None = 没拿到结果，让调用方决定怎么办
        self._tencent_fail = 0
        text = _clean(raw)
        if not text:
            return "", None              # 云端认为这段没内容，那就是没内容
        return self.apply_fixes(text), detect_script(text) or lang

    def _cloud_parallel(self, audio: np.ndarray, prefer: str | None,
                        guess: str | None) -> tuple[str, str | None] | None:
        """判语种和识别同时开跑，不要串着等。

        串行是：判语种 0.42s → 识别 0.5s = 0.92s。
        并行是：两件事一起做，取较慢的那个 ≈ 0.5s。
        猜错了才补一次识别——而猜测来自声道先验和上一句，命中率很高。
        """
        other = "zh" if guess == "th" else "th"
        det = self._pool.submit(self._pick_language, audio, prefer)
        asr = self._pool.submit(self._cloud, audio, guess)
        # both_engines：两种语言都识别一遍，判语种出来直接取对应结果，
        # 永远不需要第二次往返。代价是云端调用翻倍。
        alt = self._pool.submit(self._cloud, audio, other) if self.both_engines else None

        lang = det.result()[0]
        got = asr.result()
        # 检测器没装或判定失败时返回 None。此时只能信猜测——
        # 绝不能把 None 当语种传下去，那样云端会拿不到引擎名而返回空，
        # 表现是"每句话都没有字幕"，还查不出原因。
        if lang not in ("zh", "th"):
            lang = guess

        if lang == guess:
            if alt is not None:
                alt.result()             # 收掉，别留悬空的 future
            return got                   # 猜对了，识别结果直接可用
        if alt is not None:
            other_got = alt.result()
            if other_got is not None:
                return other_got         # 猜错了，但另一种早就跑完了
        if got is None:
            return None                  # 云端挂了，让调用方退回本地
        # 猜错了又没开双引擎：用判出来的语种重来一次
        return self._cloud(audio, lang)

    def _transcribe_local(self, audio: np.ndarray, prefer: str | None = None,
                          known: str | None = None,
                          fast: bool = False) -> tuple[str, str | None]:
        audio = audio.astype(np.float32, copy=False)

        # 云端路径：判语种和识别并行。
        # known（上一句的语种）在这里只当"猜测"用，不当结论——
        # 多人会议里同一路会混着中文和泰语，沿用上一句会判错。
        if self._tencent is not None:
            guess = known if known in ("zh", "th") else (
                prefer if prefer in ("zh", "th") else "zh")
            got = self._cloud_parallel(audio, prefer, guess)
            if got is not None:
                return got
            # 云端这次没成：直接落到本地模型，不要再试一次云端
            # （刚失败过，重试只会再等一个超时，还可能返回 None 让上层解包崩）

        if known in ("zh", "th"):
            lang, prob = known, 1.0      # 本地路径才沿用，省一次 CPU 检测
        else:
            lang, prob = self._pick_language(audio, prefer)

        text, score, got, auto_prob = self._run(audio, lang)

        if lang is None:            # 没有检测模型时由主模型自己判的
            lang = "zh" if got in ("zh", "yue") else got
            prob = auto_prob

        # 第一遍就没出东西 = 这段是噪音，直接丢。绝不能再跑一遍——
        # 噪音的置信度天然很低，会触发重试，白白多花几秒钟拖慢整条流水线。
        if not text:
            return "", None

        # 只在转写出了内容、但明显不对劲时才换另一种语言重来一次。
        # 每多一次要多花 4~7 秒，所以阈值很保守：实测泰语音频被强制按中文
        # 转写时 logprob 会掉到 -1.2 以下，正常转写在 -0.4 上下。
        if self.verify and not fast and lang in ("zh", "th") and score < self.retry_below:
            other = "zh" if lang == "th" else "th"
            alt_text, alt_score, _, _ = self._run(audio, other)
            if alt_text and alt_score > score + self.switch_margin:
                text, lang = alt_text, other
        return self.apply_fixes(text), \
            detect_script(text) or (lang if lang in ("zh", "th") else None)

    def apply_fixes(self, text: str) -> str:
        """按纠错表把老被听错的专业词换回来。确定性修正，不靠模型猜。

        换完的部分要**锁住**，不能再被后面的规则咬一口。
        表按"听错的样子"长度倒序处理，所以长词先换；但换完的结果留在原文里，
        后面的短词规则还会继续在上面匹配。实测踩过：
            表里有「烟厂→验厂」，原文「香烟厂」被改成「香验厂」——
            长的那条本该保护它，可保护不住。
        所以这里换成占位符，全部换完再还原。
        """
        if not self.fixes:
            return text
        keep: list[str] = []
        for wrong, right in self.fixes:          # 已按长度倒序
            if not wrong or wrong not in text:
                continue
            # \x00 不会出现在识别结果里，拿它当占位符最安全
            token = f"\x00{len(keep)}\x00"
            keep.append(right)
            text = text.replace(wrong, token)
        for i, right in enumerate(keep):
            text = text.replace(f"\x00{i}\x00", right)
        return text

    def partial_language(self, audio: np.ndarray, prefer: str | None) -> str | None:
        """给中间结果判一次语种。每句话开头调一次就够，之后复用。

        不能直接拿声道先验去强制——对方那一路先验是泰语，他一说中文，
        中间结果就会变成泰文乱码。
        """
        if self._partial_model is None:
            return prefer
        try:
            _, _, probs = self._partial_model.detect_language(audio)
        except Exception:
            return prefer
        table = dict(probs) if not isinstance(probs, dict) else probs
        p = {
            "zh": float(table.get("zh", 0.0)) + float(table.get("yue", 0.0)),
            "th": float(table.get("th", 0.0)),
        }
        if prefer in ("zh", "th") and self.bias > 1.0:
            p[prefer] *= self.bias
        return "th" if p["th"] > p["zh"] else "zh"

    def partial(self, audio: np.ndarray, lang: str | None) -> str:
        """说话过程中的中间结果：快但粗，只为让人看到"正在听"。

        不做幻觉过滤——中间结果本来就会被后面的内容不断改写。
        """
        if self._partial_model is None:
            return ""
        try:
            segments, _ = self._partial_model.transcribe(
                audio.astype(np.float32, copy=False),
                language=lang if lang in ("zh", "th") else None,
                task="transcribe", beam_size=1, temperature=0.0,
                condition_on_previous_text=False,
                no_speech_threshold=0.6, vad_filter=False,
            )
            return to_simplified("".join(s.text for s in segments).strip())
        except Exception:
            return ""

    def _pick_language(self, audio: np.ndarray,
                       prefer: str | None) -> tuple[str | None, float]:
        """用小模型判语种，再按声道先验修正。返回 (语种, 把握)。

        返回 None 表示没有检测模型，交给主模型在转写时自己判。
        """
        if self._detector is None:
            return None, 0.0
        try:
            _, _, probs = self._detector.detect_language(audio)
        except Exception:
            return None, 0.0

        table = dict(probs) if not isinstance(probs, dict) else probs
        p = {
            "zh": float(table.get("zh", 0.0)) + float(table.get("yue", 0.0)),
            "th": float(table.get("th", 0.0)),
        }

        # 短音频上这两个概率都在 0.01 量级（实测泰语短语 p(zh)=0.007 / p(th)=0.015），
        # 也就是绝大部分概率都给了别的语种，中泰之间等于在噪声里比大小。
        # 这种时候比较结果没有意义，直接信声道先验。
        if prefer in ("zh", "th") and max(p["zh"], p["th"]) < self.noise_floor:
            return prefer, 0.0

        # 有把握的时候才用加权比较。先验只是加权，对方改说中文照样认得出来。
        if prefer in ("zh", "th") and self.bias > 1.0:
            p[prefer] *= self.bias

        lang = "th" if p["th"] > p["zh"] else "zh"
        total = p["zh"] + p["th"]
        return lang, (p[lang] / total if total > 0 else 0.0)

    def _transcribe_openai(self, audio: np.ndarray,
                           lang_hint: str | None = None) -> tuple[str, str | None]:
        """OpenAI 的语音识别。

        response_format 必须按模型选：**gpt-4o-transcribe / gpt-4o-mini-transcribe
        不支持 verbose_json**，只认 json 和 text。原来这里写死 verbose_json，
        填新模型会直接报错——而新模型正是快的那个，whisper-1 是老的慢的。
        """
        model = str(self.cfg.get("openai_model", "gpt-4o-mini-transcribe"))
        data = _to_wav_bytes(audio)
        f = io.BytesIO(data)
        f.name = "chunk.wav"

        kwargs = {"model": model, "file": f}
        # verbose_json 能多拿一个 language 字段，但只有 whisper-1 支持
        kwargs["response_format"] = "verbose_json" if "whisper" in model else "json"
        if lang_hint in ("zh", "th"):
            # 告诉它说的是哪种语言：省掉它自己判语种，也不会把泰语听成别的
            kwargs["language"] = lang_hint
        hint = self.cfg.get("openai_prompt") or ""
        if hint:
            # 专业词、人名喂给它当提示，识别准确率会明显提高
            kwargs["prompt"] = hint[:900]

        try:
            resp = self._openai.audio.transcriptions.create(**kwargs)
        except Exception as e:
            print(f"[识别] OpenAI 出错：{e}")
            return "", None

        text = _clean(getattr(resp, "text", "") or "")
        if not text:
            return "", None
        text = self.apply_fixes(text)
        lang = detect_script(text)
        if lang is None:
            raw = (getattr(resp, "language", "") or "").lower()
            if raw.startswith("chin") or raw == "zh":
                lang = "zh"
            elif raw.startswith("thai") or raw == "th":
                lang = "th"
        return text, lang or lang_hint


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False
