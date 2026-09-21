"""纯函数自测：语种判定 / 噪音过滤 / 译文清洗 / 断句。不联网、不需要 Key。"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path as _Path

# 这个脚本在「工具」子目录里，import translator 要指到上一层
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import numpy as np

from translator.asr import _clean, detect_script
from translator.translate import _strip
from translator.vad import Segmenter

fails = []


def check(name, got, want):
    ok = got == want
    print(f"  {'✓' if ok else '✗'} {name}\n      got={got!r} want={want!r}" if not ok
          else f"  ✓ {name}")
    if not ok:
        fails.append(name)


print("\n【语种判定】")
check("纯中文", detect_script("我们今天讨论发货计划"), "zh")
check("纯泰文", detect_script("สวัสดีครับ ยินดีที่ได้รู้จัก"), "th")
check("中文夹数字", detect_script("大概有3000件现货"), "zh")
check("泰文夹数字", detect_script("ประมาณ 3,000 ชิ้น"), "th")
check("中泰混合偏泰", detect_script("ตกลงครับ 好"), "th")
check("纯英文→判不出", detect_script("hello world"), None)
check("空串→判不出", detect_script(""), None)

print("\n【噪音 / 幻觉过滤】")
check("正常中文保留", _clean("我们今天讨论发货计划"), "我们今天讨论发货计划")
check("正常泰文保留", _clean("ขอบคุณสำหรับข้อมูลครับ"), "ขอบคุณสำหรับข้อมูลครับ")
check("幻觉字幕组", _clean("谢谢观看"), "")
check("幻觉带标点", _clean("谢谢观看。"), "")
check("英文短噪音", _clean("you"), "")
check("重复字噪音", _clean("啊啊啊啊"), "")
check("空白", _clean("   "), "")

print("\n【译文清洗】")
check("去前缀", _strip("译文：สวัสดีครับ"), "สวัสดีครับ")
check("去英文前缀", _strip("Translation: hello"), "hello")
check("去中文引号", _strip("「我周四发给你」"), "我周四发给你")
check("去直角引号", _strip('"好的"'), "好的")
check("正常不动", _strip("ได้ครับ ผมจะส่งให้"), "ได้ครับ ผมจะส่งให้")

print("\n【断句 Segmenter】")
cfg = {"aggressiveness": 2, "start_frames": 3, "end_frames": 10, "preroll_ms": 300,
       "max_seconds": 8.0, "min_seconds": 0.25, "rms_threshold": 0.006}
sr = 16000
rng = np.random.default_rng(0)


def tone(sec):
    """模拟人声：带噪声的复合音，能通过 WebRTC VAD。"""
    t = np.arange(int(sr * sec)) / sr
    sig = (0.35 * np.sin(2 * np.pi * 180 * t)
           + 0.20 * np.sin(2 * np.pi * 420 * t)
           + 0.12 * np.sin(2 * np.pi * 900 * t))
    sig *= 1 + 0.5 * np.sin(2 * np.pi * 4 * t)          # 模拟音节起伏
    return (sig + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def silence(sec):
    return np.zeros(int(sr * sec), dtype=np.float32)


def feed(seg, chunks, t0=0.0, block=0.09):
    """按真实节奏喂：音频是一小块一小块到的，不是整段砸进来。

    一次性灌一大段会让"断流兜底"误判成卡住（时间戳跳了一大截），
    那是测试写法的问题，不是断句器的问题。
    """
    out, t = [], t0
    for chunk in chunks:
        n = int(sr * block)
        for i in range(0, len(chunk), n):
            piece = chunk[i:i + n]
            out += seg.push(piece, t)
            t += len(piece) / sr
    return out


seg = Segmenter(cfg)
out = feed(seg, (silence(0.3), tone(1.2), silence(0.8), tone(1.0), silence(0.8)))
check("两段语音断出两句", len(out), 2)
if len(out) == 2:
    check("第一句长度合理(1.2s±0.6)", 0.6 < len(out[0][0]) / sr < 2.0, True)
    check("带回了开始时刻", out[0][1] < out[1][1], True)

# 断流兜底：说到一半流断了（系统突然不出声了），靠挂钟超时收尾
seg2 = Segmenter(cfg)
out2 = feed(seg2, (silence(0.3), tone(1.2)))
check("断流前未收尾", len(out2), 0)
out2 += seg2.tick(1.5 + 0.4)        # 模拟 0.4 秒没有新数据
check("tick 超时后收尾", len(out2), 1)

# 过短的声音应当被丢弃
seg3 = Segmenter(cfg)
out3 = feed(seg3, (silence(0.2), tone(0.15), silence(0.8)))
check("过短片段被丢弃", len(out3), 0)

# 长句强切：说太久必须切一刀，而且要切在停顿处而不是词中间。
# 连续说 2.4 秒 → 停 0.25 秒（不够长，不会自然断句）→ 再说 1.5 秒
seg4 = Segmenter(dict(cfg, max_seconds=3.0))
out4 = feed(seg4, (silence(0.2), tone(2.4), silence(0.25), tone(1.5)))
check("超长句被强制切开", len(out4) >= 1, True)
if out4:
    check("切出来的片段标记为「半句」", out4[0][2], True)
    cut_at = len(out4[0][0]) / sr
    # 停顿在 2.6s 附近，切点应该落在它周围而不是硬切在 3.0s
    check(f"切在停顿处而非硬切(实际 {cut_at:.2f}s)", 2.3 < cut_at < 3.05, True)

# 正常说完的句子不该被标记成半句
check("正常结束的句子不标记半句", out[0][2], False)

# ================================================================
# 回归测试：2026-09-15 代码审查发现的 bug
# 这几个都是"静默失效"型的——不报错，只让字幕凭空消失，必须有测试钉住。
# 需要加载模型，所以放在最后；不需要联网。
# ================================================================
import shutil
import tempfile
import threading
import time as _time
from pathlib import Path

from translator.asr import ASR
from translator.config import load_config
from translator.transcript import Transcript

_cfg = load_config()
_TONE = (0.3 * np.sin(2 * np.pi * 200 * np.arange(sr * 2) / sr)).astype(np.float32)

print("\n【复读压缩】")
# 来自 2026-09-16 那场真实会议：7 秒音频被识别成 110 个"谢谢"，
# 翻译卡了 40.9 秒还把相邻句子拖到超时
from translator.asr import _collapse_repeats

check("110 次复读被压掉", len(_collapse_repeats("哎呀 " + "谢谢 " * 110)) < 20, True)
check("嵌套复读也能压", _collapse_repeats("没有没有有有有有有有。"), "没有没有有。")
# 正常的叠词、三连绝不能误伤
for _s in ("对对对，我也正在想这事儿呢", "好的好的王总啊，您稍等啊", "是的是的",
           "这批纸托盘有问题的大概120件"):
    check(f"不误伤「{_s[:8]}」", _collapse_repeats(_s), _s)

print("\n【识别层兜底】")
_asr = ASR(dict(_cfg["asr"], backend="local"))
_asr.load()

# 语种检测模型加载失败时，绝不能把 None 当语种传下去——
# 那样云端拿不到引擎名会返回空，表现是"每句话都没字幕"且查不出原因
_asr._detector = None
_t, _l = _asr.transcribe(_TONE, "zh", None)
check("检测器挂掉仍返回可用语种", _l in ("zh", "th") or _t == "", True)

# 上层是 `text, lang = transcribe(...)`，返回 None 会在解包时抛异常
_asr._transcribe_local = lambda *a, **k: None
check("内部返回 None 被兜住", isinstance(_asr.transcribe(_TONE, "zh", None), tuple), True)


def _boom(*a, **k):
    raise ValueError("模拟内部异常")


_asr._transcribe_local = _boom
check("内部抛异常被兜住", isinstance(_asr.transcribe(_TONE, "zh", None), tuple), True)

print("\n【会议记录并发写】")
# 多句话会并发翻译完成，add() 从不同线程进来，没锁会丢记录
_tmp = Path(tempfile.mkdtemp())
_ts = Transcript({"save_transcript": True, "transcript_dir": "rec"}, _tmp)
_ts._min_interval = 0                      # 每次都写盘，最大化竞争


def _writer(lo, hi):
    for i in range(lo, hi):
        _ts.add({"id": i, "channel": "me", "speaker": "我方", "src_lang": "zh",
                 "tgt_lang": "th", "text": f"第{i}句", "translation": f"แปล{i}",
                 "ts": _time.time()})


_threads = [threading.Thread(target=_writer, args=(i * 10, i * 10 + 10))
            for i in range(6)]
for _th in _threads:
    _th.start()
for _th in _threads:
    _th.join()
_ts.close()
check("60 条并发写入无丢失无重复",
      sorted(x["id"] for x in _ts.items) == list(range(60)), True)
_jl = list((_tmp / "rec").glob("*.jsonl"))
check("jsonl 行数正确",
      bool(_jl) and len(_jl[0].read_text(encoding="utf-8").strip().split("\n")) == 60,
      True)
shutil.rmtree(_tmp, ignore_errors=True)

print("\n【口头语：该丢的丢、该留的留】")
from translator import quick

for _t in ("嗯", "嗯。", "嗯嗯。", "啊啊啊", "哦哦。", "呃", "唔", "诶",
           "อืม", "อืมๆ", "เอ่อ", "uh", "Hmm"):
    check(f"丢弃 {_t!r}", quick.is_filler(_t), True)
# 语气词开头但有内容的句子绝不能丢——这类误伤最伤人
for _t in ("嗯，这个我们再确认一下", "啊，那个报告改好了吗", "哦对了还有一件事",
           "好的", "收到", "没有", "对", "ครับ", "ไม่มี", "十月二号验厂"):
    check(f"保留 {_t!r}", quick.is_filler(_t), False)

print("\n【口头语：查表秒回】")
check("好的 → 泰语", bool(quick.lookup("好的。", "zh", "th")), True)
check("收到 → 泰语", quick.lookup("收到", "zh", "th"), "รับทราบครับ")
check("没有 → 泰语", quick.lookup("没有", "zh", "th"), "ไม่มีครับ")
check("女性用 ค่ะ", quick.lookup("好的", "zh", "th", "female"), "ได้ค่ะ")
check("未填性别默认 ครับ", quick.lookup("好的", "zh", "th", ""), "ได้ครับ")
check("ไม่มีครับ → 中文", quick.lookup("ไม่มีครับ", "th", "zh"), "没有")
check("ครับ 不被剥空", quick.lookup("ครับ", "th", "zh"), "好的")
# 泰语字符数天然比中文多，长度上限分语种定，否则常用短句全被挡在门外
check("เข้าใจแล้วครับ 不被长度挡住",
      quick.lookup("เข้าใจแล้วครับ", "th", "zh"), "明白了")
check("ได้ยินไหมครับ 不被长度挡住",
      quick.lookup("ได้ยินไหมครับ", "th", "zh"), "听得到吗")
# 有实质内容的句子必须交给模型，绝不能被表吃掉
for _t in ("好的，那我们周五之前发给你们", "没有问题，但是需要先确认数量",
           "对，这个型号的产能是三千件"):
    check(f"不吃掉 {_t[:8]}…",
          quick.lookup(_t, "zh", "th") or quick.lookup(_t, "th", "zh"), None)

print("\n【回声判定】")
from translator.echo import SR as _ESR
from translator.echo import EchoGuard, dilate, envelope


def _speech(seconds, seed):
    """造一段像话音的信号：音节起伏 + 共振峰 + 摩擦噪声。"""
    r = np.random.default_rng(seed)
    n = int(seconds * _ESR)
    t = np.arange(n) / _ESR
    env = np.zeros(n)
    pos = 0.0
    while pos < seconds:
        dur, gap = r.uniform(0.11, 0.26), r.uniform(0.02, 0.18)
        a, b = int(pos * _ESR), int(min(pos + dur, seconds) * _ESR)
        if b > a:
            env[a:b] = (np.hanning(b - a) ** 0.7) * r.uniform(0.45, 1.0)
        pos += dur + gap
    f0 = r.uniform(95, 190)
    sig = np.zeros(n)
    for h, amp in ((1, 1.0), (2, .55), (3, .33), (4, .2)):
        sig += amp * np.sin(2 * np.pi * f0 * h * t + r.uniform(0, 6))
    sig += 0.45 * r.normal(0, 1, n)
    return (sig / (np.abs(sig).max() + 1e-9) * env).astype(np.float32)


def _echo_of(x, atten, seed=5):
    """音箱 → 房间 → 麦克风：衰减、混响拖尾、底噪、轻微非线性。"""
    r = np.random.default_rng(seed)
    y = np.zeros(len(x) + _ESR // 2, dtype=np.float32)
    y[:len(x)] += x * atten
    for tap_ms, g in ((17, .38), (31, .26), (53, .17), (89, .10), (140, .05)):
        k = int(tap_ms / 1000 * _ESR)
        y[k:k + len(x)] += x * atten * g
    y += r.normal(0, 0.0016, len(y))
    return np.tanh(y * 1.6).astype(np.float32) / 1.6


_T0, _DELAY = 1000.0, 0.15


def _guard_with(far, at=_T0 + 1.0):
    """把 far 按 30ms 一块喂进去，模拟采集线程。"""
    g = EchoGuard()
    blk = int(0.03 * _ESR)
    for i in range(0, len(far), blk):
        seg = far[i:i + blk]
        g.push_reference(seg, at + (i + len(seg)) / _ESR)
    return g


# 纯回声：麦克风里只有音箱漏回来的对方的声音
_far = _speech(4.0, 11)
_g = _guard_with(_far)
_mic = _echo_of(_far, 0.13)
check("纯回声被拦下",
      _g.check(_mic, _T0 + 1.0 + _DELAY,
               _T0 + 1.0 + _DELAY + len(_mic) / _ESR)[0], True)

# 自己说话、对方那一路安静 —— 绝不能丢
_g2 = EchoGuard()
check("对方没出声时不判回声",
      _g2.check(_speech(3.0, 21), _T0, _T0 + 3.0)[0], False)

# 两人同时说话 —— 最危险的一类，误丢等于把用户的话弄没了
_far3 = _speech(6.0, 31)
_me3 = _speech(3.0, 41)
_g3 = _guard_with(_far3)
_mix = _echo_of(_far3, 0.22)[int(1.0 * _ESR):int(4.0 * _ESR)] + _me3 * 0.9
check("两人同时说话时保留我方",
      _g3.check(_mix.astype(np.float32), _T0 + 2.0, _T0 + 5.0)[0], False)

# 对方说完后自己接话 —— 参考里有声，但不是同一段
_far4 = _speech(4.0, 51)
_g4 = _guard_with(_far4)
check("对方说完后接话不算回声",
      _g4.check(_speech(2.6, 61), _T0 + 5.2, _T0 + 7.8)[0], False)

# 没标定过两路时间差之前，短句一律不判（帧数太少会撞出假的高相关）
_far5 = _speech(6.0, 71)
_g5 = _guard_with(_far5)
_short = _echo_of(_far5, 0.2)[int(1.0 * _ESR):int(1.7 * _ESR)]
_got, _why = _g5.check(_short.astype(np.float32), _T0 + 2.0, _T0 + 2.7)
check("未标定时短句不判", (_got, "尚未标定" in _why), (False, True))

# 混响拖尾：不做膨胀的话覆盖率会被拖尾拉垮（实测能从 96% 掉到 41%）
_m = np.array([True, False, False, False, True, False], dtype=bool)
check("包络膨胀往后摊开",
      dilate(_m, 2, 0).tolist(),
      [True, True, True, False, True, True])
check("能量包络长度正确", len(envelope(np.zeros(3200, dtype=np.float32))), 10)

print("\n【回声：管线接线端到端】")
# 单测 check() 只能证明算法对，证明不了接线对：断句器给的 start_ts 是
# "这句开始的挂钟时刻"，push_reference 记的是"采集线程收到这块的时刻"，
# 两者对不上的话判定就永远落在错的位置。这里走真实的 Segmenter + 队列。
import queue as _q

from translator.config import DEFAULTS as _DEF


def _echo_of(x, atten, seed=5):
    r = np.random.default_rng(seed)
    y = np.zeros(len(x) + _ESR // 2, dtype=np.float32)
    y[:len(x)] += x * atten
    for tap_ms, g in ((17, .38), (31, .26), (53, .17), (89, .10), (140, .05)):
        k = int(tap_ms / 1000 * _ESR)
        y[k:k + len(x)] += x * atten * g
    y += r.normal(0, 0.0016, len(y))
    return np.tanh(y * 1.6).astype(np.float32) / 1.6


def _pipe(mic_track, ref_track, enabled=True):
    """照搬 pipeline._segment_loop / _dispatch 的结构和调用顺序。"""
    segs = {"me": Segmenter(_DEF["vad"]), "them": Segmenter(_DEF["vad"])}
    guard = EchoGuard(dict(_DEF["echo_cancel"], enabled=enabled))
    out, dropped = [], 0

    def dispatch(ch, finished):
        nonlocal dropped
        for audio, start_ts, _cut in finished:
            if ch == "me":
                hit, _why = guard.check(audio, start_ts,
                                        start_ts + len(audio) / 16000.0)
                if hit:
                    dropped += 1
                    continue
            out.append(ch)

    blk = int(0.03 * _ESR)
    base = 1000.0
    n = max(len(mic_track), len(ref_track))
    for i in range(0, n, blk):
        now = base + (i + blk) / _ESR
        for ch, track in (("them", ref_track), ("me", mic_track)):
            if i >= len(track):
                continue
            if ch == "them":
                guard.push_reference(track[i:i + blk], now)
            dispatch(ch, segs[ch].push(track[i:i + blk], now))
            for c2, s2 in segs.items():
                if c2 != ch:
                    dispatch(c2, s2.tick(now))
    for k in range(80):                      # 断流收尾
        for c2, s2 in segs.items():
            dispatch(c2, s2.tick(base + n / _ESR + 0.03 * (k + 1)))
    return out, dropped


def _sil(s):
    return np.zeros(int(s * _ESR), dtype=np.float32)


_sp = lambda s, seed: _speech(s, seed) * 0.5

# 甲：对方连说三句全从音箱漏进麦克风，我方全程没开口
_far = np.concatenate([_sil(.5), _sp(2.4, 11), _sil(1.4), _sp(2.2, 12),
                       _sil(1.4), _sp(2.6, 13), _sil(1.5)]).astype(np.float32)
_mic = np.concatenate([_sil(.65),
                       _echo_of(_far[int(.5 * _ESR):], 0.22)]).astype(np.float32)
_out, _drop = _pipe(_mic, _far)
check("外放漏音：对方那一路照常出句", _out.count("them") >= 3, True)
check("外放漏音：我方一句都没冒出来", _out.count("me"), 0)
check("外放漏音：确实是被回声拦下的", _drop >= 3, True)

# 乙：我方说话、对方安静 —— 绝不能丢，丢了这软件就废了
_me = np.concatenate([_sil(.5), _sp(2.5, 21), _sil(1.4),
                      _sp(2.3, 22), _sil(1.5)]).astype(np.float32)
_out, _drop = _pipe(_me, _sil(len(_me) / _ESR))
check("我方说话对方安静：一句没丢", _out.count("me") >= 2, True)
check("我方说话对方安静：零误判", _drop, 0)

# 丙：关掉开关后，行为和改动前完全一致
_out, _drop = _pipe(_mic, _far, enabled=False)
check("关掉 echo_cancel 后不再拦截", _drop, 0)
check("关掉 echo_cancel 后我方照常出句", _out.count("me") >= 1, True)

print("\n【纠错表：不能误伤，也不能被后续规则咬一口】")
from translator.asr import ASR as _ASR

_fx = _ASR({"asr_corrections": {
    "烟厂": "验厂",            # 老被听错的词
    "香烟厂": "香烟厂",        # 保护：映射到自己 = 这个词别动
    "采药会": "产销会",
    "专柜的客诉": "装柜的客诉",
}})
check("保护条目排在最前", _fx.fixes[0][0], "香烟厂")
check("该改的照改", _fx.apply_fixes("之前烟厂那怎么样了"), "之前验厂那怎么样了")
# 这条是实测踩过的坑：「烟厂的→验厂的」和保护条目同为 3 字，
# 只按长度排的话谁先谁后看写入顺序，结果「香烟厂」被改成「香验厂」
check("被保护的词不受影响",
      _fx.apply_fixes("这是香烟厂的订单"), "这是香烟厂的订单")
check("同一句里两者共存",
      _fx.apply_fixes("香烟厂的订单和烟厂报告"), "香烟厂的订单和验厂报告")
check("换完的结果不会被再咬一口",
      _fx.apply_fixes("采药会上说专柜的客诉"), "产销会上说装柜的客诉")
check("空表不出错", _ASR({}).apply_fixes("随便一句话"), "随便一句话")

print("\n【翻译后端：报错和状态要说对是哪一家】")
from translator.pipeline import Pipeline as _P
from translator.pipeline import friendly_error as _fe


class _E(Exception):
    pass


_rate = _E("rate_limit exceeded 429")
check("OpenAI 限流报错点名 OpenAI", "OpenAI" in _fe(_rate, "openai"), True)
check("Anthropic 限流报错点名 Anthropic",
      "Anthropic" in _fe(_rate, "anthropic"), True)
_auth = _E("invalid_api_key 401")
check("OpenAI 认证失败指向 openai_api_key",
      "openai_api_key" in _fe(_auth, "openai"), True)
check("Anthropic 认证失败指向 anthropic_api_key",
      "anthropic_api_key" in _fe(_auth, "anthropic"), True)
_404 = _E("model does not exist 404")
check("OpenAI 模型名错误指向 openai_translate_model",
      "openai_translate_model" in _fe(_404, "openai"), True)
# 顶栏显示的必须是当前真正在用的模型，不能永远显示 Anthropic 那个
_ps = _P.public_status


class _St:
    def __init__(self, tr, cfg):
        self.translator, self.cfg = tr, cfg
        self.muted = {"me": False, "them": False}
        self.locks = {"me": None, "them": None}
        self.paused = False
        self.stream = None
        self.fast_error = ""
        self.status = {"asr": "", "devices": {}, "ready": True}
        self.history = []
        self.translate_error = ""

    public_status = _ps


class _FakeTr:
    def __init__(self, m):
        self.model = m


check("状态里显示当前后端的模型",
      _St(_FakeTr("gpt-5.6-luna"), {"model": "claude-opus-5"}
          ).public_status()["model"], "gpt-5.6-luna")
check("没有 model 属性时退回配置值",
      _St(object(), {"model": "claude-opus-5"}).public_status()["model"],
      "claude-opus-5")
check("没有翻译器时说明未启用",
      _St(None, {"model": "x"}).public_status()["model"], "翻译未启用")

print("\n【回声：文字兜底 + 事后撤回】")
from translator.pipeline import Pipeline as _P
# 实测 8 条漏音里 4 条是"麦克风先出字幕"——麦克风只听到对方一整句里
# 最响的几个字，那段音频更短、更早收尾。判断当下对方那条还没出现，
# 结构上就拦不住，只能等对方那条到了再回头撤。
_pl = _P.__new__(_P)
_pl.cfg = {"dedupe_ratio": 0.65, "dedupe_window_sec": 10.0}
_pl._recent = []

_pl._remember("them", "所以说，你没有走过这个流程，呃，不太清楚，周建国不太清楚，"
                      "当时他就在这个流程的基础上做了一个调整。", 1)
check("对方先到：我方的残片被当场丢弃",
      _P._is_duplicate(_pl, "me",
                       "所以说，你要走过这个流程，不太清楚，不太清楚，"
                       "当时他就在流程的基础上做了一个调整。"), True)
check("对方先到：无关的话不受影响",
      _P._is_duplicate(_pl, "me", "那我们下周再确认一下产能的数字。"), False)

# 我方先到的情况
_pl2 = _P.__new__(_P)
_pl2.cfg = {"dedupe_ratio": 0.65, "dedupe_window_sec": 10.0}
_pl2._recent = []
_pl2._remember("me", "不同的工厂了不同的人，是这样子的。", 77)
_gone = _P.late_echo(_pl2, {"channel": "them",
                            "text": "李明华，根据不同的工厂去加签不同的人，是这样子的。"})
check("我方先到：对方那条一到就撤回我方的", _gone and _gone["id"], 77)

# 我方那条更长 = 不是残片，不能撤
_pl3 = _P.__new__(_P)
_pl3.cfg = {"dedupe_ratio": 0.65, "dedupe_window_sec": 10.0}
_pl3._recent = []
_pl3._remember("me", "李明华，根据不同的工厂去加签不同的人，是这样子的，我确认过了。", 88)
check("我方那条更长时不撤（不像残片）",
      _P.late_echo(_pl3, {"channel": "them", "text": "不同的工厂了不同的人。"}), None)

# 内容不像的绝不能撤
_pl4 = _P.__new__(_P)
_pl4.cfg = {"dedupe_ratio": 0.65, "dedupe_window_sec": 10.0}
_pl4._recent = []
_pl4._remember("me", "那这个报价单你们什么时候能发过来？", 99)
check("内容不像时绝不撤回",
      _P.late_echo(_pl4, {"channel": "them",
                          "text": "李明华，根据不同的工厂去加签不同的人。"}), None)

# 对方自己说的话不会触发撤回
_pl5 = _P.__new__(_P)
_pl5.cfg = {"dedupe_ratio": 0.65, "dedupe_window_sec": 10.0}
_pl5._recent = []
_pl5._remember("them", "不同的工厂了不同的人，是这样子的。", 55)
check("只撤我方的，不会撤对方的",
      _P.late_echo(_pl5, {"channel": "them",
                          "text": "李明华，根据不同的工厂去加签不同的人，是这样子的。"}),
      None)

# 会议记录里也要删掉，否则纪要会把同一句算两遍、还把说话人标错
_td = Path(tempfile.mkdtemp())
_ts2 = Transcript({"save_transcript": True, "transcript_dir": "rec"}, _td)
_ts2._min_interval = 0
for _i in (1, 2, 3):
    _ts2.add({"id": _i, "channel": "me", "speaker": "我方", "src_lang": "zh",
              "tgt_lang": "th", "text": f"第{_i}句", "translation": f"แปล{_i}",
              "ts": _time.time()})
_ts2.remove(2)
check("记录里删掉了被撤回的那条",
      sorted(x["id"] for x in _ts2.items), [1, 3])
_ts2.remove(999)                       # 不存在的 id 不能出事
check("删不存在的 id 不出错", sorted(x["id"] for x in _ts2.items), [1, 3])
_jl2 = list((_td / "rec").glob("*.jsonl"))
check("jsonl 文件里也真的删掉了",
      bool(_jl2) and len(_jl2[0].read_text(encoding="utf-8").strip().split("\n")), 2)
shutil.rmtree(_td, ignore_errors=True)

print("\n【暂停】")
# 会开完了人还在屋里说话，照样识别+翻译，钱就这么流走。
# 暂停期间必须是零调用，继续之后不能冒出停之前的陈年旧话。
import queue as _queue

from translator.config import DEFAULTS as _D2
from translator.pipeline import Pipeline as _P


class _Mini:
    """照搬 pipeline 里和暂停有关的结构，不起真实管线。"""

    def __init__(self):
        self.cfg = dict(_D2)
        self.seg_q = _queue.Queue(maxsize=6)
        self.segmenters = {"me": Segmenter(self.cfg["vad"]),
                           "them": Segmenter(self.cfg["vad"])}
        self.paused = False
        self._paused_at = 0.0
        self.paused_total = 0.0
        self.stream = None            # 流式识别默认不开
        self._lang_of = {}
        self.pushed = []

    def _push_sync(self, ev):
        self.pushed.append(ev)

    def public_status(self):
        return {"paused": self.paused}

    set_paused = _P.set_paused          # 用真实实现

    def pump(self, ch, pcm, now):
        if self.paused:
            return
        for a, st, cut in self.segmenters[ch].push(pcm, now):
            try:
                self.seg_q.put_nowait((ch, a, now, cut, ""))
            except _queue.Full:
                pass
        for c2, sg in self.segmenters.items():
            if c2 != ch:
                for a, st, cut in sg.tick(now):
                    try:
                        self.seg_q.put_nowait((c2, a, now, cut, ""))
                    except _queue.Full:
                        pass


def _feed(mp, ch, track, t0):
    blk = int(0.03 * 16000)
    for i in range(0, len(track), blk):
        mp.pump(ch, track[i:i + blk], t0 + (i + blk) / 16000)
    return t0 + len(track) / 16000


_zeros = lambda s: np.zeros(int(s * 16000), dtype=np.float32)
_T = 5000.0

_mp = _Mini()
_feed(_mp, "me", np.concatenate([_zeros(.3), _sp(2.5, 301), _zeros(1.6)]), _T)
check("没暂停时能正常断句", _mp.seg_q.qsize() >= 1, True)

_mp = _Mini()
_mp.set_paused(True)
_feed(_mp, "me", np.concatenate([_zeros(.3), _sp(6.0, 302), _zeros(1.6)]), _T)
_feed(_mp, "them", np.concatenate([_zeros(.3), _sp(6.0, 303), _zeros(1.6)]), _T)
check("暂停期间一句都不产生（= 零调用）", _mp.seg_q.qsize(), 0)
check("暂停时撤掉了界面上的「正在识别」",
      sum(1 for e in _mp.pushed
          if e.get("type") == "partial" and e.get("text") == ""), 2)

_mp = _Mini()
_feed(_mp, "me", np.concatenate([_zeros(.3), _sp(2.0, 304), _zeros(1.6)]), _T)
_feed(_mp, "me", _sp(2.0, 305), _T + 10)          # 再攒半句在断句器里
check("暂停前：队列有货且断句器攒着半句",
      _mp.seg_q.qsize() >= 1 and _mp.segmenters["me"].speaking, True)
_mp.set_paused(True)
check("暂停清空了排队的句子", _mp.seg_q.qsize(), 0)
check("暂停丢掉了攒着的半句", _mp.segmenters["me"].speaking, False)

_mp.set_paused(False)
_t2 = _feed(_mp, "me",
            np.concatenate([_zeros(.3), _sp(2.2, 306), _zeros(1.6)]), _T + 300)
_got = []
while not _mp.seg_q.empty():
    _got.append(_mp.seg_q.get_nowait())
check("继续后能正常断句", len(_got) >= 1, True)
if _got:
    # flush 必须清全部状态：preroll / tail / voiced 计数留着的话，
    # 上一段的残留会被拼到新句子开头，长度和时间戳都会不对
    check("断出的是新句子不是旧的", _t2 - _got[0][2] < 5, True)
    check("没被拼上暂停前的旧音频", 1.0 < len(_got[0][1]) / 16000 < 4.5, True)

_mp = _Mini()
_mp.set_paused(True)
_mp.set_paused(True)
check("重复按暂停不出问题", _mp.paused, True)
_mp._paused_at = _time.time() - 120
_mp.set_paused(False)
_mp.set_paused(False)
check("重复按继续不出问题", _mp.paused, False)
check("累计暂停时长记账正确", 115 < _mp.paused_total < 125, True)

print("\n【语种设置写回 config.json】")
# 界面上的语种选择只活在内存里的话，下次启动会退回上一场会议的设置——
# 实测就因为这个，和泰国同事开会时开场两句泰语被塞进中文引擎，出来一堆乱码。
# 这里最要紧的一条：文件里有 API 密钥，写回时一个字节都不能碰。
import json

from translator import config as _C
from translator.pipeline import Pipeline as _P


class _Stub:
    def __init__(self, cfg):
        self.cfg = cfg
        lk = cfg.get("language_lock") or {}
        self.locks = {"me": lk.get("me"), "them": lk.get("them")}
        self._lang_of = {"me": "zh", "them": "zh"}


_cdir = Path(tempfile.mkdtemp())
_cp = _cdir / "config.json"
_cp.write_text(json.dumps({
    "anthropic_api_key": "sk-FAKE-FOR-TEST",
    "asr": {"tencent_secret_key": "FAKE", "beam_size": 1},
    "language_lock": {"me": None, "them": "zh"},
    "language_bias": {"me": None, "them": "zh"},
    "glossary": {"验厂": "ตรวจโรงงาน"},
}, ensure_ascii=False, indent=2), encoding="utf-8")
_orig_cp = _C.CONFIG_PATH
_C.CONFIG_PATH = _cp

_st = _Stub({"language_lock": {"me": None, "them": "zh"},
             "language_bias": {"me": None, "them": "zh"}})
_P.set_lock(_st, "them", "th")
_disk = json.loads(_cp.read_text(encoding="utf-8"))
check("界面改语种会写回文件", _disk["language_lock"]["them"], "th")
check("先验偏向跟着一起改", _disk["language_bias"]["them"], "th")
check("另一路不受影响", _disk["language_lock"]["me"], None)
check("密钥原样保留", _disk["anthropic_api_key"], "sk-FAKE-FOR-TEST")
check("嵌套字段没被冲掉", _disk["asr"]["tencent_secret_key"], "FAKE")
check("术语表没丢", _disk["glossary"], {"验厂": "ตรวจโรงงาน"})
check("没把默认值灌进文件", len(_disk), 5)

_P.set_lock(_st, "them", None)
_disk = json.loads(_cp.read_text(encoding="utf-8"))
check("改回自动写的是 null", _disk["language_lock"]["them"], None)
# 解锁时**必须**把偏向一起清掉。这条最初写反了，代价是实测中一场会
# 延迟中位数从 0.9s 涨到 1.9s：对方从"泰语"改回"自动"后偏向还留着 th，
# 而那场对方 87% 说中文，判错语种就要补打一次云端往返（每句 1.34 次调用）。
check("解锁时偏向一起清掉（留着会拿过时先验判语种）",
      _disk["language_bias"]["them"], None)

_P.set_lock(_st, "them", "en")          # 非法值一律按自动处理
check("非法语种不会写进文件",
      json.loads(_cp.read_text(encoding="utf-8"))["language_lock"]["them"], None)

_bad = _cdir / "bad.json"
_bad.write_text("{ 坏掉的 json ", encoding="utf-8")
_C.CONFIG_PATH = _bad
_before = _bad.read_bytes()
check("文件损坏时写回失败而不是覆盖",
      _C.save_fields({"language_lock": {"me": "zh"}}), False)
check("损坏的文件没被改动", _bad.read_bytes() == _before, True)
check("没留下临时文件", list(_cdir.glob("*.tmp")), [])

_C.CONFIG_PATH = _orig_cp
shutil.rmtree(_cdir, ignore_errors=True)

print("\n【会议纪要】")
from translator.minutes import _split, digest, meta_of

_items = [{"ts": 1758000000 + i * 10, "channel": "me" if i % 2 else "them",
           "speaker": "我方" if i % 2 else "对方", "src_lang": "zh",
           "text": f"第{i}句话"} for i in range(12)]
_d = digest(_items)
check("摘要每句一行", len(_d.split("\n")), 12)
check("摘要带说话人", "我方" in _d and "对方" in _d, True)
check("空文本不进摘要", digest([{"ts": 1, "text": "  "}]), "")
_m2 = meta_of(_items)
check("统计条数", _m2["count"], 12)
check("统计两路", (_m2["me"], _m2["them"]), (6, 6))
# 长会要能切块，而且切完拼回去不能丢内容
_long = "\n".join(f"[10:00] 我方：这是第{i}句" for i in range(4000))
_parts = _split(_long, 26000)
check("长记录被切成多块", len(_parts) > 1, True)
check("切块不丢内容", "\n".join(_parts), _long)
check("短记录不切块", len(_split("一行", 26000)), 1)
# 块要尽量一样大：装满一块再开下一块会得到 26000+7000，
# 后面那块太薄，提要出来的东西明显比前面粗
check("切块大小均衡", max(len(p) for p in _parts) - min(len(p) for p in _parts) < 2000,
      True)

print("\n【Markdown → Word】")
try:
    import re as _re2
    import zipfile as _zip

    from translator.minutes import write_docx

    _MD = (
        "# 会议纪要 / บันทึกการประชุม\n\n"
        "## 一、会议信息\n\n- **时间**：2026-09-16\n\n"
        "## 三、待办事项 / รายการ\n\n"
        "| # | 事项 | รายการ | 负责方 |\n"
        "|---|------|--------|--------|\n"
        "| 1 | 准备电工证 | เตรียมใบรับรองช่างไฟฟ้า | 泰国工厂 |\n"
        "| 2 | 整理花名册 | จัดทำทะเบียนพนักงาน | HR |\n"
    )
    _out = Path(tempfile.mkdtemp()) / "t.docx"
    write_docx(_MD, _out)
    with _zip.ZipFile(_out) as _z:
        _xml = _z.read("word/document.xml").decode("utf-8")
    _runs = _re2.findall(r"<w:r\b.*?</w:r>", _xml, _re2.S)
    _thai = [r for r in _runs if _re2.search(r"[฀-๿]", r)]
    _txt = "".join(_re2.findall(r"<w:t[^>]*>(.*?)</w:t>", _xml, _re2.S))
    # Word 把泰文当"复杂文种"，必须单独指定 w:cs 字体，否则全渲染成方框
    check("泰文都设了 w:cs 字体",
          bool(_thai) and all('w:cs="Leelawadee UI"' in r for r in _thai), True)
    check("表格渲染成 Word 表格", _xml.count("<w:tbl>"), 1)
    check("表格行数正确", len(_re2.findall(r"<w:tr[ >]", _xml)), 3)
    check("分隔行 |---| 没当成内容", "---" in _txt, False)
    check("加粗标记被吃掉", "**" in _txt, False)
    check("表格里的泰文在", "เตรียมใบรับรองช่างไฟฟ้า" in _txt, True)
    shutil.rmtree(_out.parent, ignore_errors=True)
except ImportError:
    print("  · 没装 python-docx，跳过")

print()
if fails:
    print(f"✗ {len(fails)} 项未通过：{fails}")
    sys.exit(1)
print("✓ 全部通过")
