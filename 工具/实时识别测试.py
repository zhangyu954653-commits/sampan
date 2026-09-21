"""实测腾讯云「实时语音识别」到底比现在快多少。

    .venv\\Scripts\\python.exe 实时识别测试.py

对着麦克风说一句中文（比如"这次装柜的客诉要在产销会上汇报"），
脚本会同时跑两条路，把时间戳打出来对比：

    现在这条：说完 → 整段发出去 → 等结果
    流式这条：边说边发 → 说的过程中文字就在回来

需要在 config.json 的 asr 里补一个 tencent_appid（腾讯云「访问密钥」页面
上那串数字，和 SecretId 在同一页）。这个测试会用掉几十秒的免费额度
（实时识别每月免费 5 小时）。
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 在「工具」子目录里
sys.stdout.reconfigure(encoding="utf-8")

from translator.config import load_config
from translator.tencent_stream import TencentStreamASR

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
SR = 16000


def record(seconds: float) -> tuple[np.ndarray, list]:
    """录一段麦克风，同时记下每一块到达的时刻（流式要按这个节奏回放）。"""
    from translator.audio_capture import find_input_device, resample

    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        import pyaudio

    p = pyaudio.PyAudio()
    dev = find_input_device(p)
    if dev is None:
        print("✗ 找不到麦克风")
        sys.exit(1)
    rate = int(dev["defaultSampleRate"])
    block = int(rate * 0.1)
    print(f"  设备：{dev['name']}　{rate}Hz")
    st = p.open(format=pyaudio.paFloat32, channels=1, rate=rate,
                input=True, input_device_index=dev["index"],
                frames_per_buffer=block)
    print(f"\n★ 开始说话（{seconds:.0f} 秒）…\n")
    parts, stamps, t0 = [], [], time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        raw = st.read(block, exception_on_overflow=False)
        x = np.frombuffer(raw, dtype=np.float32)
        if rate != SR:
            x = resample(x, rate, SR)
        parts.append(x)
        stamps.append(time.perf_counter() - t0)
        lvl = min(int(np.abs(x).max() * 40), 30)
        print(f"\r  录音中 {time.perf_counter()-t0:4.1f}s  "
              f"{'█' * lvl}{' ' * (30 - lvl)}", end="")
    st.close()
    p.terminate()
    print("\n  录完了。\n")
    return parts, stamps


async def run_stream(cfg, parts, stamps):
    asr = TencentStreamASR(cfg["asr"])
    ok, why = asr.available()
    if not ok:
        print(f"✗ {why}")
        return None

    marks = []
    t0 = None

    async def feeder():
        nonlocal t0
        t0 = time.perf_counter()
        for x, at in zip(parts, stamps):
            # 按当初录到的节奏回放 = 1:1 实时率，和真开会时一模一样
            wait = at - (time.perf_counter() - t0)
            if wait > 0:
                await asyncio.sleep(wait)
            yield x

    async def on_partial(text, stable):
        marks.append((time.perf_counter() - t0, text, stable))
        tag = "稳定" if stable else "识别中"
        print(f"    [{marks[-1][0]:5.2f}s] {tag}：{text}")

    print("【流式识别】边说边出字（时间从开口那一刻算起）")
    out = await asr.recognize(feeder(), "zh", on_partial)
    audio_end = stamps[-1] if stamps else 0
    if out.error:
        print(f"    ✗ {out.error}")
        return None
    done = marks[-1][0] if marks else 0
    print(f"\n    最终文本：{out.text}")
    print(f"    说完的时刻 {audio_end:.2f}s　拿到最终结果 {done:.2f}s"
          f"　→ **说完之后只等了 {max(0, done - audio_end):.2f}s**")
    print(f"    本次用掉免费额度：{asr.seconds:.1f} 秒音频")
    return max(0, done - audio_end)


def run_batch(cfg, parts, stamps):
    """现在这条路：整段音频发出去，等结果。"""
    from translator.asr import ASR

    print("\n【现在这条：一句话识别】说完才开始发")
    a = ASR(cfg["asr"])
    a.load()
    audio = np.concatenate(parts)
    t0 = time.perf_counter()
    text, lang = a.transcribe(audio, prefer="zh", known="zh")
    cost = time.perf_counter() - t0
    print(f"    最终文本：{text}")
    print(f"    → **说完之后等了 {cost:.2f}s**")
    return cost


async def main():
    cfg = load_config()
    if not cfg["asr"].get("tencent_appid"):
        print("=" * 66)
        print("需要先补一个 tencent_appid 才能测")
        print("=" * 66)
        print("""
腾讯云控制台 →「访问密钥」→ 页面上方的 **APPID**（一串数字，
和 SecretId 在同一页，不是 SecretId 本身）。

填进 config.json：

    "asr": {
      "tencent_appid": "1250000000",
      ...
    }

实时识别的 WebSocket 地址里必须带 AppID，现在用的一句话识别不需要，
所以之前没配过。每月免费 5 小时，这个测试只用几十秒。
""")
        return

    print("=" * 66)
    print("实时流式 vs 现在这条：对着麦克风说一句中文")
    print("=" * 66)
    parts, stamps = record(SECONDS)
    if not parts:
        return

    s = await run_stream(cfg, parts, stamps)
    b = run_batch(cfg, parts, stamps)

    if s is not None and b is not None:
        print("\n" + "=" * 66)
        print(f"  流式：说完后等 {s:.2f}s")
        print(f"  现在：说完后等 {b:.2f}s")
        print(f"  → 省下 {b - s:.2f}s")
        print("\n  这还只是识别这一段。断句等待 1.14s 和翻译首字 1.47s 不变，")
        print(f"  端到端大约从 {1.14 + b + 1.47:.1f}s 降到 {1.14 + s + 1.47:.1f}s。")


asyncio.run(main())
