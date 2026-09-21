"""OpenAI vs 现在这套：翻译和识别两条路各测一遍。

    .venv\\Scripts\\python.exe 试试OpenAI.py          只测翻译（不用说话）
    .venv\\Scripts\\python.exe 试试OpenAI.py --录音     连识别也测（要对着麦克风说话）

翻译那部分用你真实会议记录里的句子，提示词和术语表两边完全一样，
比的是模型本身。识别那部分需要你说一句中文，两个引擎跑同一段音频。
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent      # 这个脚本在「工具」子目录里
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from translator.config import load_config

ROUNDS = 3
SR = 16000


# ----------------------------------------------------------------- 翻译
def real_sentences(n=5):
    d = ROOT / "会议记录"
    rows = []
    for f in sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime,
                    reverse=True)[:3]:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("src_lang") == "zh" and (r.get("translation") or ""):
                    rows.append(r["text"].strip())
    if not rows:
        return ["行，那这个就说到这儿。",
                "这个客诉的详情描述，发生时间和涉及的金额都要写清楚。"]
    rows.sort(key=len)
    step = max(1, len(rows) // n)
    return [rows[i] for i in range(0, len(rows), step)][:n]


async def bench_translate(name, tr, sents):
    first, total, outs = [], [], []
    for s in sents:
        for _ in range(ROUNDS):
            tr._cache.clear()
            mark, t0 = {}, time.perf_counter()

            async def on_delta(_p, _m=mark, _t=t0):
                _m.setdefault("f", time.perf_counter() - _t)

            try:
                out = await tr.translate_stream(s, "zh", "th", None, on_delta)
            except Exception as e:
                print(f"    ✗ {name} 出错：{type(e).__name__}: {str(e)[:90]}")
                return None
            total.append(time.perf_counter() - t0)
            first.append(mark.get("f", total[-1]))
            outs.append(out)
    return statistics.median(first), statistics.median(total), outs


async def translate_ab(cfg):
    sents = real_sentences()
    print("=" * 70)
    print(f"翻译对比　{len(sents)} 句 × 每句 {ROUNDS} 次（取自你的真实会议记录）")
    print("=" * 70)
    for i, s in enumerate(sents, 1):
        print(f"  {i}. [{len(s):>3} 字] {s[:44]}{'…' if len(s) > 44 else ''}")

    res = {}
    if cfg.get("anthropic_api_key"):
        from translator.translate import Translator

        print(f"\n  正在测 Anthropic {cfg['model']} …")
        r = await bench_translate("Anthropic", Translator(cfg), sents)
        if r:
            res[f"Anthropic {cfg['model']}"] = r
    if cfg.get("openai_api_key"):
        from translator.translate_openai import OpenAITranslator

        print(f"  正在测 OpenAI {cfg['openai_translate_model']} …")
        r = await bench_translate("OpenAI", OpenAITranslator(cfg), sents)
        if r:
            res[f"OpenAI {cfg['openai_translate_model']}"] = r
    else:
        print("\n  （没有 OpenAI Key，跳过。填 config.json 的 openai_api_key）")

    if res:
        print(f"\n  {'后端':<32}{'首字':>10}{'整句':>10}")
        print("  " + "-" * 52)
        for k, (f, t, _) in res.items():
            print(f"  {k:<32}{f:>8.2f}s{t:>9.2f}s")
        print("\n  同一句的译文对照（第二句）：")
        for k, (_, _, outs) in res.items():
            idx = min(ROUNDS, len(outs) - 1)
            print(f"\n    {k}")
            print(f"      {outs[idx]}")
    return res


# ----------------------------------------------------------------- 识别
def record(seconds=6.0):
    from translator.audio_capture import find_input_device, resample

    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        import pyaudio

    p = pyaudio.PyAudio()
    dev = find_input_device(p)
    if dev is None:
        print("  ✗ 找不到麦克风")
        return None
    rate = int(dev["defaultSampleRate"])
    blk = int(rate * 0.1)
    st = p.open(format=pyaudio.paFloat32, channels=1, rate=rate, input=True,
                input_device_index=dev["index"], frames_per_buffer=blk)
    print(f"\n★ 请说一句中文（{seconds:.0f} 秒），比如")
    print("  「这次装柜的客诉，咱们要在产销会上汇报，一页可能不够」\n")
    parts, t0 = [], time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        x = np.frombuffer(st.read(blk, exception_on_overflow=False),
                          dtype=np.float32)
        parts.append(resample(x, rate, SR) if rate != SR else x)
        lvl = min(int(np.abs(x).max() * 40), 30)
        print(f"\r  {time.perf_counter()-t0:4.1f}s  {'█'*lvl}{' '*(30-lvl)}",
              end="")
    st.close()
    p.terminate()
    print("\n")
    return np.concatenate(parts)


def asr_ab(cfg, audio):
    from translator.asr import ASR

    print("=" * 70)
    print("识别对比　同一段音频，两个引擎各跑一次")
    print("=" * 70)
    for backend, label in (("tencent", "腾讯云 一句话识别（现在用的）"),
                           ("openai", f"OpenAI {cfg['asr'].get('openai_model')}")):
        acfg = dict(cfg["asr"])
        acfg["backend"] = backend
        if backend == "openai" and not acfg.get("openai_api_key"):
            print(f"\n  {label}：没有 Key，跳过")
            continue
        try:
            a = ASR(acfg)
            a.load()
            t0 = time.perf_counter()
            text, lang = a.transcribe(audio, prefer="zh", known="zh")
            cost = time.perf_counter() - t0
            print(f"\n  {label}")
            print(f"    耗时 {cost:.2f}s　语种 {lang}")
            print(f"    识别：{text}")
        except Exception as e:
            print(f"\n  {label}：✗ {type(e).__name__}: {str(e)[:90]}")


async def main():
    cfg = load_config()
    has_openai = bool(cfg.get("openai_api_key"))
    print(f"OpenAI Key：{'已配置' if has_openai else '✗ 还没填'}"
          f"　　基地址：{cfg.get('openai_base_url') or '官方'}\n")
    if not has_openai:
        print("请先在 config.json 里填 openai_api_key，然后重跑这个脚本。\n")

    await translate_ab(cfg)

    if "--录音" in sys.argv or "--record" in sys.argv:
        audio = record()
        if audio is not None:
            asr_ab(cfg, audio)
    else:
        print("\n" + "=" * 70)
        print("想连识别也测的话，加 --录音 重跑（需要对着麦克风说一句中文）：")
        print("  .venv\\Scripts\\python.exe 试试OpenAI.py --录音")
        print("=" * 70)


asyncio.run(main())
