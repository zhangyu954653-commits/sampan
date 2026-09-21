"""中泰实时会议翻译 —— 启动入口。

用法：
    python main.py                  正常启动
    python main.py --list-devices   列出所有音频输入设备（用于手动指定设备序号）
    python main.py --check          只检查环境和配置，不启动
    python main.py --minutes        把最近一场会整理成中泰双语纪要（.md + .docx）
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
import webbrowser

import os

if sys.platform == "win32":
    # Windows 控制台中文输出
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# HuggingFace 新的 xet 下载后端在国内网络下经常直接卡死（进度停在几 MB 不动），
# 走传统 HTTP 下载反而正常。必须在 import faster_whisper 之前设置。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from translator.config import CONFIG_PATH, load_config

BANNER = r"""
  ┌──────────────────────────────────────────────┐
  │   中文  ⇄  ไทย   视频会议实时翻译字幕          │
  └──────────────────────────────────────────────┘
"""


def cmd_list_devices():
    from translator.audio_capture import list_devices

    print("\n可用的音频输入设备：\n")
    print(f"{'序号':<6}{'类型':<10}{'声道':<6}{'采样率':<10}名称")
    print("-" * 76)
    for d in list_devices():
        kind = "系统声音" if d["loopback"] else "麦克风"
        print(f"{d['index']:<6}{kind:<10}{d['channels']:<6}{d['rate']:<10}{d['name']}")
    print(
        "\n提示：把想用的序号填到 config.json 的 audio.mic_device_index / "
        "audio.loopback_device_index。\n"
    )


def check(cfg: dict) -> bool:
    ok = True
    # 检查**当前真正在用的**那个后端。以前这里只看 Anthropic——
    # 切到 OpenAI 之后哪怕 Key 是错的，这里照样打勾，
    # 等开会说第一句才发现翻不出来。
    use_openai = str(cfg.get("translate_backend", "anthropic")).lower() == "openai"
    if use_openai:
        who, field, env = "OpenAI", "openai_api_key", "OPENAI_API_KEY"
        key = cfg.get("openai_api_key")
        model = cfg.get("openai_translate_model")
    else:
        who, field, env = "Anthropic", "anthropic_api_key", "ANTHROPIC_API_KEY"
        key = cfg.get("anthropic_api_key")
        model = cfg.get("model")
    if not key:
        # 不算致命错误：仍然可以启动，只识别不翻译，方便先调麦克风
        print(f"! 缺少 {who} API Key —— 只会显示原文，不会翻译")
        print(f"  → 在 {CONFIG_PATH.name} 里填 {field}，或设置环境变量 {env}")
    else:
        print(f"✓ 翻译：{who} {model}")
    # 纪要走的是 Anthropic，和翻译后端无关，单独提醒一次
    if use_openai and not cfg.get("anthropic_api_key"):
        print("! 会议纪要用的是 Anthropic，没配 anthropic_api_key 的话生成不了纪要")

    if cfg["asr"]["backend"] == "openai" and not cfg["asr"].get("openai_api_key"):
        print("! asr.backend=openai 但缺少 OpenAI API Key —— 将自动改用本地模型")

    if cfg["asr"]["backend"] == "tencent":
        if cfg["asr"].get("tencent_secret_id") and cfg["asr"].get("tencent_secret_key"):
            try:
                import tencentcloud  # noqa: F401

                print("✓ 腾讯云语音识别已配置")
            except ImportError:
                print("✗ 缺少腾讯云 SDK，请执行："
                      " .venv\\Scripts\\pip install tencentcloud-sdk-python-asr")
                ok = False
        else:
            print("! asr.backend=tencent 但缺少密钥 —— 将自动改用本地模型")
            print("  → 在 config.json 里填 asr.tencent_secret_id / tencent_secret_key")

    try:
        import pyaudiowpatch  # noqa: F401

        print("✓ 音频采集可用（含系统声音环回）")
    except ImportError:
        try:
            import pyaudio  # noqa: F401

            print("! 只装了普通 pyaudio —— 抓不到对方的声音，请安装 PyAudioWPatch")
        except ImportError:
            print("✗ 没有安装 PyAudioWPatch，无法采集音频")
            ok = False

    try:
        import faster_whisper  # noqa: F401

        print("✓ 本地语音识别可用")
    except ImportError:
        print("✗ 没有安装 faster-whisper")
        ok = False

    try:
        import webrtcvad  # noqa: F401

        print("✓ 语音断句 (WebRTC VAD) 可用")
    except ImportError:
        print("! 没装 webrtcvad-wheels，将使用简易能量断句（噪音环境下效果差一些）")

    return ok


def cmd_test_audio(cfg: dict, seconds: int = 15):
    """只测音频采集：看两路声音有没有进来、断句有没有触发。不需要任何 API Key。"""
    import math
    import queue
    import time

    from translator.audio_capture import AudioCapture
    from translator.vad import Segmenter

    q: queue.Queue = queue.Queue(maxsize=400)
    cap = AudioCapture(cfg, q)
    devices = cap.start()
    if not devices:
        print("✗ 没有可用的音频设备。")
        return

    print()
    for ch, name in devices.items():
        print(f"  {'我方(麦克风)' if ch == 'me' else '对方(系统声音)'} ← {name}")
    print(f"\n请说话，并在钉钉/任意播放器里放一段声音，测试 {seconds} 秒…\n")

    segs = {"me": Segmenter(cfg["vad"]), "them": Segmenter(cfg["vad"])}
    peak = {"me": 0.0, "them": 0.0}
    top = {"me": 0.0, "them": 0.0}      # 整段测试里的最大电平
    count = {"me": 0, "them": 0}
    end = time.time() + seconds
    last_draw = 0.0

    try:
        while time.time() < end:
            try:
                ch, pcm = q.get(timeout=0.2)
            except queue.Empty:
                ch = None
            now = time.time()
            if ch is not None:
                lvl = float(abs(pcm).max()) if pcm.size else 0.0
                peak[ch] = max(peak[ch], lvl)
                top[ch] = max(top[ch], lvl)
                count[ch] += len(segs[ch].push(pcm, now))   # 每项是 (音频, 开始时刻)
            for c2, seg in segs.items():
                if c2 != ch:
                    count[c2] += len(seg.tick(now))

            if now - last_draw > 0.12:
                last_draw = now
                bars = []
                for c, label in (("me", "我方"), ("them", "对方")):
                    lvl = int(min(1.0, peak[c] * 3) * 24)
                    bars.append(f"{label} |{'█' * lvl}{'·' * (24 - lvl)}| 句:{count[c]}")
                    peak[c] *= 0.55  # 缓慢回落，看起来像电平表
                print("\r  " + "   ".join(bars) + f"   剩余{int(end - now)}s ", end="")
    finally:
        cap.stop()

    print("\n")
    for c, label in (("me", "麦克风(我方)"), ("them", "系统声音(对方)")):
        if c not in devices:
            print(f"  —  {label}：未启用")
            continue
        db = 20 * math.log10(top[c]) if top[c] > 1e-6 else -99
        lvl = f"最高电平 {db:.0f} dBFS"
        if count[c] > 0:
            print(f"  ✓  {label}：正常，断出 {count[c]} 句　（{lvl}）")
        elif top[c] > 0.002:
            print(f"  !  {label}：有声音但没断出句子，{lvl}。"
                  f"声音太小，把 config.json 的 vad.rms_threshold 调低到 0.003 试试")
        else:
            print(f"  ✗  {label}：完全没有声音进来　（{lvl}）"
                  f"{'，说话时要对着麦克风' if c == 'me' else '，确认钉钉/播放器正在出声且没静音'}")
    print()


async def cmd_demo(cfg: dict):
    """只测界面：推几条假字幕，验证网页和 WebSocket。不需要任何 API Key。"""
    import time

    from translator.server import Server

    server = Server(cfg)
    server.static_status = {"asr": "演示模式", "model": "（未调用真实 API）",
                            "devices": {}, "ready": True,
                            "muted": {"me": False, "them": False}}
    url = await server.run()
    print(f"\n[演示] 界面已启动：{url}")
    print("[演示] 窗口里应该能看到字幕逐条出现。按 Ctrl+C 结束。\n")
    if cfg["server"].get("open_browser", True):
        from pathlib import Path

        from translator.window import open_window

        threading.Timer(0.8, lambda: print(
            "[演示] " + open_window(url, cfg["server"], Path(__file__).parent))).start()

    samples = [
        ("me", "zh", "你好，我们今天主要讨论一下十月份的发货计划。",
         "สวัสดีครับ วันนี้เราจะคุยกันเรื่องแผนการจัดส่งของเดือนตุลาคมเป็นหลักครับ"),
        ("them", "th", "ครับ ตอนนี้ทางเรามีสินค้าพร้อมส่งประมาณ 3,000 ชิ้นครับ",
         "好的，我们这边目前有大约 3000 件现货可以发出。"),
        ("me", "zh", "那报价单能不能在这周五之前发给我们？",
         "งั้นขอใบเสนอราคาส่งมาให้เราก่อนวันศุกร์นี้ได้ไหมครับ"),
        ("them", "th", "ได้ครับ ผมจะส่งให้ภายในวันพฤหัสบดีครับ",
         "可以的，我周四之前发给你们。"),
    ]

    async def push():
        await asyncio.sleep(2.0)
        for i, (ch, lang, src, dst) in enumerate(samples, 1):
            item = {
                "id": i, "channel": ch,
                "speaker": "我方" if ch == "me" else "对方",
                "src_lang": lang, "tgt_lang": "th" if lang == "zh" else "zh",
                "text": src, "translation": "", "ts": time.time(), "seconds": 2.0,
            }
            await server.broadcast({"type": "utterance", "item": item})
            await asyncio.sleep(1.1)
            await server.broadcast({"type": "translation", "id": i,
                                    "translation": dst, "tgt_lang": item["tgt_lang"]})
            await asyncio.sleep(1.6)
        print("[演示] 4 条示例字幕已推送完毕。")

    asyncio.create_task(push())
    try:
        await asyncio.Event().wait()
    finally:
        await server.close()


def cmd_test_asr(cfg: dict):
    """对比测试：同一段音频，本地模型和腾讯云各识别一遍，看精度和耗时。

    用系统自带的中文语音合成生成测试音频，不需要开会也能跑。
    """
    import tempfile
    import time as _t
    from pathlib import Path

    import numpy as np

    CASES = [
        "你好，今天主要讨论一下验厂的事情",
        "十月二号到十三号会有一个验厂，你们那边需要提前准备好资料",
        "这个型号的产能大概是多少，报价单能不能这周五之前发给我们",
    ]

    tmp = Path(tempfile.gettempdir()) / "zhth_asr_test"
    tmp.mkdir(exist_ok=True)
    print("\n正在用系统语音合成生成测试音频…")
    try:
        import subprocess

        ps = ["powershell", "-NoProfile", "-Command"]
        for i, text in enumerate(CASES):
            wav = tmp / f"c{i}.wav"
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "try{$s.SelectVoice('Microsoft Huihui Desktop')}catch{}; "
                f"$s.SetOutputToWaveFile('{wav}'); $s.Speak('{text}'); $s.Dispose()"
            )
            subprocess.run(ps + [script], check=True, capture_output=True)
    except Exception as e:
        print(f"✗ 生成测试音频失败：{e}")
        return

    from faster_whisper.audio import decode_audio

    from translator.asr import ASR
    from translator.tencent_asr import TencentASR

    clips = [(decode_audio(str(tmp / f"c{i}.wav"), sampling_rate=16000), t)
             for i, t in enumerate(CASES)]

    print("正在加载本地模型…")
    local = ASR(dict(cfg["asr"], backend="local"))
    local.load()

    cloud = TencentASR(cfg["asr"])
    has_cloud = cloud.available()
    if has_cloud:
        try:
            print("腾讯云：" + cloud.load())
        except Exception as e:
            print(f"✗ 腾讯云初始化失败：{e}")
            has_cloud = False
    else:
        print("! 没有配置腾讯云密钥，只测本地")

    print()
    l_time = c_time = 0.0
    for audio, truth in clips:
        dur = len(audio) / 16000
        print(f"  原句（{dur:.1f}s）：{truth}")

        t0 = _t.time()
        lt, _ = local.transcribe(audio, "zh")
        dt = _t.time() - t0
        l_time += dt
        print(f"    本地   {dt:5.2f}s  {lt}")

        if has_cloud:
            t0 = _t.time()
            try:
                ct = cloud.transcribe(audio, "zh")
            except Exception as e:
                ct = f"（失败：{e}）"
            dt = _t.time() - t0
            c_time += dt
            print(f"    腾讯云 {dt:5.2f}s  {ct}")
        print()

    n = len(clips)
    print(f"  平均每句：本地 {l_time / n:.2f}s"
          + (f"   腾讯云 {c_time / n:.2f}s" if has_cloud else ""))
    if has_cloud and c_time < l_time:
        print(f"  → 腾讯云快 {l_time / max(c_time, 0.01):.1f} 倍\n")


async def cmd_test_translate(cfg: dict):
    """只测翻译：拿几句话真调一次 Claude，确认 Key 和网络都通。"""
    from translator.translate import Translator

    try:
        tr = Translator(cfg)
    except Exception as e:
        print(f"\n✗ 翻译没法初始化：{e}\n")
        return

    print(f"\n模型：{cfg.get('model')}")
    if cfg.get("anthropic_base_url"):
        print(f"中转地址：{cfg['anthropic_base_url']}")
    if cfg.get("glossary"):
        print(f"术语表：{len(cfg['glossary'])} 条")
    print()

    cases = [
        ("zh", "th", "你好，我们今天主要讨论十月份的发货计划。"),
        ("zh", "th", "报价单能不能在这周五之前发给我们？"),
        ("th", "zh", "ตอนนี้ทางเรามีสินค้าพร้อมส่งประมาณ 3,000 ชิ้นครับ"),
        ("th", "zh", "ผมจะส่งใบเสนอราคาให้ภายในวันพฤหัสบดีครับ"),
    ]

    ok = 0
    for src, tgt, text in cases:
        t0 = time.time()
        try:
            out = await tr.translate(text, src, tgt, [])
        except Exception as e:
            from translator.pipeline import friendly_error

            print(f"  ✗ {text}\n     {friendly_error(e)}\n     {type(e).__name__}: {e}\n")
            continue
        ok += 1
        print(f"  {'中→泰' if src == 'zh' else '泰→中'}  ({time.time() - t0:.1f}s)")
        print(f"     原文：{text}")
        print(f"     译文：{out}\n")

    if ok == len(cases):
        print("✓ 翻译正常，可以开会了。\n")
    elif ok:
        print(f"! {len(cases) - ok} 句失败，看上面的报错。\n")
    else:
        print("✗ 全部失败。常见原因：Key 填错、账户没充值、需要走中转地址。\n")


async def cmd_minutes(cfg: dict, path: str | None):
    """把某场会的记录整理成中泰双语纪要。不带文件名就用最近一场。"""
    from pathlib import Path

    from translator.minutes import (MinutesMaker, latest_transcript, load_jsonl,
                                    meta_of, save)

    root = Path(__file__).resolve().parent
    tdir = root / str(cfg.get("transcript_dir", "会议记录"))

    src = Path(path) if path else latest_transcript(tdir)
    if src is None or not src.exists():
        print(f"\n✗ 没找到会议记录。{tdir} 里应该有 .jsonl 文件。\n")
        return
    if src.suffix != ".jsonl":
        src = src.with_suffix(".jsonl")

    items = [i for i in load_jsonl(src) if (i.get("text") or "").strip()]
    if not items:
        print(f"\n✗ {src.name} 里没有字幕。\n")
        return

    m = meta_of(items)
    print(f"\n会议记录：{src.name}")
    print(f"  {m['start']:%Y-%m-%d %H:%M} — {m['end']:%H:%M}（约 {m['minutes']} 分钟）")
    print(f"  {m['count']} 句：我方 {m['me']}，对方 {m['them']}，其中泰语 {m['th']}\n")

    try:
        maker = MinutesMaker(cfg)
        md = await maker.make(items)
    except Exception as e:
        print(f"\n✗ 生成失败：{type(e).__name__}: {e}\n")
        return

    paths = save(md, tdir, src.stem)
    print(f"\n✓ 纪要已生成：\n   {paths.get('md')}")
    if paths.get("docx"):
        print(f"   {paths['docx']}")
    print()


class _Tee:
    """终端照常打印，同时抄一份到日志文件。

    回声判定、丢句原因、每分钟的进出账这些只打在控制台，
    开完会窗口一关就查不到了——出了问题没有任何依据可看。
    每写一行就 flush，程序崩了也不会丢掉最后那几行（往往正是关键的那几行）。
    """

    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, s):
        self.stream.write(s)
        try:
            self.file.write(s)
        except Exception:
            pass
        return len(s)

    def flush(self):
        self.stream.flush()
        try:
            self.file.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


def _start_log(cfg: dict):
    """把这场会的终端输出存一份到「会议记录」里，文件名和字幕记录对齐。"""
    if not cfg.get("save_log", True):
        return None
    from datetime import datetime
    from pathlib import Path

    try:
        d = Path(__file__).resolve().parent / str(
            cfg.get("transcript_dir", "会议记录"))
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{datetime.now():%Y-%m-%d_%H%M}_运行日志.txt"
        tee = _Tee(sys.stdout, path)
        sys.stdout = tee
        print(f"[日志] 本场运行日志：{path}")
        return tee
    except Exception as e:
        print(f"[日志] 开不了日志文件（不影响使用）：{e}")
        return None


async def run(cfg: dict):
    from translator.pipeline import Pipeline
    from translator.server import Server

    server = Server(cfg)
    url = await server.run()

    pipeline = Pipeline(cfg, server.broadcast)
    server.pipeline = pipeline

    print(f"\n[界面] {url}")
    print("[界面] 在钉钉里选择「共享窗口」→ 选中这个窗口，对方就能看到字幕。")
    print("[提示] 按 F 切换大字幕，「⋯」菜单里有其它选项，Ctrl+C 退出。\n")

    if cfg["server"].get("open_browser", True):
        from pathlib import Path

        from translator.window import open_window

        def _open():
            print("[界面] " + open_window(url, cfg["server"], Path(__file__).parent))

        threading.Timer(0.8, _open).start()

    try:
        await pipeline.start(asyncio.get_running_loop())
        print("\n★ 已开始监听，现在可以说话了。\n")
        await asyncio.Event().wait()
    finally:
        pipeline.stop()
        if cfg.get("auto_minutes") and pipeline.history:
            print("\n[纪要] auto_minutes 已开启，正在整理这场会的纪要…")
            try:
                await pipeline.make_minutes()
            except Exception as e:
                print(f"[纪要] 生成失败：{type(e).__name__}: {e}")
        # 收掉翻译那边的连接池。不收的话解释器退出时会往日志里吐一串
        # 无关的 traceback，而那个日志是排查问题用的。
        tr = getattr(pipeline, "translator", None)
        if tr is not None and hasattr(tr, "aclose"):
            try:
                await tr.aclose()
            except Exception:
                pass
        await server.close()


def main():
    ap = argparse.ArgumentParser(description="中泰视频会议实时翻译")
    ap.add_argument("--list-devices", action="store_true", help="列出音频输入设备")
    ap.add_argument("--check", action="store_true", help="只检查环境")
    ap.add_argument("--test-audio", action="store_true",
                    help="测试麦克风和系统声音是否都能收到（不需要 API Key）")
    ap.add_argument("--demo", action="store_true",
                    help="用假字幕演示界面效果（不需要 API Key）")
    ap.add_argument("--test-translate", action="store_true",
                    help="只测翻译：真调一次 Claude，确认 API Key 可用")
    ap.add_argument("--test-asr", action="store_true",
                    help="对比本地模型和腾讯云的识别精度与耗时")
    ap.add_argument("--minutes", nargs="?", const="", metavar="记录文件",
                    help="把会议记录整理成中泰双语纪要（不填文件名就用最近一场）")
    ap.add_argument("--seconds", type=int, default=15, help="--test-audio 的测试时长")
    ap.add_argument("--port", type=int, help="网页端口（默认 8765）")
    ap.add_argument("--no-browser", action="store_true", help="启动时不自动打开浏览器")
    args = ap.parse_args()

    if args.list_devices:
        cmd_list_devices()
        return

    print(BANNER)
    cfg = load_config()
    if args.port:
        cfg["server"]["port"] = args.port
    if args.no_browser:
        cfg["server"]["open_browser"] = False

    if args.test_audio:
        cmd_test_audio(cfg, args.seconds)
        return

    if args.test_asr:
        cmd_test_asr(cfg)
        return

    if args.minutes is not None:
        asyncio.run(cmd_minutes(cfg, args.minutes or None))
        return

    if args.test_translate:
        asyncio.run(cmd_test_translate(cfg))
        return

    if args.demo:
        try:
            asyncio.run(cmd_demo(cfg))
        except KeyboardInterrupt:
            print("\n已退出。")
        return

    ok = check(cfg)
    if args.check:
        return
    if not ok:
        print("\n环境检查未通过，请按上面的提示处理后重试。")
        sys.exit(1)

    tee = _start_log(cfg)
    try:
        asyncio.run(run(cfg))
    except KeyboardInterrupt:
        print("\n已退出。")
    finally:
        if tee is not None:
            sys.stdout = tee.stream
            tee.close()


if __name__ == "__main__":
    main()
