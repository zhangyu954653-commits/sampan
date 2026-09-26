"""打一个可以发给别人的干净压缩包。

    .venv\\Scripts\\python.exe 打包分发.py

直接把整个文件夹压缩发出去会出两件大事：

  config.json   里面有你的 3 把密钥（Anthropic + 腾讯云 SecretId/SecretKey）。
                别人拿到就能用你的账号无限跑，账单算你的。
  会议记录/      客诉、报价、产能、工厂规划——全是公司业务内容。

所以这个脚本只带程序本身（不到 1MB），配置换成空白模板，
并且**打完包会自己检查一遍，确认里面没有任何密钥**。
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent      # 这个脚本在「工具」子目录里
TOP = "Sampan"                                     # 对方解压出来的文件夹名

# 要带的：程序本身
INCLUDE = ["main.py", "启动.bat", "requirements.txt", "README.md",
           "README.en.md", "LICENSE"]
INCLUDE_DIRS = ["translator", "工具", "assets"]

# 绝对不能带的
SKIP_DIRS = {"__pycache__", ".venv", ".browser-profile", ".git", "会议记录"}
SKIP_SUFFIX = {".pyc", ".pyo", ".log", ".part"}

# 密钥字段：模板里一律留空，让对方填自己的
SECRET_KEYS = [
    ("anthropic_api_key",), ("openai_api_key",),
    ("asr", "tencent_secret_id"), ("asr", "tencent_secret_key"),
    ("asr", "tencent_appid"), ("asr", "openai_api_key"),
]
# 业务内容：术语表和纠错表带不带由用户选（里面有客户名、人名）
BUSINESS_KEYS = ["glossary", "asr_corrections", "domain_hint"]


def clean_config(keep_terms: bool) -> str:
    raw = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    for path in SECRET_KEYS:
        d = raw
        for k in path[:-1]:
            d = d.get(k, {})
        if isinstance(d, dict) and path[-1] in d:
            d[path[-1]] = ""
    # 每台机器自己判断，不该跟着模板跑
    raw["language_lock"] = {"me": None, "them": None}
    raw["language_bias"] = {"me": None, "them": None}
    raw["stream_asr"] = False
    if not keep_terms:
        raw["glossary"] = {}
        raw["asr_corrections"] = {}
        raw["domain_hint"] = "线上商务会议"
    return json.dumps(raw, ensure_ascii=False, indent=2)


def collect() -> list[tuple[Path, str]]:
    out = []
    for name in INCLUDE:
        p = ROOT / name
        if p.exists():
            out.append((p, name))
    for d in INCLUDE_DIRS:
        for p in (ROOT / d).rglob("*"):
            if not p.is_file():
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if p.suffix in SKIP_SUFFIX:
                continue
            out.append((p, str(p.relative_to(ROOT)).replace("\\", "/")))
    return out


def main():
    keep = "--带术语表" in sys.argv or "--with-glossary" in sys.argv
    files = collect()
    stamp = datetime.now().strftime("%Y%m%d")
    out = ROOT / f"{TOP}_{stamp}.zip"

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in files:
            z.write(src, f"{TOP}/{arc}")
        z.writestr(f"{TOP}/config.json", clean_config(keep))
        z.writestr(f"{TOP}/先读我.txt", READ_ME)

    size = out.stat().st_size
    print(f"打好了：{out.name}　{size/1024:.0f} KB　{len(files)+2} 个文件\n")

    # ---- 自检：确认包里没有任何密钥 ----
    print("检查包里有没有漏出去的东西：")
    bad = []
    real = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    secrets = [str(real.get("anthropic_api_key") or ""),
               str(real.get("asr", {}).get("tencent_secret_id") or ""),
               str(real.get("asr", {}).get("tencent_secret_key") or ""),
               str(real.get("openai_api_key") or "")]
    secrets = [s for s in secrets if len(s) > 8]

    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        for n in names:
            if n.endswith("/"):
                continue
            try:
                data = z.read(n).decode("utf-8", "ignore")
            except Exception:
                continue
            for s in secrets:
                if s and s in data:
                    bad.append(f"{n} 里出现了密钥！")
            if re.search(r"sk-ant-[A-Za-z0-9\-_]{20,}", data):
                bad.append(f"{n} 里有 Anthropic 形态的密钥")
            if re.search(r"\bAKID[A-Za-z0-9]{20,}", data):
                bad.append(f"{n} 里有腾讯云形态的密钥")

        cfg = json.loads(z.read(f"{TOP}/config.json"))
        print(f"  {'✓' if not bad else '✗'} 没有密钥泄漏")
        for k in ("anthropic_api_key", "openai_api_key"):
            ok = cfg.get(k) == ""
            print(f"  {'✓' if ok else '✗'} {k} 已清空")
            if not ok:
                bad.append(k)
        for k in ("tencent_secret_id", "tencent_secret_key", "tencent_appid"):
            ok = cfg.get("asr", {}).get(k) == ""
            print(f"  {'✓' if ok else '✗'} asr.{k} 已清空")
            if not ok:
                bad.append(k)
        leaked = [n for n in names if "会议记录" in n]
        print(f"  {'✓' if not leaked else '✗'} 不含会议记录"
              + (f"（发现 {leaked[:3]}）" if leaked else ""))
        if leaked:
            bad += leaked
        heavy = [n for n in names if ".venv" in n or "browser-profile" in n]
        print(f"  {'✓' if not heavy else '✗'} 不含本机环境")
        if heavy:
            bad += heavy
        print(f"  {'·'} 术语表：{'带了 ' + str(len(cfg.get('glossary') or {})) + ' 条'if keep else '已清空'}"
              f"（加 --带术语表 可以带上）")

    if bad:
        print(f"\n✗ 有问题，已删除这个包：{bad[:4]}")
        out.unlink()
        sys.exit(1)
    print(f"\n✓ 检查通过，可以发了：{out}")
    print("  对方解压后双击「启动.bat」，第一次会自动装环境（3~10 分钟），")
    print("  然后照「先读我.txt」填自己的 Key。")


READ_ME = """Sampan · 中泰实时翻译字幕 —— 拿到之后怎么开始

1. 双击「启动.bat」
   第一次会自动建 Python 环境并安装依赖，大约 3~10 分钟。
   没装 Python 的话，先去 https://www.python.org/downloads/ 装 3.10 以上版本，
   安装时记得勾选 "Add Python to PATH"。

2. 填自己的 Key（用记事本打开 config.json）

   "anthropic_api_key": "sk-ant-..."        翻译用
   "asr": {
     "tencent_secret_id":  "...",           语音识别用
     "tencent_secret_key": "..."
   }

   两个都要自己申请，不要用别人的——账单算在 Key 的主人头上。

   ⚠ 重要：如果你在中国大陆，api.anthropic.com 是连不上的，
     需要配 "anthropic_base_url" 指向可用的中转地址，否则翻译会一直失败。
     在泰国等境外地区直连即可。

3. 开会时
   · 钉钉里「共享窗口」→ 选中字幕窗口，对方就能看到字幕
   · 或者更好的办法：⋯ 菜单 →「🔗 分享字幕」，生成一条链接发给对方，
     对方点开就能看，你的屏幕共享可以继续放文档
   · 会开完记得按「暂停」，不然旁边说话也会一直识别、一直花钱

4. 遇到问题
   先跑一次自检：在这个文件夹里打开命令行，执行
       .venv\\Scripts\\python.exe 工具\\自检.py
   详细说明看 README.md

（「工具」文件夹里是诊断脚本，平时不用管，出问题时才用到。）
"""


main()
