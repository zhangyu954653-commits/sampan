"""生成一个可以直接上传到 GitHub 的干净目录。

    .venv\\Scripts\\python.exe 工具\\开源打包.py

为什么要这一步：GitHub 网页上传**不认 .gitignore**——你把文件夹拖上去，
拖了什么就传什么。所以先在本地挑好，再传那个挑好的目录，最稳。

生成的目录里：
  · 有 config.example.json（密钥全空的模板）
  · **没有** config.json（你的密钥）
  · **没有** 会议记录（客诉、报价、产能）
  · **没有** .venv / 浏览器配置这些本机环境

你自己的程序完全不受影响——这个脚本只是复制出去一份，原目录一个字都不动。
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "开源上传"

# 该公开的
FILES = ["main.py", "启动.bat", "requirements.txt", "README.md", "README.en.md",
         "LICENSE", ".gitignore", "config.example.json"]
DIRS = ["translator", "工具", "assets"]

# 目录里也要排除的
SKIP_DIRS = {"__pycache__", ".venv", ".browser-profile", ".git", "会议记录"}
SKIP_SUFFIX = {".pyc", ".pyo", ".log", ".part", ".zip", ".bak", ".tmp"}


def collect() -> list[tuple[Path, str]]:
    out = []
    for name in FILES:
        p = ROOT / name
        if p.exists():
            out.append((p, name))
        else:
            print(f"  ⚠ 缺少 {name}")
    for d in DIRS:
        for p in (ROOT / d).rglob("*"):
            if not p.is_file():
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if p.suffix in SKIP_SUFFIX:
                continue
            out.append((p, p.relative_to(ROOT).as_posix()))
    return out


if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

files = collect()
for src, rel in files:
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

total = sum((OUT / r).stat().st_size for _, r in files)
print(f"\n已生成：{OUT}")
print(f"  {len(files)} 个文件，{total/1024:.0f} KB\n")

# ------------------------------------------------ 自检：确认没有漏东西出去
problems = []
cfg = ROOT / "config.json"
secrets = []
if cfg.exists():
    real = json.loads(cfg.read_text(encoding="utf-8"))
    secrets = [str(x) for x in (
        real.get("anthropic_api_key"), real.get("openai_api_key"),
        real.get("asr", {}).get("tencent_secret_id"),
        real.get("asr", {}).get("tencent_secret_key"),
        real.get("asr", {}).get("tencent_appid")) if x and len(str(x)) > 6]

PAT = [(r"sk-ant-[A-Za-z0-9\-_]{20,}", "Anthropic Key"),
       (r"sk-proj-[A-Za-z0-9\-_]{20,}", "OpenAI Key"),
       (r"\bAKID[A-Za-z0-9]{20,}", "腾讯云 SecretId")]

# 要查的人名/客户名**从你自己的 config.json 里读**，不写死在这个脚本里
# （连举例都不能写——注释一样会被上传，第一次跑就是这么被自己拦下的）。
# 顺带好处：你以后往术语表里加同事名字，这里自动就管得上。
#
# 判据只用两条确定性强的：带称谓后缀的、全大写的客户代号。
# 像「品质」这种普通业务词不该误伤，所以不用"泰文译名像人名"这类猜测。
NAMES = []
if cfg.exists():
    _g = json.loads(cfg.read_text(encoding="utf-8")).get("glossary") or {}
    NAMES = [k for k in _g
             if re.search(r"(经理|总|工|哥|姐|先生|女士)$", k)
             or (k.isascii() and k.isupper() and len(k) >= 2)]

for p in OUT.rglob("*"):
    if not p.is_file() or p.stat().st_size > 2_000_000:
        continue
    try:
        t = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    rel = p.relative_to(OUT).as_posix()
    for s in secrets:
        if s in t:
            problems.append(f"{rel} 里有你的真实密钥")
    for pat, label in PAT:
        if re.search(pat, t):
            problems.append(f"{rel} 里有 {label} 形态的字符串")
    for n in NAMES:
        if n in t:
            problems.append(f"{rel} 里有人名/客户名「{n}」")

print("检查这个目录里有没有不该公开的东西：")
print(f"  {'✓' if not problems else '✗'} 没有密钥")
print(f"  {'✓' if not (OUT / 'config.json').exists() else '✗'} 不含 config.json")
print(f"  {'✓' if not (OUT / '会议记录').exists() else '✗'} 不含会议记录")
print(f"  {'✓' if (OUT / 'config.example.json').exists() else '✗'} 含配置模板")
print(f"  {'✓' if (OUT / 'LICENSE').exists() else '✗'} 含 LICENSE")

if (OUT / "config.example.json").exists():
    e = json.loads((OUT / "config.example.json").read_text(encoding="utf-8"))
    ok = (e.get("anthropic_api_key") == "" and e.get("openai_api_key") == ""
          and e.get("asr", {}).get("tencent_secret_key") == "")
    print(f"  {'✓' if ok else '✗'} 模板里的密钥字段都是空的")
    if not ok:
        problems.append("config.example.json 里有非空的密钥字段")

if problems:
    print(f"\n✗ {len(problems)} 个问题，已删除这个目录：")
    for x in problems:
        print(f"   · {x}")
    shutil.rmtree(OUT, ignore_errors=True)
    sys.exit(1)

print(f"""
✓ 检查通过

下一步（不需要装 git）：
  1. 去 https://github.com/new 建一个空仓库，**不要**勾选任何初始化选项
  2. 建好后点「uploading an existing file」
  3. 把「开源上传」这个文件夹**里面的内容**全选拖进去
     （拖里面的内容，不是拖这个文件夹本身）
  4. 提交

你自己的程序不受任何影响——这个目录是复制出来的，
你的 config.json 和会议记录还在原地，照常开会。
""")
