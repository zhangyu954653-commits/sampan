"""推代码前跑一次，确认仓库里没有密钥和公司信息。

    .venv\\Scripts\\python.exe 工具\\开源前检查.py

为什么需要这个：`.gitignore` 只挡住**还没被提交过**的文件。
一旦某个文件被 git 跟踪过，后来再加进 .gitignore 也拦不住它继续被提交；
而且历史记录里那一版永远都在。所以每次推之前实际查一遍最保险。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

# 密钥的形态。即使本机 config.json 不在了，也能靠这个认出来。
PATTERNS = [
    (r"sk-ant-[A-Za-z0-9\-_]{20,}", "Anthropic API Key"),
    (r"sk-proj-[A-Za-z0-9\-_]{20,}", "OpenAI API Key"),
    (r"sk-[A-Za-z0-9]{32,}", "疑似 API Key"),
    (r"\bAKID[A-Za-z0-9]{20,}", "腾讯云 SecretId"),
    (r"AIza[0-9A-Za-z\-_]{30,}", "Google API Key"),
]
# 绝不该出现在仓库里的路径
FORBIDDEN = ["config.json", "会议记录/", "会议记录\\", ".venv/", ".venv\\",
             ".browser-profile", "cloudflared.exe"]

problems: list[str] = []
notes: list[str] = []


def git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        return r.stdout if r.returncode == 0 else None
    except FileNotFoundError:
        return None


print("=" * 66)
print("一、git 会提交哪些文件")
print("=" * 66)
tracked = git("ls-files")
if tracked is None:
    print("  还没 git init（或没装 git）。按「准备提交的文件」逐个检查：\n")
    files = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if any(x in rel for x in (".venv/", "__pycache__/", ".browser-profile/",
                                  "会议记录/", ".git/")):
            continue
        if rel in ("config.json",) or rel.endswith((".zip", ".pyc")):
            continue
        files.append(rel)
else:
    files = [f for f in tracked.splitlines() if f.strip()]
    print(f"  git 已跟踪 {len(files)} 个文件")
    for f in files:
        if any(x in f for x in FORBIDDEN):
            problems.append(f"git 已经在跟踪不该提交的文件：{f}")

print(f"  待检查 {len(files)} 个文件\n")

print("=" * 66)
print("二、文件内容里有没有密钥")
print("=" * 66)
for rel in files:
    p = ROOT / rel
    if not p.is_file() or p.stat().st_size > 2_000_000:
        continue
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    for pat, label in PATTERNS:
        for m in re.finditer(pat, text):
            s = m.group(0)
            # 示例/占位符不算
            if any(x in s.lower() for x in ("xxx", "fake", "your", "abc123",
                                            "example", "...")):
                continue
            line = text[:m.start()].count("\n") + 1
            problems.append(f"{rel}:{line} 发现 {label}：{s[:14]}…")
print("  ✓ 没有发现密钥" if not problems else f"  ✗ {len(problems)} 处")

print("\n" + "=" * 66)
print("三、和本机真实配置比对（最可靠的一道）")
print("=" * 66)
cfg = ROOT / "config.json"
if not cfg.exists():
    print("  本机没有 config.json，跳过")
else:
    real = json.loads(cfg.read_text(encoding="utf-8"))
    secrets = [str(x) for x in (
        real.get("anthropic_api_key"), real.get("openai_api_key"),
        real.get("asr", {}).get("tencent_secret_id"),
        real.get("asr", {}).get("tencent_secret_key"),
        real.get("asr", {}).get("tencent_appid")) if x and len(str(x)) > 6]
    # 人名从你自己的 config.json 里读，连举例都不写在脚本里——
    # 写死的话这个脚本本身就成了泄漏源
    _g = real.get("glossary") or {}
    names = [k for k in _g
             if re.search(r"(经理|总|工|哥|姐|先生|女士)$", k)
             or (k.isascii() and k.isupper() and len(k) >= 2)]
    hit = 0
    for rel in files:
        p = ROOT / rel
        if not p.is_file() or p.stat().st_size > 2_000_000:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for s in secrets:
            if s in text:
                problems.append(f"{rel} 里出现了你真实的密钥/AppID")
                hit += 1
        for n in names:
            if n in text:
                notes.append(f"{rel} 里出现了术语表中的人名「{n}」")
    print(f"  {'✓' if not hit else '✗'} 没有本机密钥泄漏"
          f"（比对了 {len(secrets)} 个凭据）")
    print(f"  {'✓' if not notes else '⚠'} 人名检查"
          f"（术语表里的人名 {len(names)} 个）")

print("\n" + "=" * 66)
print("四、.gitignore 是否到位")
print("=" * 66)
gi = ROOT / ".gitignore"
if not gi.exists():
    problems.append("没有 .gitignore —— 一次 `git add .` 就会把密钥推上去")
    print("  ✗ 不存在")
else:
    body = gi.read_text(encoding="utf-8")
    for must in ("config.json", "会议记录", ".venv", "__pycache__"):
        ok = must in body
        print(f"  {'✓' if ok else '✗'} 忽略 {must}")
        if not ok:
            problems.append(f".gitignore 里没有忽略 {must}")

print("\n" + "=" * 66)
print("五、给使用者的配置模板")
print("=" * 66)
ex = ROOT / "config.example.json"
if not ex.exists():
    problems.append("缺少 config.example.json，别人不知道该怎么配")
    print("  ✗ 不存在")
else:
    e = json.loads(ex.read_text(encoding="utf-8"))
    for k, v in (("anthropic_api_key", e.get("anthropic_api_key")),
                 ("openai_api_key", e.get("openai_api_key")),
                 ("asr.tencent_secret_key",
                  e.get("asr", {}).get("tencent_secret_key")),
                 ("asr.tencent_appid", e.get("asr", {}).get("tencent_appid"))):
        ok = v == ""
        print(f"  {'✓' if ok else '✗'} {k} 是空的")
        if not ok:
            problems.append(f"config.example.json 的 {k} 不是空的")

print("\n" + "=" * 66)
if problems:
    print(f"✗ {len(problems)} 个问题，**先别推**")
    for p in problems:
        print(f"   · {p}")
    if notes:
        print("\n  另外这些值得看一眼（不一定是问题）：")
        for n in notes:
            print(f"   · {n}")
    sys.exit(1)
print("✓ 检查通过，可以推了")
if notes:
    print("\n  值得看一眼（不一定是问题）：")
    for n in notes:
        print(f"   · {n}")
