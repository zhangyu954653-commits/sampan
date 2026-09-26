"""GitHub 发布时用：把仓库里的程序打成 Sampan-<版本号>-Windows.zip。

    python .github/scripts/build_release.py v1.0.0

和「工具/打包分发.py」的分工：
  · 打包分发.py 在你本机跑，会带上你调好的配置（密钥清空），可选带术语表；
  · 这个脚本只在 GitHub 上跑，只打包仓库里已提交的文件——仓库里本来就
    没有 config.json 和会议记录。对方第一次运行时，程序会照
    config.example.json 自动生成 config.json。

打完包照样再扫一遍：有密钥形态的字符串、不该有的文件、.bat 换行不对，
都直接报错、删包，不发布。
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
TOP = "Sampan"                                # 解压出来的文件夹名

# 只给开发者看的，不进发布包
SKIP_PREFIX = (".github/", ".githooks/")
SKIP_FILES = {".gitignore", ".gitattributes", "CLAUDE.md"}

# 和「工具/开源前检查.py」同一套密钥形态
PATTERNS = [
    (r"sk-ant-[A-Za-z0-9\-_]{20,}", "Anthropic API Key"),
    (r"sk-proj-[A-Za-z0-9\-_]{20,}", "OpenAI API Key"),
    (r"sk-[A-Za-z0-9]{32,}", "疑似 API Key"),
    (r"\bAKID[A-Za-z0-9]{20,}", "腾讯云 SecretId"),
    (r"AIza[0-9A-Za-z\-_]{30,}", "Google API Key"),
]
PLACEHOLDERS = ("xxx", "fake", "your", "abc123", "example", "...")
FORBIDDEN = ("config.json", "会议记录", ".venv", ".browser-profile", "cloudflared")


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, check=True).stdout
    names = [n for n in out.decode("utf-8").split("\0") if n]
    return [n for n in names
            if not n.startswith(SKIP_PREFIX) and n not in SKIP_FILES]


def read_me() -> str:
    """「先读我.txt」的正文直接取自 打包分发.py，两边永远一致。"""
    src = (ROOT / "工具" / "打包分发.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "READ_ME" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("✗ 工具/打包分发.py 里找不到 READ_ME")


def scan(zip_path: Path) -> list[str]:
    bad = []
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for n in names:
            rel = n.split("/", 1)[1] if "/" in n else n
            if any(x in rel for x in FORBIDDEN):
                bad.append(f"包里不该有：{n}")
            data = z.read(n).decode("utf-8", "ignore")
            for pat, label in PATTERNS:
                for m in re.finditer(pat, data):
                    # 示例/占位符不算（和 开源前检查.py 同一套规则）
                    if any(x in m.group(0).lower() for x in PLACEHOLDERS):
                        continue
                    bad.append(f"{n} 里有 {label} 形态的字符串")
        if f"{TOP}/启动.bat" not in names:
            bad.append("缺少 启动.bat")
        for n in names:
            if not n.endswith(".bat"):
                continue
            bat = z.read(n)
            if b"\r\n" not in bat or bat.count(b"\n") != bat.count(b"\r\n"):
                bad.append(f"{n} 不是 Windows 换行（CRLF），双击可能执行异常")
        if f"{TOP}/先读我.txt" not in names:
            bad.append("缺少 先读我.txt")
    return bad


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "dev"
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    out = dist / f"{TOP}-{version}-Windows.zip"

    files = tracked_files()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            z.write(ROOT / rel, f"{TOP}/{rel}")
        z.writestr(f"{TOP}/先读我.txt", read_me())

    size = out.stat().st_size
    print(f"打好了：{out.name}　{size / 1024:.0f} KB　{len(files) + 1} 个文件")

    bad = scan(out)
    if bad:
        out.unlink()
        print("✗ 有问题，已删除这个包，不发布：")
        for b in bad:
            print(f"   · {b}")
        sys.exit(1)
    print("✓ 包内检查通过：无密钥、无配置和会议记录、所有 .bat 都是 CRLF")


if __name__ == "__main__":
    main()
