"""把本机的字幕页面开一条临时公网地址，让总部的人点链接就能看。

为什么要这个：字幕现在只活在跑程序的那台机器上，唯一的输出通道是屏幕共享，
于是共享位被字幕占了，PPT 和客诉模板反而发不出去。开一条公网地址之后，
对方在手机或第二块屏上打开链接看字幕，主屏共享继续放文档。

用的是 Cloudflare 的「快速隧道」：单个 exe，不用注册、不用配置，
跑起来就给一个随机的 https 地址，程序退出隧道就消失。

**安全**：随机地址本身已经猜不到，但链接会被贴进聊天窗口、可能被转发，
所以再加一道口令（每次启动重新生成）。你们聊的是客诉和报价，
不能让人撞进来就看。
"""

from __future__ import annotations

import asyncio
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

# Windows 64 位的单文件版本。其它平台按需换名字。
_DL = {
    ("Windows", "AMD64"): "cloudflared-windows-amd64.exe",
    ("Windows", "ARM64"): "cloudflared-windows-arm64.exe",
    ("Darwin", "arm64"): "cloudflared-darwin-arm64.tgz",
    ("Linux", "x86_64"): "cloudflared-linux-amd64",
}
_BASE = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
_URL_RE = re.compile(r"https://[-\w]+\.trycloudflare\.com")


def make_token() -> str:
    """每次启动换一个，旧链接自动失效。"""
    return secrets.token_urlsafe(9)


def find_cloudflared(root: Path) -> Path | None:
    """先看系统里有没有，再看我们自己下过没有。"""
    got = shutil.which("cloudflared")
    if got:
        return Path(got)
    local = root / "cloudflared.exe"
    if local.exists():
        return local
    local2 = root / "cloudflared"
    if local2.exists():
        return local2
    return None


def download(root: Path) -> Path | None:
    """下载单文件 cloudflared。约 30MB，只下一次。"""
    key = (platform.system(), platform.machine())
    name = _DL.get(key)
    if not name:
        print(f"[分享] 这个平台没有现成的 cloudflared（{key}），"
              f"请手动装一个放到项目目录")
        return None
    if name.endswith(".tgz"):
        print("[分享] macOS 请先自行安装：brew install cloudflared")
        return None

    import urllib.request

    target = root / ("cloudflared.exe" if platform.system() == "Windows"
                     else "cloudflared")
    tmp = target.with_suffix(".part")
    print(f"[分享] 首次使用，正在下载 cloudflared（约 30MB，只下这一次）…")
    try:
        with urllib.request.urlopen(_BASE + name, timeout=120) as r, \
                open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r[分享]   {done*100//total}%", end="")
        print()
        os.replace(tmp, target)
        if platform.system() != "Windows":
            os.chmod(target, 0o755)
        return target
    except Exception as e:
        print(f"\n[分享] 下载失败：{e}")
        try:
            tmp.unlink()
        except Exception:
            pass
        return None


class Tunnel:
    def __init__(self, port: int, root: Path):
        self.port = int(port)
        self.root = root
        self.proc: subprocess.Popen | None = None
        self.url = ""
        self.error = ""

    async def start(self, timeout: float = 30.0) -> tuple[bool, str]:
        """拉起隧道，等它把公网地址吐出来。"""
        exe = find_cloudflared(self.root)
        if exe is None:
            exe = await asyncio.get_running_loop().run_in_executor(
                None, download, self.root)
        if exe is None:
            self.error = "没有 cloudflared，装不了隧道"
            return False, self.error

        cmd = [str(exe), "tunnel", "--no-autoupdate",
               "--url", f"http://127.0.0.1:{self.port}"]
        flags = 0
        if platform.system() == "Windows":
            flags = subprocess.CREATE_NO_WINDOW      # 别弹个黑框出来
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                bufsize=1, creationflags=flags)
        except Exception as e:
            self.error = f"启动失败：{e}"
            return False, self.error

        # 地址是打在 stderr/stdout 里的，只能读日志捞出来
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            line = await loop.run_in_executor(None, self._readline)
            if line is None:
                break
            m = _URL_RE.search(line or "")
            if m:
                self.url = m.group(0)
                return True, ""
        self.stop()
        self.error = "等了 30 秒没等到公网地址，可能是网络不通"
        return False, self.error

    def _readline(self):
        if self.proc is None or self.proc.stdout is None:
            return None
        try:
            return self.proc.stdout.readline()
        except Exception:
            return None

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except Exception:
                    self.proc.kill()
            except Exception:
                pass
            self.proc = None
        self.url = ""

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None
