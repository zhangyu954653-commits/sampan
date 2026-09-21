"""用浏览器的「应用窗口」模式打开字幕页。

普通浏览器标签页里，标签栏 + 地址栏 + 书签栏会吃掉窗口上半部分——
字幕窗口本来就只占屏幕 1/4，这部分损耗无法接受。

Chrome / Edge 的 --app= 模式只画页面本身：没有标签、没有地址栏、没有书签，
只有一条写着页面标题的系统标题栏。在钉钉里按普通窗口共享即可。

用独立的 user-data-dir 有两个好处：
  1. --window-size 只在新开浏览器进程时生效，共用默认配置时会被忽略
  2. 干净环境，不带扩展和书签，窗口大小位置也单独记住
"""
from __future__ import annotations

import os
import shutil
import subprocess
import webbrowser
from pathlib import Path

WINDOW_TITLE = "中泰实时翻译字幕"      # 就是页面的 <title>，应用窗口的标题栏只显示它

CANDIDATES = [
    ("Chrome", [
        r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
        r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
        r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
    ]),
    ("Edge", [
        r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
        r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
    ]),
]


def find_browser() -> tuple[str, str] | None:
    for name, paths in CANDIDATES:
        for p in paths:
            full = os.path.expandvars(p)
            if os.path.isfile(full):
                return name, full
    for name, exe in (("Chrome", "chrome"), ("Edge", "msedge")):
        found = shutil.which(exe)
        if found:
            return name, found
    return None


def open_window(url: str, cfg: dict, root: Path) -> str:
    """打开字幕窗口。返回一句说明，用于打印。"""
    mode = str(cfg.get("window_mode", "app")).lower()
    if mode == "none":
        return "未自动打开（window_mode=none）"

    if mode == "app":
        found = find_browser()
        if found:
            name, exe = found
            w, h = (cfg.get("window_size") or [900, 620])[:2]
            profile = root / ".browser-profile"
            args = [
                exe,
                f"--app={url}",
                f"--window-size={int(w)},{int(h)}",
                f"--user-data-dir={profile}",
                "--no-first-run",
                "--no-default-browser-check",
                # 独立配置目录下没必要跑这些
                "--disable-background-networking",
                "--disable-sync",
            ]
            try:
                subprocess.Popen(args, close_fds=True)
                return f"已用 {name} 应用窗口打开（无地址栏/书签栏）"
            except Exception as e:
                print(f"[界面] 应用窗口启动失败（{e}），改用普通浏览器。")
        else:
            print("[界面] 没找到 Chrome / Edge，改用普通浏览器。")

    try:
        webbrowser.open(url)
        return "已用默认浏览器打开"
    except Exception as e:
        return f"打不开浏览器（{e}），请手动访问 {url}"


# ---------------------------------------------------------------- 窗口置顶
# 浏览器没有"总在最前"的开关，只能用 Windows API 把窗口设成 topmost。
# 投屏时字幕不会被别的窗口盖住——共享的是窗口本身，被遮住对方就看不到了。
_HWND_TOPMOST, _HWND_NOTOPMOST = -1, -2
_SWP_NOSIZE, _SWP_NOMOVE, _SWP_NOACTIVATE = 0x0001, 0x0002, 0x0010
_GWL_EXSTYLE, _WS_EX_TOPMOST = -20, 0x0008


def _user32():
    import ctypes
    from ctypes import wintypes

    u = ctypes.windll.user32
    u.SetWindowPos.argtypes = [wintypes.HWND, ctypes.c_void_p, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_uint]
    u.SetWindowPos.restype = wintypes.BOOL
    u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    u.IsWindowVisible.argtypes = [wintypes.HWND]
    return u, ctypes, wintypes


def find_window(title_part: str = WINDOW_TITLE):
    """按标题找窗口。只看可见窗口，避免命中后台的隐藏窗口。"""
    if os.name != "nt":
        return None
    try:
        u, ctypes, wintypes = _user32()
    except Exception:
        return None

    hits = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        if not u.IsWindowVisible(hwnd):
            return True
        n = u.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(hwnd, buf, n + 1)
        if title_part in buf.value:
            hits.append(hwnd)
            return False        # 找到就停
        return True

    try:
        u.EnumWindows(visit, 0)
    except Exception:
        return None
    return hits[0] if hits else None


def set_topmost(on: bool, title_part: str = WINDOW_TITLE) -> bool:
    """把字幕窗口设为/取消"总在最前"。找不到窗口返回 False。"""
    hwnd = find_window(title_part)
    if not hwnd:
        return False
    try:
        u, ctypes, _ = _user32()
        return bool(u.SetWindowPos(
            hwnd, ctypes.c_void_p(_HWND_TOPMOST if on else _HWND_NOTOPMOST),
            0, 0, 0, 0, _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE))
    except Exception as e:
        print(f"[界面] 置顶失败：{e}")
        return False


def is_topmost(title_part: str = WINDOW_TITLE) -> bool:
    hwnd = find_window(title_part)
    if not hwnd:
        return False
    try:
        import ctypes
        from ctypes import wintypes

        u = ctypes.windll.user32
        f = getattr(u, "GetWindowLongPtrW", None) or u.GetWindowLongW
        f.argtypes = [wintypes.HWND, ctypes.c_int]
        f.restype = ctypes.c_ssize_t
        return bool(f(hwnd, _GWL_EXSTYLE) & _WS_EX_TOPMOST)
    except Exception:
        return False
