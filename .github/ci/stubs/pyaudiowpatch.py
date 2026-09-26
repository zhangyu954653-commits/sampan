"""CI 替身：PyAudioWPatch 只有 Windows 有。

GitHub 的检查机是 Linux，没有声卡。自检只测纯逻辑，用不到真的录音，
这里只保证 `import pyaudiowpatch` 不报错。谁要是真去开声卡，会立刻报错，
不会假装成功。

只在 GitHub 自动检查时通过 PYTHONPATH 生效，不会进发布包，也不影响你本机。
"""

paInt16 = 8
paFloat32 = 1
paWASAPI = 13


class PyAudio:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        raise RuntimeError("CI 替身：检查机上没有声卡")


def __getattr__(name):
    return 0
