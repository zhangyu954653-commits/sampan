"""CI 替身：faster-whisper 要下几百 MB 模型，自动检查用不上。

自检里「识别层兜底」那组测的是 translator/asr.py 的出错兜底逻辑，
不是识别准不准。替身模型什么都听不到（返回空结果），
正好让兜底代码走一遍真实路径。

只在 GitHub 自动检查时通过 PYTHONPATH 生效，不会进发布包，也不影响你本机。
"""


class _Info:
    language = "zh"
    language_probability = 1.0


class WhisperModel:
    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio, **kwargs):
        return iter(()), _Info()

    def detect_language(self, audio, **kwargs):
        return "zh", 1.0, [("zh", 0.5), ("th", 0.5)]
