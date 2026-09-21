"""腾讯云「一句话识别」。

为什么用它：本机 CPU 跑 Whisper 一句要 5~8 秒，两路加起来跑不赢说话速度，
会积压丢句。走云端后识别只要 1 秒上下，而且不占 CPU、多句能并发，
排队和丢句的问题一起消失。

引擎是按语种分开的（16k_zh / 16k_th），没有能同时自动认中、泰的引擎，
所以语种仍然由本地 tiny 模型先判一次再决定用哪个引擎。
"""
from __future__ import annotations

import base64
import io
import wave

import numpy as np

SR = 16000
MAX_SECONDS = 58.0        # 接口上限 60 秒，留点余量
MAX_BYTES = 3 * 1024 * 1024

ENGINE = {"zh": "16k_zh", "th": "16k_th"}


def _to_wav(audio: np.ndarray) -> bytes:
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return bio.getvalue()


class TencentASR:
    """线程安全：SDK 的 client 可以并发调用，每次请求各自独立。"""

    def __init__(self, cfg: dict):
        self.secret_id = cfg.get("tencent_secret_id") or ""
        self.secret_key = cfg.get("tencent_secret_key") or ""
        self.engines = dict(ENGINE, **(cfg.get("tencent_engines") or {}))
        self._client = None
        # 腾讯云按「调用次数」计费，每月免费 5000 次。数一下让人心里有底。
        self.calls = 0
        self.free_quota = int(cfg.get("tencent_free_quota", 5000))

    def available(self) -> bool:
        return bool(self.secret_id and self.secret_key)

    def load(self) -> str:
        if not self.available():
            raise RuntimeError("没有配置腾讯云 tencent_secret_id / tencent_secret_key")

        from tencentcloud.asr.v20190614 import asr_client
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile

        http = HttpProfile()
        http.reqTimeout = 15          # 开会场景宁可失败也不能卡住
        http.endpoint = "asr.tencentcloudapi.com"
        prof = ClientProfile()
        prof.httpProfile = http

        cred = credential.Credential(self.secret_id, self.secret_key)
        self._client = asr_client.AsrClient(cred, "", prof)
        return "腾讯云一句话识别（" + " / ".join(
            f"{k}={v}" for k, v in self.engines.items()) + "）"

    # ------------------------------------------------------------------
    def transcribe(self, audio: np.ndarray, lang: str) -> str:
        """把一段音频交给云端识别。lang 决定用哪个引擎。"""
        if self._client is None or lang not in self.engines:
            return ""

        seconds = len(audio) / SR
        if seconds > MAX_SECONDS:
            audio = audio[: int(SR * MAX_SECONDS)]

        data = _to_wav(audio)
        if len(data) > MAX_BYTES:
            return ""

        from tencentcloud.asr.v20190614 import models

        req = models.SentenceRecognitionRequest()
        req.EngSerViceType = self.engines[lang]
        req.SourceType = 1                 # 1 = 直接上传音频数据
        req.VoiceFormat = "wav"
        req.Data = base64.b64encode(data).decode()
        req.DataLen = len(data)

        try:
            resp = self._client.SentenceRecognition(req)
        except Exception as e:
            print(f"[腾讯云] 识别失败：{_short(e)}")
            raise
        self.calls += 1

        return (getattr(resp, "Result", "") or "").strip()


def _short(e: Exception) -> str:
    """腾讯云的报错很长，截一段能定位问题的就够。"""
    s = str(e)
    for key in ("AuthFailure", "InvalidParameter", "RequestLimitExceeded",
                "FailedOperation", "UnsupportedOperation"):
        if key in s:
            return f"{key} — {s[:160]}"
    return s[:160]
