"""腾讯云「实时语音识别」——WebSocket 双向流式。

和现在用的「一句话识别」的区别，全在一件事上：

    一句话识别：等人说完 → 整段音频发出去 → 等 0.9~1.9 秒 → 拿到文字
    实时识别　：边说边发 → 说的过程中文字就在回来 → 说完只剩收尾的 0.2~0.3 秒

一段 7 秒的发言，前者那 7 秒里识别引擎完全没开工；后者已经把活干完了。
实测这一段占端到端延迟的 1.4 秒，是除了断句等待之外最大的一块。

**计费方式和一句话识别完全不同**：按音频秒数算，每月免费 5 小时
（一句话识别是每月免费 5000 次）。所以这里**只在有人说话时才连**，
不能挂着连续推流——一场两小时的会两路就是 4 小时，一场用掉 80% 的免费额度。

协议要点（来自腾讯云文档）：
  · 地址   wss://asr.cloud.tencent.com/asr/v2/<appid>?{参数}&signature={签名}
  · 签名   除 signature 外所有参数按字典序排 → 拼成不含 wss:// 的 URL
           → HMAC-SHA1(secretkey) → base64 → urlencode
  · 发送   16k PCM，建议每 200ms 发 200ms（6400 字节）。
           **快于 1:1 实时率或间隔超过 6 秒，引擎会报错并断开。**
  · 结束   发 {"type":"end"}
  · 返回   result.slice_type  0=开始  1=识别中（还会变）  2=这段说完了（稳定）
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.parse
import uuid

import numpy as np

HOST = "asr.cloud.tencent.com"
PATH = "/asr/v2/"
SR = 16000
CHUNK_MS = 200                       # 文档建议值
CHUNK_BYTES = SR * 2 * CHUNK_MS // 1000      # 16k × 2 字节 × 0.2s = 6400


def build_url(appid: str, secret_id: str, secret_key: str, engine: str,
              voice_id: str, *, now: int | None = None,
              nonce: int | None = None, extra: dict | None = None) -> str:
    """拼出带签名的 wss 地址。

    签名原文必须**按字典序**排参数，而且不含 "wss://"。顺序错了就鉴权失败，
    而腾讯云返回的错误只说"签名不对"，不会告诉你错在哪——所以这里排序、
    编码两步都不能省事。
    """
    ts = int(now if now is not None else time.time())
    params = {
        "secretid": secret_id,
        "timestamp": ts,
        "expired": ts + 3600,
        "nonce": int(nonce if nonce is not None else ts),
        "engine_model_type": engine,
        "voice_id": voice_id,
        "voice_format": 1,            # 1 = pcm
        "needvad": 1,                 # 让引擎自己切句，超过 60 秒也不会出错
        "filter_dirty": 0,
        "filter_modal": 0,            # 语气词我们自己有一套过滤，别让它插手
        "filter_punc": 0,
        "filter_empty_result": 1,
    }
    if extra:
        params.update(extra)

    # 字典序 + 不含协议头 = 签名原文
    ordered = "&".join(f"{k}={params[k]}" for k in sorted(params))
    raw = f"{HOST}{PATH}{appid}?{ordered}"
    sign = base64.b64encode(
        hmac.new(secret_key.encode("utf-8"), raw.encode("utf-8"),
                 hashlib.sha1).digest()).decode()
    # signature 必须 urlencode：里面会出现 + 和 =，不编码会偶发鉴权失败
    return (f"wss://{raw}&signature="
            f"{urllib.parse.quote(sign, safe='')}")


def to_pcm16(audio: np.ndarray) -> bytes:
    """float32 [-1,1] → 16bit 小端 PCM。"""
    a = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (a * 32767.0).astype("<i2").tobytes()


class StreamResult:
    """一次流式识别的过程和结果。"""

    def __init__(self):
        self.finals: list[str] = []      # slice_type=2，稳定下来的句子
        self.partial = ""                # slice_type=1，还会变的那截
        self.error = ""

    @property
    def text(self) -> str:
        """最终文本 = 所有稳定段拼起来（不含还在变的那截）。"""
        return "".join(self.finals)

    @property
    def live(self) -> str:
        """当前能显示给人看的全部内容，含还在变的那截。"""
        return self.text + self.partial


class TencentStreamASR:
    def __init__(self, cfg: dict):
        self.appid = str(cfg.get("tencent_appid") or "").strip()
        self.secret_id = str(cfg.get("tencent_secret_id") or "").strip()
        self.secret_key = str(cfg.get("tencent_secret_key") or "").strip()
        self.engines = cfg.get("tencent_engines") or {"zh": "16k_zh",
                                                      "th": "16k_th"}
        self.hotword_id = str(cfg.get("tencent_hotword_id") or "").strip()
        self.connect_timeout = float(cfg.get("stream_connect_timeout", 5.0))
        self.idle_timeout = float(cfg.get("stream_idle_timeout", 8.0))
        self.calls = 0
        self.seconds = 0.0               # 累计送出去的音频秒数（= 计费口径）

    def available(self) -> tuple[bool, str]:
        if not self.appid:
            return False, ("缺少 tencent_appid。实时识别的地址里要带 AppID，"
                           "在腾讯云「访问密钥」页面能看到（就是账号 APPID，"
                           "一串数字），填进 config.json 的 asr.tencent_appid")
        if not (self.secret_id and self.secret_key):
            return False, "缺少腾讯云 SecretId / SecretKey"
        return True, ""

    async def probe(self) -> tuple[bool, str]:
        """连一下试试，马上知道能不能用。

        用户在界面上点「极速」时调这个：要么立刻告诉他开好了，
        要么立刻告诉他缺什么。不能等他开完会才发现每句话都在偷偷退回批量。
        只发 0.3 秒音频，花掉的额度可以忽略。
        """
        ok, why = self.available()
        if not ok:
            return False, why

        async def tiny():
            yield np.zeros(int(SR * 0.3), dtype=np.float32)

        out = await self.recognize(tiny(), "zh", pace=False)
        if out.error:
            if "6001" in out.error or "跨境" in out.error:
                return False, ("腾讯云还没开通「跨境」服务。你这台机器在境外，"
                               "实时识别要单独开通：\n"
                               "  https://console.cloud.tencent.com/asr/settings\n"
                               "  开通跨境服务 + 开启后付费（¥8.6/小时音频，无免费额度）")
            return False, out.error
        return True, ""

    # ------------------------------------------------------------------
    async def recognize(self, chunks, lang: str, on_partial=None,
                        pace: bool = True) -> StreamResult:
        """把音频流送进去，边送边收。

        chunks: 异步可迭代对象，每次给一段 float32 音频（长度不限，内部会重切）。
                管线那边是"麦克风采到一块就 yield 一块"，天然就是 1:1 实时率。
        pace:   True 时按 1:1 实时率发（发快了引擎会报错断开）。
                用文件回放测试时也必须开着，否则一次性灌进去必挂。
        """
        import aiohttp

        ok, why = self.available()
        out = StreamResult()
        if not ok:
            out.error = why
            return out

        engine = self.engines.get(lang, self.engines.get("zh", "16k_zh"))
        voice_id = uuid.uuid4().hex
        extra = {"hotword_id": self.hotword_id} if self.hotword_id else None
        url = build_url(self.appid, self.secret_id, self.secret_key,
                        engine, voice_id, extra=extra)

        timeout = aiohttp.ClientTimeout(total=None,
                                        sock_connect=self.connect_timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.ws_connect(url, heartbeat=None) as ws:
                    await self._pump(ws, chunks, out, on_partial, pace)
        except Exception as e:
            if not out.error:
                out.error = f"{type(e).__name__}: {e}"
        return out

    async def _pump(self, ws, chunks, out: StreamResult, on_partial, pace):
        import asyncio

        # 握手：第一条消息是服务端确认，code!=0 就是参数或签名有问题
        first = await asyncio.wait_for(ws.receive(),
                                       timeout=self.connect_timeout)
        if first.type != 1:                      # 1 = TEXT
            out.error = f"握手异常：{first.type} {str(first.data)[:120]}"
            return
        hello = json.loads(first.data)
        if hello.get("code") != 0:
            out.error = f"握手被拒（code={hello.get('code')}）：{hello.get('message')}"
            return
        self.calls += 1

        done = asyncio.Event()

        async def reader():
            try:
                while not done.is_set():
                    msg = await asyncio.wait_for(ws.receive(),
                                                 timeout=self.idle_timeout)
                    if msg.type != 1:
                        break
                    data = json.loads(msg.data)
                    if data.get("code") != 0:
                        out.error = (f"识别出错（code={data.get('code')}）："
                                     f"{data.get('message')}")
                        break
                    r = data.get("result") or {}
                    txt = (r.get("voice_text_str") or "").strip()
                    st = r.get("slice_type")
                    if st == 2 and txt:
                        out.finals.append(txt)
                        out.partial = ""
                        if on_partial:
                            await on_partial(out.live, True)
                    elif st == 1:
                        out.partial = txt
                        if on_partial:
                            await on_partial(out.live, False)
                    if data.get("final") == 1:
                        break
            except asyncio.TimeoutError:
                if not out.error:
                    out.error = "等结果超时"
            except Exception as e:
                if not out.error:
                    out.error = f"读取出错：{type(e).__name__}: {e}"
            finally:
                done.set()

        rt = asyncio.create_task(reader())
        try:
            buf = b""
            async for block in chunks:
                if done.is_set():
                    break
                buf += to_pcm16(block)
                while len(buf) >= CHUNK_BYTES:
                    await ws.send_bytes(buf[:CHUNK_BYTES])
                    buf = buf[CHUNK_BYTES:]
                    self.seconds += CHUNK_MS / 1000
                    if pace:
                        # 发快了引擎会报错断开，必须压到 1:1 实时率
                        await asyncio.sleep(CHUNK_MS / 1000)
            if buf and not done.is_set():
                await ws.send_bytes(buf)         # 最后不足一块的尾巴
                self.seconds += len(buf) / (SR * 2)
            if not done.is_set():
                await ws.send_str(json.dumps({"type": "end"}))
            await asyncio.wait_for(rt, timeout=self.idle_timeout)
        except asyncio.TimeoutError:
            if not out.error:
                out.error = "收尾超时"
        finally:
            done.set()
            if not rt.done():
                rt.cancel()
