"""把流式识别接到管线上。

设计目标只有一个：**下游一行都不用改，而且流式挂了不能丢句子。**

接法：
    断句线程发现某一路开口  → 开一个流式会话，边采边喂
    会话过程中回来的中间结果 → 直接当"正在识别"推给界面（比本地小模型准）
    断句线程判定这一句说完  → 收尾，把最终文本塞进一个 future
    _worker 取到这一句时     → await 那个 future 拿文字，跳过批量识别

所以 seg_q 里多带一个 future，_worker 里多一个分支，别的都不动。

**失败一律退回批量识别**：网络断了、握手被拒、超时、文本为空——
任何一种情况 future 都返回 None，_worker 照老路走一遍。
宁可这一句慢 1 秒，也不能整句丢掉。
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent import futures

import numpy as np

from .tencent_stream import TencentStreamASR

END = object()          # 塞进音频队列表示"这一句说完了"


class _Session:
    """一路声道上的一次发言。

    future 用的是 concurrent.futures 那个而不是 asyncio 的：
    它要在断句线程里创建、在事件循环里填值、再被 _worker await，
    跨线程只有这个是安全的（asyncio.Future 不是线程安全的）。
    _worker 那边用 asyncio.wrap_future() 包一下就能 await。
    """

    def __init__(self, channel: str, lang: str | None):
        self.channel = channel
        self.lang = lang
        self.audio_q: asyncio.Queue = asyncio.Queue()
        self.future: futures.Future = futures.Future()
        self.started = time.time()
        self.text = ""


class StreamManager:
    """在 asyncio 那边跑流式会话，给断句线程提供线程安全的入口。"""

    def __init__(self, cfg: dict, loop: asyncio.AbstractEventLoop,
                 on_partial=None):
        self.asr = TencentStreamASR(cfg.get("asr") or {})
        self.loop = loop
        self.on_partial = on_partial
        self.enabled = bool(cfg.get("stream_asr", False))
        self.sessions: dict[str, _Session] = {}
        self.fallbacks = 0
        self.ok = 0
        self._lock = threading.Lock()
        self._warned = False

    def available(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "未开启（config.json 的 stream_asr）"
        return self.asr.available()

    # ---------------------------------------------------- 断句线程调用的入口
    def feed(self, channel: str, pcm: np.ndarray, lang: str | None):
        """断句线程每采到一块就喂一次。没有会话就开一个。"""
        if not self.enabled:
            return
        with self._lock:
            sess = self.sessions.get(channel)
            if sess is None:
                sess = self._open(channel, lang)
                if sess is None:
                    return
        self.loop.call_soon_threadsafe(sess.audio_q.put_nowait, pcm.copy())

    def finish(self, channel: str):
        """断句线程判定这一句说完了。返回一个 future，_worker 去 await 它。

        返回 None 表示这一路压根没有在跑流式（没开、配置不全、或者刚失败过），
        调用方照老路走批量识别。
        """
        if not self.enabled:
            return None
        with self._lock:
            sess = self.sessions.pop(channel, None)
        if sess is None:
            return None
        self.loop.call_soon_threadsafe(sess.audio_q.put_nowait, END)
        return sess.future

    def cancel(self, channel: str):
        """暂停、静音、被判成回声——这一句不要了。"""
        with self._lock:
            sess = self.sessions.pop(channel, None)
        if sess is not None:
            self.loop.call_soon_threadsafe(sess.audio_q.put_nowait, END)

    def cancel_all(self):
        for ch in list(self.sessions):
            self.cancel(ch)

    # ---------------------------------------------------------------- 内部
    def _open(self, channel: str, lang: str | None):
        """开会话。**一步都不能阻塞断句线程**——那个线程一停，
        audio_q 就没人取，两路音频一起丢帧。所以这里只是建对象 + 派个任务，
        不等任何东西。"""
        ok, why = self.asr.available()
        if not ok:
            if not self._warned:
                print(f"[流式] 用不了，本场继续用一句话识别：{why}")
                self._warned = True
            self.enabled = False
            return None
        sess = _Session(channel, lang)
        self.sessions[channel] = sess
        self.loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(self._run(sess)))
        return sess

    async def _run(self, sess: _Session):
        """跑完一次会话，把结果塞进 future。**任何异常都返回 None 让上游退回批量。**"""

        async def chunks():
            while True:
                item = await sess.audio_q.get()
                if item is END:
                    return
                yield item

        async def on_partial(text, stable):
            sess.text = text
            if self.on_partial:
                try:
                    await self.on_partial(sess.channel, text)
                except Exception:
                    pass

        result = None
        try:
            out = await self.asr.recognize(chunks(), sess.lang or "zh",
                                           on_partial, pace=False)
            # pace=False：音频是麦克风实时采来的，本来就是 1:1，
            # 再 sleep 一次会让收尾比人说完晚整整一个块的时间。
            if out.error:
                print(f"[流式] {sess.channel} 出错，退回一句话识别：{out.error}")
            elif out.text.strip():
                result = (out.text.strip(), sess.lang)
                self.ok += 1
        except Exception as e:
            print(f"[流式] {sess.channel} 异常，退回一句话识别：{type(e).__name__}: {e}")
        if result is None:
            self.fallbacks += 1
        if not sess.future.done():
            sess.future.set_result(result)

    def wrap(self, fut):
        """把跨线程的 future 转成能 await 的。_worker 用。"""
        return asyncio.wrap_future(fut, loop=self.loop) if fut else None

    def report(self) -> str:
        if not (self.ok or self.fallbacks):
            return ""
        return (f"  流式识别：成功 {self.ok} 句，退回一句话识别 {self.fallbacks} 句"
                f"　本场音频 {self.asr.seconds/60:.1f} 分钟"
                f"（跨境版约 ¥{self.asr.seconds/3600*8.6:.2f}）")
