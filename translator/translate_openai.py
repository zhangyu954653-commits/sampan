"""OpenAI 翻译后端，接口和 Anthropic 那个完全一样，可以直接对调。

单独一个文件、单独一个类，是为了**不碰现在能用的那条路**。
两边共用同一套提示词和术语表，比较的才是模型本身，不是提示词差异。

提示词缓存：OpenAI 是自动的（输入超过一定长度就自动命中，不用像 Anthropic
那样显式打 cache_control 标记），所以系统提示照常整段传就行。
"""

from __future__ import annotations

import asyncio

from .translate import (LANG_NAME, SPEAKER, SYSTEM_TMPL, _build_glossary,
                        _final)


class OpenAITranslator:
    def __init__(self, cfg: dict):
        self.model = cfg.get("openai_translate_model", "gpt-4o-mini")
        self.glossary = cfg.get("glossary") or {}
        self.domain = cfg.get("domain_hint") or "线上商务会议"
        self.context_turns = int(cfg.get("context_turns", 4))
        self.speaker = SPEAKER.get(str(cfg.get("speaker_gender", "")).lower(),
                                   SPEAKER[""])
        self.cache_max_chars = int(cfg.get("cache_max_chars", 28))
        self._cache: dict[tuple, str] = {}
        self._cache_order: list[tuple] = []
        # 正在翻的句子。缓存只在译完之后才写得进去，所以**同一句话同时在飞**
        # 的时候两边都会各打一次 API，还可能译出两个版本。
        # 实测遇到过：本场第一句冷启动要 4.5 秒，第二句 3 秒后就到了，
        # 同样一句"喂，可以听到我说话吗？"译出了 สวัสดีครับ 和 ฮัลโหล 两种。
        self._inflight: dict[tuple, asyncio.Future] = {}

        from openai import AsyncOpenAI

        key = cfg.get("openai_api_key") or ""
        if not key:
            raise RuntimeError(
                "没有找到 OpenAI API Key。\n"
                "请在 config.json 里填 openai_api_key，"
                "或设置环境变量 OPENAI_API_KEY。"
            )
        kwargs = {"api_key": key, "max_retries": 1,
                  "timeout": float(cfg.get("translate_timeout", 20.0))}
        if cfg.get("openai_base_url"):
            kwargs["base_url"] = cfg["openai_base_url"]
        self.client = AsyncOpenAI(**kwargs)

    # ------------------------------------------------------------------
    def _build(self, text: str, src: str, tgt: str,
               context: list[dict] | None, fragment: bool = False) -> dict:
        """和 Anthropic 那边同一套提示词，只是换成 OpenAI 的消息格式。"""
        system = SYSTEM_TMPL.format(
            domain=self.domain,
            speaker=self.speaker,
            src=LANG_NAME.get(src, src),
            tgt=LANG_NAME.get(tgt, tgt),
            glossary=_build_glossary(self.glossary),
        )

        parts = []
        ctx = (context or [])[-self.context_turns:]
        if ctx:
            lines = []
            for c in ctx:
                who = "我方" if c["channel"] == "me" else "对方"
                lines.append(f"{who}原文：{c['text']}")
                if c.get("translation"):
                    lines.append(f"{who}译文：{c['translation']}")
            parts.append(
                "<会议上文 仅供参考，不要翻译这部分。"
                "请延续上文的用词、称谓和语气>\n" + "\n".join(lines) + "\n</会议上文>"
            )
        if fragment:
            parts.append(
                "注意：说话人还没说完，下面只是他这段话的**前半截**。"
                "照着现有内容翻就行，不要替他补完整、不要加句号收尾、"
                "不要加「ครับ/ค่ะ」这类收尾敬语。"
            )
        parts.append(f"<需要翻译的这一句>\n{text}\n</需要翻译的这一句>")

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "\n\n".join(parts)},
            ],
        }
        # 泰文分词很占 token，长发言的译文可能上千，给足余量免得被截断。
        #
        # 新一代模型用 max_completion_tokens 且不接受 temperature。
        # 这里按**老模型白名单**判断，不是按新模型前缀——模型迭代很快，
        # 写成 startswith("gpt-5") 那种，下次出 gpt-6 就又漏了（实测踩过）。
        m = self.model.lower()
        legacy = m.startswith(("gpt-3", "gpt-4")) and "realtime" not in m
        if legacy:
            kwargs["max_tokens"] = 2000
            kwargs["temperature"] = 0.0      # 翻译要稳定，不要发挥
        else:
            kwargs["max_completion_tokens"] = 2000
        return kwargs

    async def aclose(self):
        """退出时收掉 HTTP 连接池，免得在解释器关闭阶段报一串无关的错。"""
        try:
            await self.client.close()
        except Exception:
            pass

    async def translate(self, text: str, src: str, tgt: str,
                        context: list[dict] | None = None) -> str:
        resp = await self.client.chat.completions.create(
            **self._build(text, src, tgt, context))
        return _final(resp.choices[0].message.content or "")

    # ------------------------------------------------------------------
    def _cache_key(self, text: str, src: str, tgt: str):
        import re

        norm = re.sub(r"[\s，。,.!！?？~～…]+", "", text)
        if not norm or len(norm) > self.cache_max_chars:
            return None
        return (src, tgt, norm)

    def _cache_put(self, key, value: str):
        if key is None or not value:
            return
        if key not in self._cache:
            self._cache_order.append(key)
            if len(self._cache_order) > 300:
                self._cache.pop(self._cache_order.pop(0), None)
        self._cache[key] = value

    async def translate_stream(self, text: str, src: str, tgt: str,
                               context: list[dict] | None,
                               on_delta, fragment: bool = False) -> str:
        key = None if fragment else self._cache_key(text, src, tgt)
        hit = self._cache.get(key) if key else None
        if hit:
            await on_delta(hit)
            return hit

        # 同一句话已经在翻了：等它的结果，别再打一次 API。
        # 这样既省钱，也保证同一句话在这场会里译法一致。
        if key is not None and key in self._inflight:
            try:
                out = await asyncio.shield(self._inflight[key])
                if out:
                    await on_delta(out)
                    return out
            except Exception:
                pass          # 那一边失败了，自己重新翻一次

        fut = asyncio.get_running_loop().create_future() if key else None
        if fut is not None:
            self._inflight[key] = fut

        buf = []
        try:
            stream = await self.client.chat.completions.create(
                stream=True, **self._build(text, src, tgt, context, fragment))
            # 必须显式关流。不关的话底层 httpx 的异步生成器会留到解释器退出时
            # 才被回收，那时循环已经没了，于是往终端吐一串
            # "generator didn't stop after athrow()" 的 traceback——
            # 运行日志是我们排查问题用的，不能被这种噪音淹掉。
            try:
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    piece = chunk.choices[0].delta.content
                    if not piece:
                        continue
                    buf.append(piece)
                    await on_delta(piece)
            finally:
                try:
                    await stream.close()
                except Exception:
                    pass

            out = _final("".join(buf))
            self._cache_put(key, out)
            if fut is not None and not fut.done():
                fut.set_result(out)          # 等着的那一边可以直接用了
            return out
        except Exception as e:
            # 失败也要唤醒等待方，否则它会一直挂着
            if fut is not None and not fut.done():
                fut.set_exception(e)
            raise
        finally:
            if key is not None:
                self._inflight.pop(key, None)
