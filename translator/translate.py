"""调用 Claude 做中↔泰翻译。为会议口语场景优化：短、准、只出译文。"""
from __future__ import annotations

import asyncio
import re

LANG_NAME = {"zh": "中文（简体）", "th": "泰语"}

SYSTEM_TMPL = """你是视频会议的同声传译引擎，负责把发言逐句翻译。

场景：{domain}
本次任务：把【{src}】翻译成【{tgt}】。

{speaker}
硬性规则：
1. 只输出译文本身。不要解释、不要拼音/音标、不要加引号、不要写"翻译："之类的前缀。
   译文里**绝对不能残留源语言的文字**：中译泰时不许出现任何汉字，
   泰译中时不许出现任何泰文字母（阿拉伯数字和英文型号除外）。
2. **必须是口语**，就像现场翻译在开口说话。
   - 泰语禁止用「สามารถ...ได้หรือไม่」这类公文腔，要用「...ไหม / ...มั้ย」这种日常问法。
   - 中文禁止用"是否""能否""予以"这类书面语，要用"能不能""行不行"。
   - 句子要短。一句话能说清就不要拆成两句。
3. 同一句话每次都要翻成同一个说法，不要换着花样说。
4. 数字、金额、日期、百分比、型号必须准确无误。
   **所有数目一律写成阿拉伯数字**（120 / 3000 / 2),不要拼写成文字。
   字幕会把数字标色供双方核对，拼成文字就核对不了。
   但**量词、单位、月份等词必须用目标语言**，绝对不能出现源语言的字：
   泰语要写「120 ชิ้น」「3000 ชิ้น」「วันที่ 2 ถึง 13 ตุลาคม」，
   中文要写「120 件」「3000 件」「10月2日到13日」。
5. 人名、公司名、地名按术语表；术语表没有的，中译泰时保留原文或用通用音译，泰译中时保留原文拼写。
6. 原文有口误、重复、结巴时，翻成通顺的说法。
7. 原文来自语音识别，**经常有听错的字**，尤其是泰语。请按发音相近的原则还原：
   - 优先往商务会议里的常用词上猜（星期几、月份、数量、单位、金额、部门、单据名称），
     不要往罕见词、宗教节日、人名地名上猜。
   - 例：泰语里像"วันพระรุษฐราบดี"这种拼不出来的，应还原成发音最接近的
     常用词"วันพฤหัสบดี"（星期四），而不是理解成某个节日。
   - 例："ใบเสนอรักค้า"应还原成"ใบเสนอราคา"（报价单）。
8. **日期、数量、金额宁可保守也不要编。** 如果某个数字或日期实在辨认不出，
   就用模糊说法（比如"这周晚些时候"），不要凭空给一个具体的。
9. 如果原文没有实质内容（只是"嗯""啊""喂喂"、咳嗽、噪音），输出两个减号：--
{glossary}"""


SPEAKER = {
    "male": "说话人是男性：泰语第一人称一律用「ผม」，句尾敬语一律用「ครับ」，"
            "绝不要用「ฉัน / ดิฉัน / ค่ะ」。",
    "female": "说话人是女性：泰语第一人称一律用「ดิฉัน」或「เรา」，句尾敬语一律用「ค่ะ」，"
              "绝不要用「ผม / ครับ」。",
    "": "泰语第一人称统一用「เรา」，句尾敬语统一用「ครับ」，全程不要变来变去。",
}


def _build_glossary(glossary: dict) -> str:
    if not glossary:
        return ""
    lines = "\n".join(f"  {k} → {v}" for k, v in glossary.items())
    return (
        f"\n术语表（必须严格遵守）：\n{lines}\n"
        "注意：这些是本次会议的专有词，**语音识别最容易听错的就是它们**。\n"
        "原文里出现发音接近但写错的字（比如「厌厂」其实是「验厂」），"
        "一律按术语表还原后再翻译。\n"
    )


def _strip(text: str) -> str:
    t = (text or "").strip()
    # 关掉思考后，模型偶尔会把 <thinking> 标签漏进正文，清掉
    t = re.sub(r"<\/?(thinking|antml:thinking)[^>]*>", "", t, flags=re.I).strip()
    t = re.sub(r"^(译文|翻译|Translation)\s*[:：]\s*", "", t, flags=re.I)
    if len(t) >= 2 and t[0] in "「『\"'“‘《" and t[-1] in "」』\"'”’》":
        t = t[1:-1].strip()
    return t


def _final(raw: str) -> str:
    out = _strip(raw)
    if out in ("--", "—", "-"):
        return ""
    # 泰文句尾偶尔会跟一个中文句号（实测 gpt-4o-mini 出过：
    # "…ในการประเมินครับ。"）。泰语本来就不用句号，混进来的中文标点
    # 是模型把源语言的标点顺手带过来了，投屏时很显眼。
    if re.search(r"[฀-๿]", out):        # 确实是泰文才清
        out = re.sub(r"[。．｡]+\s*$", "", out).strip()
    return out


class Translator:
    def __init__(self, cfg: dict):
        self.model = cfg.get("model", "claude-sonnet-5")
        self.glossary = cfg.get("glossary") or {}
        self.domain = cfg.get("domain_hint") or "线上商务会议"
        self.context_turns = int(cfg.get("context_turns", 4))
        self.speaker = SPEAKER.get(str(cfg.get("speaker_gender", "")).lower(),
                                   SPEAKER[""])
        # 短句缓存：会议里"好的""能听到吗"这类话会重复很多遍。
        # 当前这代模型的 API 已经不支持 temperature，同样输入每次输出都可能不同，
        # 缓存能保证重复的话译法一致，顺带省钱省时间。
        self.cache_max_chars = int(cfg.get("cache_max_chars", 28))
        self._cache: dict[tuple, str] = {}
        self._cache_order: list[tuple] = []
        # 正在翻的句子，防止同一句话同时打两次 API（见 translate_stream）
        self._inflight: dict[tuple, asyncio.Future] = {}

        from anthropic import AsyncAnthropic

        key = cfg.get("anthropic_api_key") or ""
        if not key:
            raise RuntimeError(
                "没有找到 Anthropic API Key。\n"
                "请在 config.json 里填 anthropic_api_key，"
                "或设置环境变量 ANTHROPIC_API_KEY。"
            )
        # 开会场景宁可漏一句也不能卡住：撞到限流时 SDK 默认会等满一整个
        # 限流窗口（可能 60 秒）再重试，字幕就死在那儿了。只重试 1 次、10 秒超时。
        # 10 秒太紧，实测两小时会议里有 2 条长句因此丢失（ReadTimeout）。
        # 放到 20 秒：正常句子 1~3 秒就回来了，这个上限只对付偶发的长句。
        kwargs = {"api_key": key, "max_retries": 1,
                  "timeout": float(cfg.get("translate_timeout", 20.0))}
        if cfg.get("anthropic_base_url"):
            kwargs["base_url"] = cfg["anthropic_base_url"]
        self.client = AsyncAnthropic(**kwargs)

        # anthropic SDK 1.x 的 messages.create() 去掉了 temperature，
        # 老版本还有。按实际签名决定传不传，两边都能跑。
        try:
            import inspect

            params = inspect.signature(type(self.client.messages).create).parameters
            self._supports_temperature = "temperature" in params
        except Exception:
            self._supports_temperature = False

        # 只有默认会思考的模型才需要显式关掉；Haiku 本来就不思考。
        # Opus 5 不要关思考（官方文档说关了会有副作用），改用 effort 控制深度。
        m = self.model.lower()
        want = bool(cfg.get("disable_thinking", True))
        # Opus 5 官方明确建议不要关思考（关了有副作用），用 effort 控制深度；
        # Haiku 本来就不思考，传这个参数反而可能报错。只有 Sonnet 需要显式关。
        self._no_thinking = want and "opus" not in m and "haiku" not in m
        self.effort = cfg.get("effort") or ("low" if "opus" in m else None)
        self.cache_prompt = bool(cfg.get("prompt_cache", True))

    # ------------------------------------------------------------------
    def _build(self, text: str, src: str, tgt: str,
               context: list[dict] | None, fragment: bool = False) -> dict:
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
            # 连原文带译文一起给。汇报型长发言会被切成很多段，
            # 只给原文的话，后面几段的用词和语气会跟前面对不上。
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
            # 说话太长被中途切开的半句。当成完整句子翻会硬加上不存在的结尾。
            parts.append(
                "注意：说话人还没说完，下面只是他这段话的**前半截**。"
                "照着现有内容翻就行，不要替他补完整、不要加句号收尾、"
                "不要加「ครับ/ค่ะ」这类收尾敬语。"
            )
        parts.append(f"<需要翻译的这一句>\n{text}\n</需要翻译的这一句>")

        kwargs = {
            "model": self.model,
            # 泰文分词很占 token，一段长发言的译文可能上千。给足余量，
            # 免得译到一半被截断（实测见过一次）。
            "max_tokens": 2000,
            "system": system,
            "messages": [{"role": "user", "content": "\n\n".join(parts)}],
        }

        # 系统提示词（规则 + 术语表）每次调用都一模一样，缓存住省掉重复处理。
        # 实测输入 token 1674 → 54，成本降 45%。
        # 注意有最小长度要求：提示词太短时缓存不生效，术语表填得越多越划算。
        if self.cache_prompt:
            kwargs["system"] = [{
                "type": "text", "text": system,
                "cache_control": {"type": "ephemeral"},
            }]

        if self._supports_temperature:
            kwargs["temperature"] = 0.0        # 翻译要稳定，不要发挥

        # 关掉思考。Sonnet 5 / Opus 5 默认开自适应思考，但"把一句话翻成另一种语言"
        # 用不上推理——思考 token 白白拖慢首字时间（实测 1593ms → 1308ms）。
        # Haiku 4.5 本来就不思考，不用传。
        if self._no_thinking:
            kwargs["thinking"] = {"type": "disabled"}
        # Opus 5 用 effort 控制深度而不是关思考。翻译一句话属于"简单任务"，
        # low 就够，实测质量不降反而更快。
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        return kwargs

    async def translate(self, text: str, src: str, tgt: str,
                        context: list[dict] | None = None) -> str:
        resp = await self.client.messages.create(**self._build(text, src, tgt, context))
        out = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return _final(out)

    # ------------------------------------------------------------------
    def _cache_key(self, text: str, src: str, tgt: str):
        """短句才进缓存。长句基本不会一字不差地重复，而且更依赖上下文。"""
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
        """边翻边推：每收到一小段就回调一次，字幕像打字一样长出来。

        对方看到的不是"空白等 5 秒然后整句蹦出来"，而是立刻开始有字，
        感知上的等待时间短得多。
        """
        key = None if fragment else self._cache_key(text, src, tgt)
        hit = self._cache.get(key) if key else None
        if hit:
            await on_delta(hit)      # 命中缓存就一次推完，几毫秒的事
            return hit

        # 同一句话已经在翻了：等它的结果，别再打一次 API。
        # 缓存只在译完之后才写得进去，所以**同一句话同时在飞**的时候
        # 两边会各打一次，还可能译出两个版本。实测遇到过：
        # 本场第一句冷启动要 4.5 秒，第二句 3 秒后就到了，
        # 同样一句"喂，可以听到我说话吗？"译出了两种说法。
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
            async with self.client.messages.stream(
                    **self._build(text, src, tgt, context, fragment)) as stream:
                async for chunk in stream.text_stream:
                    if not chunk:
                        continue
                    buf.append(chunk)
                    await on_delta(chunk)

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
