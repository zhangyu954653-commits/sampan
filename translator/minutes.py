"""自动生成中泰双语会议纪要。

输入是一场会的字幕记录（内存里的 items 或硬盘上的 .jsonl），
输出一份 Markdown + 一份 Word。

两件事值得说明：

1. **长会要分段。** 两小时的会有 800 多句、五六万字，一次喂进去既慢又容易
   丢细节。超过阈值就按时间切块，每块先提要点，最后再合成一份完整纪要。

2. **待办事项宁可空着也不要编。** 纪要是拿去分派工作的，编一条不存在的
   负责人或日期，比漏掉一条危害大得多。这一条写进提示里反复强调。
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

CHUNK_CHARS = 26000        # 单块最多这么多字，超了就分段提要
LANG = {"zh": "中文", "th": "泰语"}

SYSTEM = """你是一位中泰双语的会议记录员，要把一场线上会议的实时字幕整理成正式的会议纪要。

背景：{domain}
字幕来自语音识别，会有错字、断句不全、口语重复。整理时要理解意图、还原成书面表达。

**铁律：**
1. 只写字幕里真实出现过的内容。**绝不允许编造**人名、日期、数字、金额、负责人。
   某个细节没说清楚，就写"待确认"，不要替他们填一个看起来合理的值。
2. 识别错的词按上下文还原成业务上说得通的词（例如"厌厂/严厂"→"验厂"，
   "指托班"→"纸托盘"）。拿不准的词保留原样并标注"（原文如此）"。
3. 中泰双语对照：每个要点先中文、紧跟泰语，泰语要用口语化的商务表达，
   不要用书面语的僵硬句式。
4. 数字、日期、金额原样保留，不要换算、不要四舍五入。
5. 泰语句尾敬语统一用 ครับ，全程不要变来变去。

**输出格式**（严格用这个结构，纯 Markdown，不要加代码块围栏）：

# 会议纪要 / บันทึกการประชุม

## 摘要 / สรุปย่อ

**这一节必须能单独看懂，控制在一页以内。** 看的人多半只读这一节，
后面的逐条细节是给需要追溯的人看的。这里只写"要拿去执行的东西"，
不要复述讨论过程。

### 决议 / ข้อตกลง
最多 5 条，每条一行，中泰对照。只写**真的拍板了**的事，
还在讨论、没有结论的不要放进来。

### 待办 / สิ่งที่ต้องทำ
一个 Markdown 表格，表头固定为：
| # | 事项 | รายการ | 负责方 | 时限 |
只放这一节，后面不要重复。负责方或时限原文没说，就写"待确认 / รอยืนยัน"——
**绝不允许替他们指派人或编一个日期**。

### 待确认 / ต้องยืนยัน
最多 5 条，只列**会卡住执行**的问题（数字对不上、责任人没定、
文件到底有没有）。纯粹说得含糊但不影响做事的，不要列。
没有就写"无 / ไม่มี"。

---

## 一、会议信息 / ข้อมูลการประชุม
用 `- **项目**：值` 的形式，列出时间、时长、参与方、主题。

## 二、议题与结论 / หัวข้อและข้อสรุป
按议题分小节（### 1. 标题 / หัวข้อภาษาไทย），每节内：
- **讨论**：中文……
- **ภาษาไทย**：泰语……
- **结论 / ข้อสรุป**：中文…… ／ 泰语……

## 三、其它待确认与遗留 / ประเด็นค้างอื่น ๆ
摘要那一节已经放了决议、待办和关键待确认，**这里不要重复**。
只写那些值得留档、但还排不上"卡住执行"的遗留点。没有就写"无 / ไม่มี"。
"""

CHUNK_SYSTEM = """你在为一场长会议做分段提要，这是第 {i}/{n} 段字幕。

不要写成纪要，只要把这一段里**真实出现过**的内容提炼成条目：
- 讨论了哪些议题、各自的结论
- 谁要做什么、什么时候之前做完（原文没说就写"未提及"）
- 提到的具体数字、日期、金额、文件名、条款编号，原样抄下来
- 没说清楚的地方，单独列出来

绝不允许编造。语音识别的错字按上下文还原，拿不准就保留原样。
只用中文写，简洁的条目式，不要客套话。
"""


def load_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
    return items


def latest_transcript(dir_path: str | Path) -> Path | None:
    """挑最近一场会的记录。文件名形如 2026-09-16_1003.jsonl。"""
    d = Path(dir_path)
    if not d.is_dir():
        return None
    files = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def digest(items: list[dict]) -> str:
    """把字幕压成给模型看的稿子：时间、谁说的、说了什么。

    不带译文——纪要本来就要重新组织语言，把译文也塞进去只是白烧 token。
    """
    lines = []
    for it in items:
        text = (it.get("text") or "").strip()
        if not text:
            continue
        t = datetime.fromtimestamp(it.get("ts", 0)).strftime("%H:%M")
        who = it.get("speaker") or ("我方" if it.get("channel") == "me" else "对方")
        lang = LANG.get(it.get("src_lang"), "")
        tag = f"（{lang}）" if lang == "泰语" else ""
        lines.append(f"[{t}] {who}{tag}：{text}")
    return "\n".join(lines)


def meta_of(items: list[dict]) -> dict:
    ts = [i.get("ts", 0) for i in items if i.get("ts")]
    start = datetime.fromtimestamp(min(ts)) if ts else datetime.now()
    end = datetime.fromtimestamp(max(ts)) if ts else start
    return {
        "start": start,
        "end": end,
        "minutes": max(1, round((end - start).total_seconds() / 60)),
        "count": len(items),
        "me": sum(1 for i in items if i.get("channel") == "me"),
        "them": sum(1 for i in items if i.get("channel") == "them"),
        "th": sum(1 for i in items if i.get("src_lang") == "th"),
    }


def _split(text: str, limit: int) -> list[str]:
    """按行切块，尽量不把一句话切开，而且每块尽量一样大。

    不能简单地"装满一块再开下一块"：3.3 万字按 2.6 万切会得到
    26000 + 7000，后面那块太薄，提要出来的东西明显比前面粗。
    先算出需要几块，再按块数均分。
    """
    if len(text) <= limit:
        return [text]
    lines = text.split("\n")
    n_chunks = -(-len(text) // limit)          # 向上取整
    target = -(-len(text) // n_chunks)
    out, cur, n = [], [], 0
    for line in lines:
        if n + len(line) > target and cur and len(out) < n_chunks - 1:
            out.append("\n".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += len(line) + 1
    if cur:
        out.append("\n".join(cur))
    return out


class MinutesMaker:
    def __init__(self, cfg: dict):
        from anthropic import AsyncAnthropic

        key = cfg.get("anthropic_api_key") or ""
        if not key:
            raise RuntimeError("没有配置 Anthropic API Key，无法生成纪要。")
        kwargs = {"api_key": key, "max_retries": 2,
                  "timeout": float(cfg.get("minutes_timeout", 600.0))}
        if cfg.get("anthropic_base_url"):
            kwargs["base_url"] = cfg["anthropic_base_url"]
        self.client = AsyncAnthropic(**kwargs)
        self.model = cfg.get("minutes_model") or cfg.get("model", "claude-opus-5")
        self.domain = cfg.get("domain_hint") or "线上商务会议"
        self.glossary = cfg.get("glossary") or {}
        # 纪要是一次性长任务，不像字幕那样要抢毫秒，可以让它想得久一点
        self.effort = cfg.get("minutes_effort") or "medium"
        # 泰文非常耗 token：实测 1.7 万字的双语纪要里泰文占 1.2 万字，
        # 16000 根本写不完，会在"待办事项"之前就被截断。
        self.max_tokens = int(cfg.get("minutes_max_tokens", 32000))

    async def _once(self, system: str, messages: list, max_tokens: int,
                    on_progress=None):
        """必须走流式。

        一次要吐三万个 token 的中泰双语长文，非流式请求会直接超时——
        不是把 timeout 调大就能解决的，官方文档对长请求明确要求用流式。
        （实测：非流式 + max_tokens=32000 稳定 APITimeoutError。）
        顺带还能一边生成一边往界面报进度，不用干等几分钟。
        """
        kwargs = dict(model=self.model, max_tokens=max_tokens,
                      system=system, messages=messages)
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}

        async def _run(kw):
            buf = []
            last = 0
            async with self.client.messages.stream(**kw) as stream:
                async for piece in stream.text_stream:
                    buf.append(piece)
                    n = sum(len(x) for x in buf)
                    if on_progress and n - last > 1200:
                        last = n
                        on_progress(n)
                final = await stream.get_final_message()
            return "".join(buf), getattr(final, "stop_reason", "")

        try:
            return await _run(kwargs)
        except TypeError:
            kwargs.pop("output_config", None)
            return await _run(kwargs)

    async def _ask(self, system: str, user: str, max_tokens: int = 8000,
                   rounds: int = 3, on_progress=None) -> str:
        """问一次；写到一半被 max_tokens 截断就接着让它写完。

        泰文特别耗 token——一份 1.7 万字的双语纪要里泰文占 1.2 万字，
        一口气写不完是常态。截断的后果很隐蔽：文件看着像模像样，
        其实"待办事项"整段没了，而那恰恰是最要紧的一节。
        """
        msgs = [{"role": "user", "content": user}]
        out = ""
        for i in range(rounds):
            base = len(out)
            text, stop = await self._once(
                system, msgs, max_tokens,
                (lambda n, b=base: on_progress(b + n)) if on_progress else None)
            out += text
            if stop != "max_tokens":
                break
            if i == rounds - 1:
                print("[纪要] 内容太长，续写次数已用尽，结果可能不完整")
                break
            print(f"[纪要] 一次没写完，继续写（第 {i + 2} 段）…")
            msgs = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": out.rstrip()},
                {"role": "user", "content":
                    "接着上面断掉的地方继续写完，不要重复已经写过的内容，"
                    "也不要重新开头。"},
            ]
        return out.strip()

    def _glossary_note(self) -> str:
        if not self.glossary:
            return ""
        pairs = "、".join(f"{k}→{v}" for k, v in list(self.glossary.items())[:40])
        return f"\n\n公司固定译法（泰语部分务必沿用）：{pairs}"

    async def make(self, items: list[dict], progress=None) -> str:
        """生成纪要 Markdown。progress 是个可选回调，用来往界面报进度。"""
        def say(msg):
            print(f"[纪要] {msg}")
            if progress:
                try:
                    progress(msg)
                except Exception:
                    pass

        if not items:
            raise RuntimeError("这场会还没有字幕，没什么可整理的。")

        body = digest(items)
        m = meta_of(items)
        head = (f"会议时间：{m['start']:%Y-%m-%d %H:%M} — {m['end']:%H:%M}"
                f"（约 {m['minutes']} 分钟）\n"
                f"字幕 {m['count']} 句：我方 {m['me']} 句，对方 {m['them']} 句，"
                f"其中泰语 {m['th']} 句\n")

        chunks = _split(body, CHUNK_CHARS)
        if len(chunks) > 1:
            say(f"字幕较长（{len(body)} 字），分 {len(chunks)} 段先提要")
            notes = []
            for i, ch in enumerate(chunks, 1):
                say(f"正在提炼第 {i}/{len(chunks)} 段…")
                notes.append(await self._ask(
                    CHUNK_SYSTEM.format(i=i, n=len(chunks)), ch, max_tokens=4000))
            body = "\n\n".join(f"【第 {i} 段要点】\n{t}"
                               for i, t in enumerate(notes, 1))
            say("各段要点已就绪，正在合成完整纪要…")
        else:
            say(f"正在整理（{m['count']} 句，约 {m['minutes']} 分钟）…")

        system = SYSTEM.format(domain=self.domain) + self._glossary_note()
        md = await self._ask(system, head + "\n以下是会议字幕：\n\n" + body,
                             max_tokens=int(self.max_tokens),
                             on_progress=lambda n: say(f"正在写…已生成 {n} 字"))
        md = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", md.strip())

        # 结构自查：缺了哪一节要说出来，别让人拿到一份看着完整、
        # 其实"待办事项"整段没了的文件。
        missing = [name for name, key in
                   (("会议信息", "会议信息"), ("议题与结论", "议题与结论"),
                    ("待办事项", "待办事项"), ("需要确认的事项", "需要确认"))
                   if key not in md]
        if missing:
            say("⚠ 这几节没生成出来：" + "、".join(missing))
        elif "|" not in md:
            say("⚠ 待办事项没做成表格")
        say("完成")
        return md


# ======================================================================
# 存盘：Markdown + Word
# ======================================================================
ZH_FONT = "微软雅黑"
TH_FONT = "Leelawadee UI"          # Windows 自带的泰文字体
_THAI = re.compile(r"[฀-๿]")


def _set_fonts(run, size: float, bold: bool = False, color=None):
    """中泰混排的关键：泰文属于"复杂文种"，必须单独指定 w:cs 字体和字号，
    否则 Word 会用默认字体渲染，泰文全变成方框。"""
    from docx.oxml.ns import qn
    from docx.shared import Pt

    run.font.size = Pt(size)
    run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    f = rpr.get_or_add_rFonts()
    f.set(qn("w:ascii"), ZH_FONT)
    f.set(qn("w:hAnsi"), ZH_FONT)
    f.set(qn("w:eastAsia"), ZH_FONT)
    f.set(qn("w:cs"), TH_FONT)
    for tag, val in (("w:szCs", str(int(size * 2))),):
        rpr.append(rpr.makeelement(qn(tag), {qn("w:val"): val}))
    if bold:
        from docx.oxml.ns import qn as _q
        rpr.append(rpr.makeelement(_q("w:bCs"), {}))


_BOLD = re.compile(r"\*\*(.+?)\*\*")


def _add_rich(p, text: str, size: float, color=None):
    """处理 **加粗**，其余原样。"""
    pos = 0
    for m in _BOLD.finditer(text):
        if m.start() > pos:
            _set_fonts(p.add_run(text[pos:m.start()]), size, False, color)
        _set_fonts(p.add_run(m.group(1)), size, True, color)
        pos = m.end()
    if pos < len(text):
        _set_fonts(p.add_run(text[pos:]), size, False, color)
    if not text:
        _set_fonts(p.add_run(""), size, False, color)


def write_docx(md: str, path: str | Path):
    """把纪要的 Markdown 渲染成 Word。只认纪要用到的那几种结构。"""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    BLUE = RGBColor(0x1F, 0x4E, 0x79)
    GRAY = RGBColor(0x5A, 0x63, 0x73)

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    sec.left_margin = sec.right_margin = Cm(2.2)

    normal = doc.styles["Normal"]
    normal.font.name = ZH_FONT
    normal.font.size = Pt(10.5)
    rf = normal.element.get_or_add_rPr().get_or_add_rFonts()
    for tag in ("w:ascii", "w:hAnsi", "w:eastAsia"):
        rf.set(qn(tag), ZH_FONT)
    rf.set(qn("w:cs"), TH_FONT)

    def para(space_after=4, indent=None, align=None):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.line_spacing = 1.3
        if indent:
            p.paragraph_format.left_indent = Cm(indent)
        if align:
            p.alignment = align
        return p

    lines = md.split("\n")
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        i += 1

        if not line:
            continue

        # ---- 表格 ----
        if line.startswith("|") and line.endswith("|"):
            rows = []
            j = i - 1
            while j < len(lines) and lines[j].strip().startswith("|"):
                cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                j += 1
            i = j
            if not rows:
                continue
            width = max(len(r) for r in rows)
            table = doc.add_table(rows=0, cols=width)
            table.style = "Table Grid"
            for ri, r in enumerate(rows):
                cells = table.add_row().cells
                for ci in range(width):
                    txt = r[ci] if ci < len(r) else ""
                    cell = cells[ci]
                    cell.text = ""
                    p = cell.paragraphs[0]
                    p.paragraph_format.space_after = Pt(2)
                    _add_rich(p, txt, 9.5, BLUE if ri == 0 else None)
                    if ri == 0:
                        for run in p.runs:
                            run.font.bold = True
            para(space_after=8)
            continue

        # ---- 标题 ----
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level, text = len(m.group(1)), m.group(2)
            size = {1: 17, 2: 13.5, 3: 11.5, 4: 11}[level]
            p = para(space_after=6 if level > 1 else 10,
                     align=WD_ALIGN_PARAGRAPH.CENTER if level == 1 else None)
            p.paragraph_format.space_before = Pt(0 if level == 1 else 10)
            _add_rich(p, text, size, BLUE)
            for run in p.runs:
                run.font.bold = True
            continue

        # ---- 分隔线 ----
        if re.fullmatch(r"[-*_]{3,}", line):
            continue

        # ---- 列表 ----
        m = re.match(r"^([-*+]|\d+\.)\s+(.*)$", line)
        if m:
            indent = (len(raw) - len(raw.lstrip())) // 2
            p = para(space_after=3, indent=0.6 + indent * 0.5)
            bullet = "• " if not m.group(1)[0].isdigit() else m.group(1) + " "
            _set_fonts(p.add_run(bullet), 10.5, False, GRAY)
            _add_rich(p, m.group(2), 10.5)
            continue

        # ---- 引用 ----
        if line.startswith(">"):
            p = para(space_after=4, indent=0.8)
            _add_rich(p, line.lstrip("> ").strip(), 10, GRAY)
            continue

        # ---- 普通段落 ----
        p = para(space_after=5)
        _add_rich(p, line, 10.5)

    doc.save(str(path))
    return path


def save(md: str, out_dir: str | Path, stem: str) -> dict:
    """写 .md，能写 Word 就再写一份 .docx。返回生成的文件路径。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"{stem}_纪要.md"
    md_path.write_text(md, encoding="utf-8")
    result = {"md": str(md_path)}
    try:
        docx_path = out / f"{stem}_纪要.docx"
        write_docx(md, docx_path)
        result["docx"] = str(docx_path)
    except ImportError:
        print("[纪要] 没装 python-docx，只生成了 Markdown。"
              "想要 Word 版就运行：pip install python-docx")
    except Exception as e:
        print(f"[纪要] Word 生成失败（Markdown 已保存）：{e}")
    return result
