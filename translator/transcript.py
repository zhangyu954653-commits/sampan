"""把每场会议的字幕存到硬盘。

之前字幕只活在内存和浏览器页面里，程序一关就没了——开完会想回看纪要
或者核对某句话说了什么，完全没辙。

每场会议写两个文件：
  .md    给人看的，带时间、说话人、原文、译文
  .jsonl 给程序读的，一行一句，方便以后做检索或统计

手动改过的句子会重写对应那一行，所以文件里始终是修正后的版本。
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

LANG = {"zh": "中文", "th": "ไทย"}


class Transcript:
    def __init__(self, cfg: dict, root: Path):
        # 多句话会并发翻译完成，add() 是从不同线程进来的，
        # 没有锁的话 items 会错乱、文件也可能被写坏。
        self._lock = threading.Lock()
        self.enabled = bool(cfg.get("save_transcript", True))
        self.dir = root / str(cfg.get("transcript_dir", "会议记录"))
        self.items: list[dict] = []
        self._dirty = False
        self._last_save = 0.0
        self._min_interval = 3.0        # 别每句都重写文件
        self.started = datetime.now()
        self.md: Path | None = None
        self.jsonl: Path | None = None

        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            stem = self.started.strftime("%Y-%m-%d_%H%M")
            self.md = self.dir / f"{stem}.md"
            self.jsonl = self.dir / f"{stem}.jsonl"

    # ------------------------------------------------------------------
    def add(self, item: dict):
        """一句话翻译完成后调用。同一个 id 再次调用则覆盖（手动改过的情况）。"""
        if not self.enabled:
            return
        with self._lock:
            for i, old in enumerate(self.items):
                if old["id"] == item["id"]:
                    self.items[i] = dict(item)
                    break
            else:
                self.items.append(dict(item))
            self._dirty = True
        self.flush()

    def remove(self, item_id: int):
        """事后认出是回声的那条，从记录里删掉。

        会议纪要是拿这个文件生成的，留着回声会让同一句话在纪要里出现两遍，
        还会把说话人标错——"我方"说了对方的话。
        """
        if not self.enabled:
            return
        with self._lock:
            before = len(self.items)
            self.items = [i for i in self.items if i.get("id") != item_id]
            if len(self.items) == before:
                return
            self._dirty = True
        self.flush(force=True)

    def flush(self, force: bool = False):
        if not self.enabled:
            return
        with self._lock:
            if not self._dirty:
                return
            now = time.time()
            if not force and now - self._last_save < self._min_interval:
                return
            self._last_save = now
            self._dirty = False
            try:
                self._write()
            except Exception as e:
                print(f"[记录] 写入失败：{e}")

    # ------------------------------------------------------------------
    def _write(self):
        with open(self.jsonl, "w", encoding="utf-8") as f:
            for it in self.items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

        lines = [
            f"# 会议记录 {self.started.strftime('%Y-%m-%d %H:%M')}",
            "",
            f"共 {len(self.items)} 句　"
            f"我方 {sum(1 for i in self.items if i['channel'] == 'me')} 句　"
            f"对方 {sum(1 for i in self.items if i['channel'] == 'them')} 句",
            "",
        ]
        for it in self.items:
            t = datetime.fromtimestamp(it["ts"]).strftime("%H:%M:%S")
            src = LANG.get(it.get("src_lang"), "")
            tgt = LANG.get(it.get("tgt_lang"), "")
            mark = "　✏️已手动修正" if it.get("edited") else ""
            lines.append(f"### {t}　**{it['speaker']}**　{src} → {tgt}{mark}")
            lines.append("")
            lines.append(f"> {it['text']}")
            lines.append("")
            if it.get("translation"):
                lines.append(it["translation"])
                lines.append("")

        with open(self.md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ------------------------------------------------------------------
    def close(self) -> str | None:
        """会议结束时调用，返回记录文件路径。"""
        if not self.enabled or not self.items:
            return None
        self.flush(force=True)
        return str(self.md)
