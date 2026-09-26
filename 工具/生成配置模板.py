"""从代码里的 DEFAULTS 生成 config.example.json。

    .venv\Scripts\python.exe 工具\生成配置模板.py

为什么要有这个脚本：默认值原本有两份——代码里的 DEFAULTS 和手写的
config.example.json。两份必然会漂移（实测漂了 16 项，包括识别模型名
和语种偏向这种会影响行为的），而且看的人不知道该信哪一份。

现在只有一份真相：代码。example 由它生成，密钥留空、
术语表和纠错表换成通用示例。
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from translator.config import DEFAULTS

cfg = copy.deepcopy(DEFAULTS)

# 密钥一律留空（DEFAULTS 里本来就是空的，这里再确认一遍，防止以后有人填进去）
for path in (("anthropic_api_key",), ("anthropic_base_url",),
             ("openai_api_key",), ("openai_base_url",),
             ("asr", "tencent_secret_id"), ("asr", "tencent_secret_key"),
             ("asr", "tencent_appid"), ("asr", "openai_api_key")):
    d = cfg
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = ""

# 术语表 / 纠错表：给通用示例，不带任何真实公司信息
cfg["glossary"] = {
    "验厂": "ตรวจโรงงาน",
    "验货": "ตรวจสอบสินค้า",
    "客诉": "ข้อร้องเรียนจากลูกค้า",
    "产能": "กำลังการผลิต",
    "报价单": "ใบเสนอราคา",
    "王经理": "ผู้จัดการหวัง",
}
cfg["asr_corrections"] = {
    "厌厂": "验厂",
    "验场": "验厂",
    # 把一个词映射到它自己 = 保护它，不让其它规则改到。
    # 比如有了「烟厂→验厂」之后，「香烟厂」就需要这么保护一下。
    "香烟厂": "香烟厂",
}
cfg["domain_hint"] = "线上商务会议，议题包括生产、品质、交期"
# 语种和后端按最保守的默认来，每台机器自己判断
cfg["language_lock"] = {"me": None, "them": None}
cfg["language_bias"] = {"me": None, "them": None}

out = ROOT / "config.example.json"
body = json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
out.write_text(body, encoding="utf-8")

# ---- 自检 ----
bad = []
chk = json.loads(body)
for k, v in (("anthropic_api_key", chk.get("anthropic_api_key")),
             ("openai_api_key", chk.get("openai_api_key")),
             ("asr.tencent_secret_id", chk["asr"].get("tencent_secret_id")),
             ("asr.tencent_secret_key", chk["asr"].get("tencent_secret_key")),
             ("asr.tencent_appid", chk["asr"].get("tencent_appid"))):
    if v != "":
        bad.append(f"{k} 不是空的：{v!r}")
for pat in (r"sk-ant-[A-Za-z0-9\-_]{20,}", r"sk-proj-[A-Za-z0-9\-_]{20,}",
            r"\bAKID[A-Za-z0-9]{20,}"):
    if re.search(pat, body):
        bad.append(f"出现了密钥形态的字符串：{pat}")

# 和代码默认值逐项比对，除了上面刻意改的那几项，其余必须一致
def flat(d, p=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(flat(v, p + k + "."))
        else:
            o[p + k] = v
    return o

EXPECT_DIFF = ("glossary.", "asr_corrections.", "domain_hint",
               "language_lock.", "language_bias.")
a, b = flat(DEFAULTS), flat(chk)
drift = [k for k in sorted(set(a) | set(b))
         if a.get(k) != b.get(k) and not k.startswith(EXPECT_DIFF)]

print(f"已生成 {out.name}　{len(body)} 字节，{len(chk)} 个顶层配置项\n")
print(f"  {'✓' if not bad else '✗'} 没有密钥")
print(f"  {'✓' if not drift else '✗'} 和代码默认值一致"
      + (f"（漂移：{drift}）" if drift else "（刻意不同的只有术语表/纠错表/语种）"))
if bad or drift:
    for x in bad + drift:
        print(f"     · {x}")
    sys.exit(1)
print("\n✓ 检查通过")
