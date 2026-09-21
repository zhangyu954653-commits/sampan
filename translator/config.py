"""配置读写：首次运行自动生成 config.json。"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

DEFAULTS: dict = {
    # ---------- 翻译（Claude） ----------
    "anthropic_api_key": "",          # 留空则读环境变量 ANTHROPIC_API_KEY
    "anthropic_base_url": "",         # 走代理/中转时填，例如 https://xxx/v1；留空用官方
    # 翻译后端："anthropic"（默认）或 "openai"。想 A/B 对比时改这一个值，
    # 提示词和术语表两边完全一样，比的是模型本身。
    "translate_backend": "anthropic",
    "openai_api_key": "",             # 留空则读环境变量 OPENAI_API_KEY
    "openai_base_url": "",            # 用代理/中转时填，否则留空
    "openai_translate_model": "gpt-4o-mini",
    "model": "claude-sonnet-5",       # 想更快更便宜：claude-haiku-4-5
    "context_turns": 4,               # 翻译时带上前几句作为上下文（提高代词/省略准确度）
    # 只对默认会思考的模型有意义。Opus 5 官方建议不要关思考（关了有副作用），
    # 代码里已经对 Opus / Haiku 自动忽略这一项，只有 Sonnet 会真的关。
    "disable_thinking": True,

    # 你的性别。泰语的第一人称和句尾敬语分男女，不填的话每句可能变来变去。
    # "male" = ผม/ครับ    "female" = ดิฉัน/ค่ะ    "" = 统一用中性说法
    "speaker_gender": "",

    # 短句缓存：会议里"好的""能听到吗"这类话会重复很多遍。
    # 当前这代模型的 API 已经不支持 temperature，同样输入每次输出都可能不同，
    # 缓存能保证重复的话译法一致，顺带省钱省时间。多少字以内的句子进缓存：
    "cache_max_chars": 28,

    # ---------- 语音识别 ----------
    "asr": {
        # tencent = 腾讯云一句话识别（推荐：约 1 秒出结果，不占 CPU，不会积压丢句）
        # local   = 本机 faster-whisper（免费离线，但本机一句要 5~8 秒）
        # openai  = 云端 Whisper
        "backend": "tencent",

        # 腾讯云密钥，在 https://console.cloud.tencent.com/cam/capi 获取。
        # 语种判定仍由本地 tiny 模型完成（腾讯云的引擎按语种分开，
        # 没有能同时自动认中文和泰语的引擎）。
        "tencent_secret_id": "",
        "tencent_secret_key": "",
        "tencent_engines": {"zh": "16k_zh", "th": "16k_th"},

        "model_size": "auto",         # auto / tiny / base / small / medium / large-v3
        "device": "auto",             # auto / cuda / cpu
        "compute_type": "auto",       # auto / int8 / int8_float16 / float16 / float32
        "beam_size": 1,
        "cpu_threads": 0,             # 0 = 按核数自动（实测 8 线程已吃满，再多没用）
        # 专职判语种的小模型。实测 small 自己判语种要 7.0s，
        # 而 tiny 判(0.5s) + small 按结果转写(3.9s) = 4.4s，准确率一样。
        # 填 "" 则不用小模型，由主模型自己判（更慢）。
        "lang_detect_model": "tiny",
        "language_bias_strength": 2.5,  # 声道先验权重，1 = 不偏向

        # 边说边显示：说话过程中用 tiny 快速跑出中间结果（实测 0.8 秒一次），
        # 让人立刻看到"系统在听"，而不是干等 5 秒。会有错字，说完会被正式结果替换。
        # 只用空闲算力跑——正式识别一忙就让路，绝不拖慢真正要用的那份。
        "partial_results": True,
        "partial_model": "tiny",
        "partial_min_sec": 1.0,       # 说满这么久才开始显示
        "partial_interval_sec": 1.1,  # 最快多久刷新一次
        # 语种纠错：只在转写明显失败时才用另一种语言重来一次（每次多花约 3.5 秒）
        "verify_language": True,
        "verify_logprob": -0.95,      # 转写置信度低于此值才重试
        "verify_margin": 0.15,        # 另一种语言要高出这么多才改判
        "openai_api_key": "",         # 留空则自动用顶层的 openai_api_key
        # whisper-1 是老模型，慢。gpt-4o-mini-transcribe 明显更快，
        # gpt-4o-transcribe 更准但贵一些。注意这两个新模型不支持
        # verbose_json，代码里已经按模型名自动切换返回格式。
        "openai_model": "gpt-4o-mini-transcribe",
        # 喂给识别模型的提示：专业词和人名写进来，识别准确率会明显提高。
        # 留空则自动用术语表里的词拼一份。
        "openai_prompt": "",
    },

    # 每一路"更可能说什么语言"。只是加权，不是锁定——
    # 对方偶尔说中文照样能识别出来。不想偏向就填 null。
    "language_bias": {
        "me": "zh",                   # 麦克风（你）
        "them": "th",                 # 系统声音（对方）
    },

    # 直接锁死某一路的语言，完全跳过语种判定。
    # 短音频上的语种判定本来就不可靠——实测泰语短语的 p(zh)/p(th) 都在
    # 0.01 量级，等于在噪声里比大小，所以"ครับ"这种短词有概率被判成中文，
    # 转写出一串汉字音译。如果对方全程只说泰语，锁定是 100% 可靠的解法。
    # 界面顶栏点「我方/对方」旁边的语言标就能切，改动即时生效，无需重启。
    # null = 自动判定
    "language_lock": {
        "me": None,
        "them": None,
    },

    # ---------- 音频采集 ----------
    "audio": {
        "enable_mic": True,           # 采集麦克风 = 你自己说的话
        "enable_loopback": True,      # 采集系统声音 = 钉钉里对方的声音
        "mic_device_index": None,     # None = 系统默认输入设备；用 --list-devices 查序号
        "loopback_device_index": None,  # None = 系统默认扬声器的环回设备
        "mic_gain": 1.0,
        "loopback_gain": 1.0,
    },

    # ---------- 断句（VAD） ----------
    # 注意 end_frames 不是越小越好。Whisper 内部固定按 30 秒窗口算，
    # 2 秒的音频和 8 秒的识别耗时几乎一样，所以句子切得越碎总开销越大、
    # 排队越久。宁可多等 0.2 秒让短句合并，整体反而更快。
    # end_frames 是"停多久才算说完"。定短了，人说话中途正常的换气停顿
    # 就会把一句话切成好几段，前后文全断；定长了，说完要多等一会儿才出字幕。
    # 1.15 秒是权衡后的值——正常换气停顿基本都在 1 秒以内。
    "vad": {
        "aggressiveness": 3,          # 0~3，数字越大越严格（安静环境可降到 2）
        "start_frames": 4,            # 连续多少帧有人声才算"开始说话"（1 帧 = 30ms）
        "end_frames": 38,             # 连续多少帧静音才算"说完了" 38*30ms ≈ 1.15s
        "preroll_ms": 320,            # 往前多带一点音频，避免吞掉开头
        "max_seconds": 15.0,          # 一口气说太久时强制切开（切在停顿处）
        "min_seconds": 0.5,           # 有人声的时长不足这么久，当噪音丢掉
        "rms_threshold": 0.012,       # 能量门限，过滤键盘声/底噪/空调声
    },

    # 识别排队超过这么久的句子直接丢掉。实时字幕里翻一句一分钟前的话没有意义，
    # 而且只会让后面每一句都更迟。
    "max_lag_sec": 6.0,

    # ---------- 回声去重 ----------
    # 用外放时（没戴耳机），对方的声音会被麦克风再录一遍，
    # 于是同一句话既出现在"对方"，又被当成"我方"再翻一遍。
    #
    # 主力是声学判定：把麦克风这一句和同一时刻音箱在放的内容做能量包络
    # 互相关。回声必然和音箱那段是同一个波形形状，而"对方刚才出过声"
    # 说明不了任何问题——通话时对方的声音几乎一直在响，按那个判会把
    # 自己说的话全部丢光。阈值是扫出来的，不是拍的，详见 translator/echo.py。
    #
    # 万一它在你的环境里误伤，把 enabled 改成 false 就退回纯文字去重。
    "echo_cancel": {
        "enabled": True,
        # "on"      正常工作：判定为回声就丢掉
        # "observe" 只报告不丢弃：终端会打印「这一句本来会被丢掉」+ 识别出的文字。
        #           拿不准它在你的环境里准不准，就先用这个开十分钟会核对一遍：
        #           印出来的是对方刚说过的话 = 判对了；是你自己说的 = 该关掉。
        "mode": "on",
        "corr": 0.85,                 # 包络相关度门限（长句）
        "short_corr": 0.95,           # 短句帧数少、证据弱，门槛单独抬高
        "cover": 0.70,                # 麦克风响着时，音箱也在响的比例
        "short_sec": 1.5,             # 短于这么久算"短句"
        "max_delay_ms": 700,          # 两路时间戳偏差的搜索范围
        "lock_tol_ms": 120,           # 标定后短句只在这个范围里找
        "tail_ms": 300,               # 房间混响拖尾，判覆盖率时留出这段余量
        "ref_floor": 0.004,           # 对方那一路低于这个音量就当没出声
    },

    # 文字兜底：声学没拦住时，再按识别出来的文字比一次。
    # 只丢我方的——对方那一路是从系统直接抓的，永远比麦克风干净。
    #
    # 阈值是从三场真实会议量出来的：麦克风听到的是被音箱弄糊的残片，
    # 识别出来的字和对方那条只有 0.51~0.76 像。
    #   0.82 → 一条拦不住　　0.72 → 8 条里拦住 2 条
    #   0.65 → 拦住 3 条，三场会零误伤　← 取这一档
    #   0.58 → 拦住 6 条，但会误伤真实对话里两人互相复述的句子（实测 2 条）
    # 窗口 10 秒：实测有一条间隔 6.1 秒，原来的 5 秒窗口正好漏掉。
    "dedupe_window_sec": 10.0,
    "dedupe_ratio": 0.65,             # 相似度超过这个值才算同一句

    # ---------- 口头语 ----------
    # 纯语气词（嗯/啊/哦/อืม…）直接不出字幕；"好的""收到""ครับ"这类
    # 短句查表秒回，不走云端。实测这两类占全部字幕的 24%，
    # 以前每条都要等 1~2 秒、占一行、花一次翻译调用。词表见 translator/quick.py。
    "drop_fillers": True,
    "quick_replies": True,

    # ---------- 会议记录 ----------
    # 每场会议自动存一份到「会议记录」文件夹：.md 给人看，.jsonl 给程序读。
    # 手动改过的句子会重写对应那行，文件里始终是修正后的版本。
    # 全部保存在本机，不会上传。涉密会议可以关掉。
    # 流式识别（WebSocket 双向流）。开了之后人一说话就往云端送，
    # 说的过程中文字就在回来，说完只剩收尾——实测能省约 1.1 秒。
    #
    # 默认关，两个原因：
    #   1. 计费方式完全不同。一句话识别按次数（每月免费 5000 次），
    #      流式按**音频秒数**（国内版 ¥3/小时，每月免费 5 小时）。
    #   2. 境外调用要单独开通「跨境」版：¥8.6/小时音频，**没有免费额度**。
    #      不开通的话握手会被拒（错误码 6001）。
    #      开通入口 https://console.cloud.tencent.com/asr/settings
    #
    # 任何一步失败（握手被拒、网络断、超时、文本为空）都会自动退回
    # 一句话识别，不会丢句子。
    "stream_asr": False,
    "stream_wait_sec": 2.5,           # 等流式结果最多这么久，超时就走批量
    # 这两项决定「⋯」菜单里显不显示对应的入口。都不影响默认行为——
    # 两个功能本来就是点了才起，不点什么都不会发生。
    # 设成 false 是为了**连入口都不要**：不想误点、或者发给别人时不想让他们看到。
    "fast_enabled": True,             # 显示「⚡ 极速模式」（按秒计费）
    "share_enabled": True,            # 显示「🔗 分享字幕」（开公网链接）
    "save_transcript": True,
    "transcript_dir": "会议记录",
    # 终端输出也存一份（回声判定、丢句原因、每分钟进出账）。
    # 窗口一关控制台就没了，出问题时没有任何依据可查。
    "save_log": True,

    # ---------- 会议纪要 ----------
    # 中泰双语纪要，三种触发方式：
    #   界面「⋯」菜单 →「生成会议纪要」（或按 M）——开完会随手点一下
    #   命令行  python main.py --minutes
    #   auto_minutes 改成 true，则每次退出程序时自动生成一份
    # 长会会先分段提要再合成，两小时的会大约要一两分钟。
    "auto_minutes": False,
    "minutes_model": "",              # 留空则沿用上面的 model
    "minutes_effort": "medium",       # 纪要不抢毫秒，可以让它想得久一点
    # 一次要吐三万个 token 的泰中双语长文，300 秒会不够，别在最后一步超时。
    "minutes_timeout": 600.0,
    # 泰文极耗 token：一份 1.7 万字的双语纪要里泰文就占 1.2 万字。
    # 给小了会在"待办事项"那一节之前就被截断，文件看着完整其实缺了最要紧的部分。
    "minutes_max_tokens": 32000,

    # ---------- 本地服务 ----------
    "server": {
        "host": "127.0.0.1",
        "port": 8765,
        "open_browser": True,
        # app     = 浏览器的应用窗口：没有标签栏、地址栏、书签栏，只有页面本身。
        #           字幕窗口只占屏幕 1/4 时，那些浏览器界面要吃掉上面三分之一。
        # browser = 普通浏览器标签页
        # none    = 不自动打开，自己访问上面的地址
        "window_mode": "app",
        "window_size": [900, 620],
    },

    # ---------- 识别纠错表（专业词老被听错时填这里） ----------
    # 在翻译之前直接替换识别结果，确定性修正，不依赖模型猜。
    # 格式： "听错的样子": "正确的词"。左边可以写多个变体。
    "asr_corrections": {
        # "厌厂": "验厂",
        # "严厂": "验厂",
        # "验长": "验厂"
    },

    # ---------- 术语表（强烈建议填！公司名/产品名/人名） ----------
    # 格式： "中文或原文": "对应译法"
    # 这里的词也会告诉 Claude "识别可能听错，遇到发音相近的请还原成这些词"
    "glossary": {
        # "泰佳物流": "ไทยเจีย โลจิสติกส์",
        # "验厂": "ตรวจโรงงาน",
        # "验货": "ตรวจสอบสินค้า"
    },

    # ---------- 额外提示（可写会议背景，帮助翻译更准） ----------
    "domain_hint": "跨境电商 / 物流 / 商务合作的线上会议",
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# 早期版本留下的、现在没有任何代码读的键。留着会误导——
# 看到 echo_guard: true 会以为回声功能是它在控制，其实控制它的是 echo_cancel。
OBSOLETE = ("echo_guard", "echo_hold_ms")


def _write_json(data: dict) -> bool:
    """先写临时文件再原子替换：写到一半断电也不会留下半个配置文件。"""
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, CONFIG_PATH)
        return True
    except Exception as e:
        print(f"[配置] 写 config.json 失败：{e}")
        try:
            tmp.unlink()
        except Exception:
            pass
        return False


def _sync_file(user: dict, merged: dict) -> None:
    """程序升级后新增的设置项，补写进 config.json，顺手删掉废弃的键。

    不补的话，新功能只存在于代码的默认值里——文件里看不到，
    用户也就无从知道有这些开关、更没法改。（实测就吃过亏：
    我让用户把 echo_cancel.mode 改成 observe，他文件里压根没这个键。）

    必须在读环境变量之前调用：load_config 之后会把环境变量里的
    ANTHROPIC_API_KEY 填进 merged，那时候再写回就等于把密钥落盘了。
    """
    missing = [k for k in DEFAULTS if k not in user]
    dead = [k for k in OBSOLETE if k in user]
    if not missing and not dead:
        return
    out = copy.deepcopy(merged)
    for k in dead:
        out.pop(k, None)
    if _write_json(out):
        if missing:
            print(f"[配置] 已补充 {len(missing)} 项新设置到 config.json："
                  f"{'、'.join(missing[:6])}{'…' if len(missing) > 6 else ''}")
        if dead:
            print(f"[配置] 已移除废弃的设置：{'、'.join(dead)}")


def save_fields(updates: dict) -> bool:
    """把少数几个字段写回 config.json，文件里其余内容原样不动。

    两条必须守住的规矩：

    1. **只能读磁盘原文来改，不能把内存里的 cfg 写回去。**
       load_config() 会把环境变量里的 ANTHROPIC_API_KEY 合并进内存，
       内存那份整个写回去就等于把密钥落盘了。

    2. **先写临时文件再原子替换。** 直接覆盖写到一半断电，
       换来的是一个被截断的 config.json——里面还带着密钥，
       用户得重新配一遍。

    updates 用嵌套 dict 表示，例如 {"language_lock": {"them": "th"}}，
    只覆盖点到的那几个叶子，同级的其它键保留。
    """
    if not CONFIG_PATH.exists():
        return False
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[配置] config.json 读不出来，这次不写回：{e}")
        return False
    if not isinstance(raw, dict):
        return False

    def merge(dst: dict, src: dict):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = v

    merge(raw, updates)

    return _write_json(raw)


def load_config() -> dict:
    """读取 config.json；不存在就写一份默认的。"""
    user = {}
    if CONFIG_PATH.exists():
        try:
            user = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as e:  # 配置写坏了不要直接崩
            print(f"[配置] config.json 解析失败，暂用默认配置：{e}")
            user = {}
    else:
        # 第一次运行：优先照 config.example.json 生成，那份里有注释性的
        # 示例术语表和纠错表，比一份干巴巴的默认值更容易看懂该怎么配。
        # （config.example.json 是仓库里的模板，密钥是空的；
        #   config.json 是你自己的，被 .gitignore 挡着，不会进仓库。）
        example = CONFIG_PATH.with_name("config.example.json")
        seed = DEFAULTS
        if example.exists():
            try:
                seed = _deep_merge(DEFAULTS, json.loads(
                    example.read_text(encoding="utf-8")))
            except Exception:
                seed = DEFAULTS
        CONFIG_PATH.write_text(
            json.dumps(seed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[配置] 已生成 {CONFIG_PATH.name}，请填入 API Key 后重启。")

    cfg = _deep_merge(DEFAULTS, user)

    # 补齐升级后新增的设置项。**必须在下面读环境变量之前做**，
    # 否则会把环境变量里的密钥写进文件。
    if CONFIG_PATH.exists():
        _sync_file(user, cfg)

    # 环境变量兜底
    if not cfg["anthropic_api_key"]:
        cfg["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
    if not cfg["anthropic_base_url"]:
        cfg["anthropic_base_url"] = os.environ.get("ANTHROPIC_BASE_URL", "")
    # OpenAI 的 Key 识别和翻译共用一个：填在顶层，识别那边也自动拿到，
    # 免得同一个 Key 要在两处各填一遍（填漏一处就只有一半能用）。
    if not cfg["openai_api_key"]:
        cfg["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")
    if not cfg["asr"]["openai_api_key"]:
        cfg["asr"]["openai_api_key"] = (cfg["openai_api_key"]
                                        or os.environ.get("OPENAI_API_KEY", ""))
    if not cfg["openai_api_key"]:
        cfg["openai_api_key"] = cfg["asr"]["openai_api_key"]
    if not cfg["openai_base_url"]:
        cfg["openai_base_url"] = os.environ.get("OPENAI_BASE_URL", "")
    # 识别提示没填就拿术语表凑一份——这些词本来就是最容易被听错的
    if not cfg["asr"].get("openai_prompt"):
        terms = list((cfg.get("glossary") or {}).keys())[:40]
        if terms:
            cfg["asr"]["openai_prompt"] = "、".join(terms)
    # 纠错表也传给识别那一层，OpenAI 分支要用
    cfg["asr"].setdefault("asr_corrections", cfg.get("asr_corrections") or {})

    return cfg
