"""回声识别：判断麦克风收到的这一句，是不是音箱里放出来的对方的声音。

为什么不能靠文字比对（这是真实会议记录量出来的结论）：
  · 819 条字幕里，按"相似度 0.82 / 5 秒窗口"一条都没拦住；
  · 真实回声的文字相似度落在 0.56~0.82，因为两条路听到的是同一句话，
    但麦克风那一路被音箱和房间弄糊了，识别出来的字本来就不一样；
  · 更要命的是两路的先后不固定——实测有 -0.8 秒的，即麦克风那条先到。
    按文字去重只会"留下先到的"，于是留下糊的那条、丢掉清楚的那条。

所以改成看声音本身：回声必然是"音箱先响、麦克风后收到"，
两者的**能量包络**（哪几个音节响、哪几个停）是同一个形状。
包络这个特征能扛住音箱染色、房间混响和编码损失，正好适合。

判定要过两关，都过才算回声：
  相关度  麦克风包络和对方那路包络的最大互相关（允许 0~±lag 的延迟）
  覆盖率  麦克风"响着"的时刻里，有多大比例对方那路也正响着

单看相关度不行：对方一直在说话时，覆盖率天然就高。
单看覆盖率也不行：两人同时说话（double-talk）时覆盖率也高，
但麦克风里自己的声音近得多、能量形状完全不同，相关度会掉下来。

**宁可放过，绝不错杀。** 任何一步拿不准都返回"不是回声"——
把用户自己说的话丢掉，这个软件就废了。
"""
from __future__ import annotations

import collections

import numpy as np

SR = 16000
FRAME = 320          # 20ms 一帧，包络的时间分辨率


def envelope(x: np.ndarray) -> np.ndarray:
    """逐帧 RMS，就是"这一小段有多响"的曲线。"""
    n = len(x) // FRAME
    if n == 0:
        return np.zeros(0)
    f = x[: n * FRAME].reshape(n, FRAME).astype(np.float64)
    return np.sqrt((f * f).mean(axis=1))


def dilate(mask: np.ndarray, back: int, fwd: int = 2) -> np.ndarray:
    """把"正响着"的标记沿时间轴摊开。

    回声不会和参考严丝合缝地同起同落：音箱停了之后，房间的混响还要
    拖几百毫秒才衰下去。不摊开的话，拖尾那几帧就会被算成
    "麦克风在响但对方没在响"，覆盖率白白掉一大截——
    实测重混响下会从 96% 掉到 41%。
    """
    out = mask.copy()
    for d in range(1, back + 1):
        out[d:] |= mask[:-d]          # 参考早 d 帧，回声可以晚到这里
    for d in range(1, fwd + 1):
        out[:-d] |= mask[d:]          # 对齐本身有半帧误差，往前也留一点
    return out


def best_align(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """把 a 在更长的 b 上滑动，返回 (最大相关系数, 最佳位置)。

    用皮尔逊相关而不是普通内积：前者对整体音量免疫，
    回声比原声小十几个 dB，用内积会算出很低的值。
    """
    na, nb = len(a), len(b)
    if na == 0 or nb < na:
        return 0.0, 0
    a0 = a - a.mean()
    da = float(np.sqrt((a0 * a0).sum()))
    if da <= 1e-12:
        return 0.0, 0

    best, best_k = -1.0, 0
    for k in range(nb - na + 1):
        w = b[k: k + na]
        w0 = w - w.mean()
        dw = float(np.sqrt((w0 * w0).sum()))
        if dw <= 1e-12:
            continue
        c = float((a0 * w0).sum()) / (da * dw)
        if c > best:
            best, best_k = c, k
    return max(best, 0.0), best_k


class EchoGuard:
    """存着对方那一路的近期音频，用来给麦克风的每一句做回声体检。"""

    def __init__(self, cfg: dict | None = None):
        c = dict(cfg or {})
        self.enabled = bool(c.get("enabled", True))
        # 阈值是扫出来的，不是拍的：各造 120 组"真回声"和"必须保留"，
        # 在零误判的前提下取漏判最少的一格（见 scratchpad/echo_sweep.py）。
        # 相关度 0.85 时：真回声拦住 119/120，自己说的话误丢 0/120。
        self.corr_min = float(c.get("corr", 0.85))
        self.cover_min = float(c.get("cover", 0.70))
        # 两路的时间戳都是"采集线程收到的时刻"，两个设备的缓冲深度不一样，
        # 偏差可正可负。±700ms 留足余量，代价只是多滑几十个位置。
        self.lag = float(c.get("max_delay_ms", 700)) / 1000.0
        self.keep = float(c.get("keep_sec", 30.0))
        self.min_sec = float(c.get("min_sec", 0.35))
        # 房间混响拖多久。回声判定时允许麦克风比参考晚这么久还在响。
        self.tail_frames = max(1, int(float(c.get("tail_ms", 300)) / 1000 * SR / FRAME))
        # 对方那一路低于这个 RMS 就当没出声，直接判"不是回声"
        self.ref_floor = float(c.get("ref_floor", 0.004))

        # 短句只有二十几帧，却要在 ±700ms 的 70 个位置里挑最大值——
        # 纯靠运气就能撞出 0.9 的相关度（实测 150 组短句误丢 23 组）。
        # 但两路采集的时间戳偏移是这台机器的固有常数，一场会里基本不变，
        # 所以先拿长句把它标定出来，之后短句只在标定值附近很窄的范围里找。
        self.short_sec = float(c.get("short_sec", 1.5))
        self.lock_tol = float(c.get("lock_tol_ms", 120)) / 1000.0
        self.lock_corr = float(c.get("lock_corr", 0.92))
        # 短句帧数少，同样的相关度证据力就弱，门槛要单独抬高。
        # 真回声的短句相关度中位数在 0.98，抬到 0.95 几乎不影响拦截率，
        # 却能把"自己说了半秒话正好撞上对方在说"这类误丢清零。
        self.short_corr = float(c.get("short_corr", 0.95))
        self._lag_est: float | None = None
        self._lag_hits: list[float] = []

        # 用 deque 而不是 list：这里每秒会被写进三十多次，
        # 用 list 过滤旧数据要整表重建一遍，白白占着采集线程。
        self._buf: collections.deque = collections.deque()   # (结束时刻, 音频)
        self.dropped = 0
        # 实测三场会议一次都没触发过，而那几场里确实有回声（文字兜底拦下了）。
        # 说明合成音频上调出来的阈值和真实房间对不上，但我手上没有真实音频，
        # 只能把每次的分数记下来，靠真实会议告诉我差多少、差在哪一项。
        self.scores: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    def push_reference(self, pcm: np.ndarray, now: float):
        """对方那一路每来一块音频就存一份，附上收到的时刻。"""
        if not self.enabled or pcm is None or len(pcm) == 0:
            return
        self._buf.append((now, pcm))
        cut = now - self.keep
        while self._buf and self._buf[0][0] < cut:
            self._buf.popleft()

    def reference(self, t0: float, t1: float) -> np.ndarray | None:
        """取出 [t0, t1) 这段时间里音箱在放什么。

        没数据的地方填 0——WASAPI 环回在系统不出声时根本不产数据，
        "没数据"本来就等于"当时是静音"，填 0 是对的。
        """
        n = int(round((t1 - t0) * SR))
        if n <= 0:
            return None
        out = np.zeros(n, dtype=np.float32)
        for t_end, pcm in self._buf:
            c0 = t_end - len(pcm) / SR
            a, b = max(c0, t0), min(t_end, t1)
            if b <= a:
                continue
            dst = int(round((a - t0) * SR))
            src = int(round((a - c0) * SR))
            ln = min(int(round((b - a) * SR)), n - dst, len(pcm) - src)
            if ln > 0:
                out[dst: dst + ln] = pcm[src: src + ln]
        return out

    # ------------------------------------------------------------------
    def _search_range(self, dur: float) -> tuple[float, float] | None:
        """这一句该在多大的时间范围里找对齐点。

        标定过设备偏移就只在它附近找；没标定过的话，短句一律不判——
        搜索范围一宽，短句就会撞出假的高相关，那等于随机丢用户的话。
        """
        if dur >= self.short_sec:
            # 长句帧数足够多，宽搜也撞不出假峰，而且正好用来持续校准偏移。
            # 长句永远走宽搜还有个好处：中途换了耳机/声卡导致偏移变了，
            # 下一句长句就会把标定值自动改过来，不会一直卡在旧值上。
            return -self.lag, self.lag
        if self._lag_est is not None:
            return self._lag_est - self.lock_tol, self._lag_est + self.lock_tol
        return None

    def _learn_lag(self, corr: float, offset: float, dur: float):
        """用长句里高置信度的对齐结果，把这台机器的两路时间差标定下来。"""
        if dur < self.short_sec or corr < self.lock_corr:
            return
        self._lag_hits.append(offset)
        self._lag_hits = self._lag_hits[-4:]
        if len(self._lag_hits) >= 2:
            last = self._lag_hits[-2:]
            if abs(last[0] - last[1]) <= self.lock_tol:
                self._lag_est = float(np.mean(last))

    def check(self, mic: np.ndarray, start_ts: float,
              end_ts: float) -> tuple[bool, str]:
        """这一句是回声吗？返回 (是否回声, 说明)。说明只用于打日志。"""
        if not self.enabled or mic is None:
            return False, ""
        dur = len(mic) / SR
        if dur < self.min_sec:
            return False, "太短，不判"

        mic_env = envelope(mic)
        if mic_env.size < 8:
            return False, "太短，不判"

        rng = self._search_range(dur)
        if rng is None:
            return False, "短句且尚未标定两路时间差，不判"
        lo, hi = rng

        ref = self.reference(start_ts + lo, end_ts + hi)
        if ref is None:
            return False, "没有参考音频"
        ref_env = envelope(ref)
        if ref_env.size <= mic_env.size:
            return False, "参考音频不够长"
        if float(ref_env.max()) < self.ref_floor:
            return False, "对方那一路当时没出声"

        corr, k = best_align(mic_env, ref_env)
        offset = lo + k * FRAME / SR
        self._learn_lag(corr, offset, dur)
        win = ref_env[k: k + mic_env.size]

        # "响着"的判定各自按自己的量级来，两路音量本来就差很多
        mic_hot = mic_env > max(mic_env.max() * 0.18, 1e-6)
        if not mic_hot.any():
            return False, "麦克风这句没有明显音节"
        ref_hot = win > max(win.max() * 0.12, self.ref_floor * 0.5)
        ref_hot = dilate(ref_hot, self.tail_frames)
        cover = float((mic_hot & ref_hot).sum()) / float(mic_hot.sum())

        self.scores.append((corr, cover))
        need = self.short_corr if dur < self.short_sec else self.corr_min
        why = (f"相关 {corr:.2f}／覆盖 {cover:.0%}／偏移 {offset * 1000:+.0f}ms"
               + ("／已标定" if self._lag_est is not None else ""))
        if corr >= need and cover >= self.cover_min:
            self.dropped += 1
            return True, why
        return False, why
