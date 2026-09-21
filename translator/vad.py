"""断句：把连续音频流切成一句一句的语音片段。

用 WebRTC VAD 判断有没有人声，配合能量门限过滤底噪 / 键盘声。
没装 webrtcvad 时自动退回纯能量判断。
"""
from __future__ import annotations

import collections

import numpy as np

SR = 16000
FRAME_MS = 30
FRAME_LEN = SR * FRAME_MS // 1000  # 480 个采样点

try:
    import webrtcvad

    _HAS_WEBRTC = True
except ImportError:
    _HAS_WEBRTC = False


class Segmenter:
    """流式断句器：不断 push 音频，返回已经说完的整句。"""

    def __init__(self, cfg: dict):
        self.start_frames = int(cfg.get("start_frames", 3))
        self.end_frames = int(cfg.get("end_frames", 22))
        self.max_frames = int(float(cfg.get("max_seconds", 8.0)) * 1000 / FRAME_MS)
        self.min_frames = int(float(cfg.get("min_seconds", 0.4)) * 1000 / FRAME_MS)
        self.rms_threshold = float(cfg.get("rms_threshold", 0.006))

        preroll_frames = max(1, int(cfg.get("preroll_ms", 320)) // FRAME_MS)
        self._preroll = collections.deque(maxlen=preroll_frames)

        self._vad = None
        if _HAS_WEBRTC:
            try:
                self._vad = webrtcvad.Vad(int(cfg.get("aggressiveness", 2)))
            except Exception:
                self._vad = None

        self._tail = np.zeros(0, dtype=np.float32)   # 不足一帧的余量
        self._frames: list[np.ndarray] = []          # 当前句子已收的帧
        self._flags: list[bool] = []                 # 每帧有没有人声，用于找切点
        self._speaking = False
        self._voiced_run = 0
        self._silence_run = 0
        self._voiced_in_seg = 0                      # 当前句里真正有人声的帧数
        self._seg_start_ts = 0.0                     # 当前句开始的挂钟时刻
        self.last_voice_ts = 0.0                     # 最近一次检测到人声（外部用于回声抑制）

        # WASAPI 环回在系统完全没出声时是不产数据的（而不是产静音帧），
        # 光靠静音帧计数永远等不到"说完了"。用挂钟时间兜底。
        self._last_frame_ts = 0.0
        self._stall_timeout = self.end_frames * FRAME_MS / 1000.0

    # ------------------------------------------------------------------
    def tick(self, now: float) -> list[tuple[np.ndarray, float]]:
        """没有新音频进来时调用：超时就把当前这句收尾。"""
        done: list[tuple[np.ndarray, float]] = []
        if self._speaking and self._last_frame_ts and \
                now - self._last_frame_ts > self._stall_timeout:
            self._emit(done)
        return done

    def push(self, pcm: np.ndarray, now: float) -> list[tuple[np.ndarray, float]]:
        """喂入一段音频，返回这次产生的完整句子：[(音频, 这句开始的时刻), ...]。"""
        done = self.tick(now)      # 先处理"断流"的情况
        self._last_frame_ts = now

        buf = np.concatenate([self._tail, pcm]) if self._tail.size else pcm
        n_full = len(buf) // FRAME_LEN
        self._tail = buf[n_full * FRAME_LEN:].copy()

        for i in range(n_full):
            self._feed_frame(buf[i * FRAME_LEN:(i + 1) * FRAME_LEN], now, done)
        return done

    # ------------------------------------------------------------------
    def _is_voice(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-12))
        if rms < self.rms_threshold:
            return False
        if self._vad is None:
            return rms > self.rms_threshold * 2.0
        pcm16 = np.clip(frame * 32767.0, -32768, 32767).astype(np.int16)
        try:
            return self._vad.is_speech(pcm16.tobytes(), SR)
        except Exception:
            return rms > self.rms_threshold * 2.0

    def _feed_frame(self, frame: np.ndarray, now: float, done: list) -> bool:
        voiced = self._is_voice(frame)
        if voiced:
            self.last_voice_ts = now

        if not self._speaking:
            self._preroll.append(frame)
            if voiced:
                self._voiced_run += 1
                if self._voiced_run >= self.start_frames:
                    # 开始一句：把 preroll 也带上，避免吞头
                    self._speaking = True
                    self._frames = list(self._preroll)
                    self._flags = [False] * (len(self._frames) - self._voiced_run) + \
                                  [True] * self._voiced_run
                    self._preroll.clear()
                    self._silence_run = 0
                    self._voiced_in_seg = self._voiced_run
                    self._seg_start_ts = now - len(self._frames) * FRAME_MS / 1000.0
            else:
                self._voiced_run = 0
            return False

        # 说话中
        self._frames.append(frame)
        self._flags.append(voiced)
        if voiced:
            self._voiced_in_seg += 1
        self._silence_run = 0 if voiced else self._silence_run + 1

        if self._silence_run >= self.end_frames:
            self._emit(done)
            return True

        if len(self._frames) >= self.max_frames:
            # 说太久了必须切一刀，否则字幕要等到他说完才出。
            # 但别硬切在当前位置——那多半正落在词中间。
            self._force_cut(done)
            return True
        return False

    def _find_pause(self) -> int | None:
        """在后半段里找最长的一处停顿，返回切点（帧序号）。"""
        n = len(self._flags)
        start = int(n * 0.5)
        best_len, best_at, run = 0, None, 0
        for i in range(start, n):
            if not self._flags[i]:
                run += 1
                if run > best_len:
                    best_len, best_at = run, i - run // 2
            else:
                run = 0
        # 至少 3 帧（90ms）静音才算一处停顿，否则宁可硬切
        return best_at if best_len >= 3 else None

    def _force_cut(self, done: list):
        """长句强切：尽量切在停顿处，剩下的接着当下一句的开头。"""
        cut = self._find_pause() or len(self._frames)
        head, head_flags = self._frames[:cut], self._flags[:cut]
        tail, tail_flags = self._frames[cut:], self._flags[cut:]

        if sum(head_flags) >= self.min_frames:
            done.append((np.concatenate(head), self._seg_start_ts, True))

        # 继续录下半句，时间戳往后推
        self._seg_start_ts += len(head) * FRAME_MS / 1000.0
        self._frames, self._flags = tail, tail_flags
        self._voiced_in_seg = sum(tail_flags)
        self._silence_run = 0
        self._voiced_run = 0
        self._preroll.clear()

    def _emit(self, done: list):
        frames = self._frames
        voiced = self._voiced_in_seg
        start_ts = self._seg_start_ts
        self._frames, self._flags = [], []
        self._voiced_run = 0
        self._silence_run = 0
        self._voiced_in_seg = 0
        self._speaking = False
        self._preroll.clear()

        # 只看真正有人声的时长：preroll 和尾部静音不算，
        # 否则 0.15 秒的咳嗽/键盘声也会凑够长度送进识别，引发幻觉字幕。
        if voiced >= self.min_frames:
            done.append((np.concatenate(frames), start_ts, False))

    def current_audio(self) -> np.ndarray | None:
        """正在说的这半句（还没说完）。给"边说边显示"用。"""
        if not self._speaking or not self._frames:
            return None
        return np.concatenate(self._frames)

    @property
    def speaking(self) -> bool:
        return self._speaking

    def flush(self) -> list[tuple[np.ndarray, float, bool]]:
        """把攒着的半句吐出来（够长的话），然后把状态清干净。

        必须清**全部**状态，不能只清 _frames：preroll 里存着最近 320ms 的音频，
        _tail 里存着不足一帧的余量，_voiced_in_seg 记着已经数到的人声帧数。
        只清一半的话，暂停再继续时，上一段的残留会被接到新句子的开头，
        而且旧的计数会让下一句提前满足 min_frames。
        """
        done: list[tuple[np.ndarray, float, bool]] = []
        if self._speaking and self._voiced_in_seg >= self.min_frames:
            done.append((np.concatenate(self._frames), self._seg_start_ts, False))
        self._frames, self._flags = [], []
        self._speaking = False
        self._preroll.clear()
        self._tail = np.zeros(0, dtype=np.float32)
        self._voiced_run = 0
        self._silence_run = 0
        self._voiced_in_seg = 0
        self._seg_start_ts = 0.0
        self._last_frame_ts = 0.0
        return done
