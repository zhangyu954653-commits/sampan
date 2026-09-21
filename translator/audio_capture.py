"""双路音频采集：麦克风（你）+ 系统声音环回（对方）。

Windows 上用 WASAPI Loopback 直接抓扬声器输出，不需要装虚拟声卡 / 立体声混音。
统一输出 16kHz 单声道 float32，塞进同一个队列。
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

try:
    import pyaudiowpatch as pyaudio
except ImportError:  # 非 Windows 或没装 patch 版，退回普通 pyaudio（只能抓麦克风）
    import pyaudio  # type: ignore

TARGET_SR = 16000

# ---------- 重采样 ----------
try:
    import soxr

    def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if sr_in == sr_out:
            return x.astype(np.float32, copy=False)
        return soxr.resample(x, sr_in, sr_out).astype(np.float32, copy=False)

except ImportError:  # 退化方案：线性插值，够用但音质略差

    def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if sr_in == sr_out:
            return x.astype(np.float32, copy=False)
        n = int(round(len(x) * sr_out / float(sr_in)))
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        idx = np.linspace(0, len(x) - 1, n)
        return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


# ---------- 设备查找 ----------
def list_devices() -> list[dict]:
    p = pyaudio.PyAudio()
    try:
        out = []
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if int(d.get("maxInputChannels", 0)) <= 0:
                continue
            out.append(
                {
                    "index": i,
                    "name": d.get("name", ""),
                    "channels": int(d.get("maxInputChannels", 0)),
                    "rate": int(d.get("defaultSampleRate", 0)),
                    "loopback": bool(d.get("isLoopbackDevice", False)),
                }
            )
        return out
    finally:
        p.terminate()


def find_loopback_device(p, index: int | None = None) -> dict | None:
    """找到"默认扬声器"对应的环回输入设备。"""
    if index is not None:
        return p.get_device_info_by_index(index)

    # 新版 PyAudioWPatch 直接提供
    getter = getattr(p, "get_default_wasapi_loopback", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            pass

    # 兜底：找默认输出设备同名的 loopback
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        speakers = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    except Exception:
        return None

    if speakers.get("isLoopbackDevice", False):
        return speakers

    gen = getattr(p, "get_loopback_device_info_generator", None)
    if callable(gen):
        try:
            for lb in gen():
                if speakers["name"] in lb["name"]:
                    return lb
        except Exception:
            pass
    return None


def find_input_device(p, index: int | None = None) -> dict | None:
    if index is not None:
        return p.get_device_info_by_index(index)
    try:
        return p.get_default_input_device_info()
    except Exception:
        return None


# ---------- 采集线程 ----------
class _ReaderThread(threading.Thread):
    """一路音频：持续读取已打开的流 → 转单声道 16k float32 → 投进队列。

    注意：流由 AudioCapture 统一打开。PortAudio 的初始化不是线程安全的，
    多个线程各自 new 一个 PyAudio() 会让设备列表失效（表现为 Invalid device）。
    """

    def __init__(self, channel: str, stream, rate: int, channels: int,
                 fmt, frames: int, out_q: queue.Queue, gain: float = 1.0):
        super().__init__(daemon=True, name=f"audio-{channel}")
        self.channel = channel
        self.stream = stream
        self.rate = rate
        self.channels = channels
        self.frames = frames
        self.out_q = out_q
        self.gain = float(gain)
        self._dtype = np.int16 if fmt == pyaudio.paInt16 else np.float32
        self._scale = 1.0 / 32768.0 if self._dtype == np.int16 else 1.0
        self._stop_evt = threading.Event()
        self.error: str | None = None

    def stop(self):
        self._stop_evt.set()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                raw = self.stream.read(self.frames, exception_on_overflow=False)
            except Exception as e:
                if self._stop_evt.is_set():
                    break
                self.error = str(e)
                time.sleep(0.1)
                continue

            buf = np.frombuffer(raw, dtype=self._dtype).astype(np.float32) * self._scale
            if self.channels > 1:
                usable = (len(buf) // self.channels) * self.channels
                buf = buf[:usable].reshape(-1, self.channels).mean(axis=1)
            if self.gain != 1.0:
                buf = buf * self.gain

            pcm = resample(buf, self.rate, TARGET_SR)
            if pcm.size:
                try:
                    self.out_q.put_nowait((self.channel, pcm))
                except queue.Full:
                    pass  # 处理跟不上就丢帧，宁可丢也不要卡住采集


class AudioCapture:
    """管理两路采集：共用一个 PyAudio 实例，避免 PortAudio 竞态。"""

    BLOCK_MS = 30

    def __init__(self, cfg: dict, out_q: queue.Queue):
        self.cfg = cfg
        self.out_q = out_q
        self.threads: list[_ReaderThread] = []
        self.streams: list = []
        self.info: dict[str, str] = {}
        self._pa = None

    def start(self) -> dict[str, str]:
        a = self.cfg["audio"]
        self._pa = pyaudio.PyAudio()

        plan = []
        if a.get("enable_mic", True):
            dev = find_input_device(self._pa, a.get("mic_device_index"))
            if dev:
                plan.append(("me", dev, float(a.get("mic_gain", 1.0))))
            else:
                print("[音频] 没找到麦克风输入设备。")
        if a.get("enable_loopback", True):
            dev = find_loopback_device(self._pa, a.get("loopback_device_index"))
            if dev:
                plan.append(("them", dev, float(a.get("loopback_gain", 1.0))))
            else:
                print("[音频] 没找到系统声音环回设备（对方说话将无法识别）。")

        for channel, dev, gain in plan:
            try:
                stream, rate, channels, fmt, frames = self._open(dev)
            except Exception as e:
                print(f"[音频] {channel} 打开失败：{e}")
                continue
            self.streams.append(stream)
            self.info[channel] = dev["name"]
            self.threads.append(
                _ReaderThread(channel, stream, rate, channels, fmt, frames,
                              self.out_q, gain)
            )

        for t in self.threads:
            t.start()
        return self.info

    def _open(self, dev: dict):
        """按设备原生参数打开；失败则退而求其次试几种常见组合。"""
        native_rate = int(dev.get("defaultSampleRate") or 48000)
        native_ch = max(1, int(dev.get("maxInputChannels") or 1))
        index = int(dev["index"])

        combos = []
        for fmt in (pyaudio.paInt16, pyaudio.paFloat32):
            combos.append((fmt, native_rate, native_ch))
        for fmt in (pyaudio.paInt16, pyaudio.paFloat32):
            for rate in (48000, 44100, 16000):
                for ch in (native_ch, 1, 2):
                    if (fmt, rate, ch) not in combos:
                        combos.append((fmt, rate, ch))

        last = None
        for fmt, rate, ch in combos:
            frames = max(160, int(rate * self.BLOCK_MS / 1000))
            try:
                stream = self._pa.open(
                    format=fmt, channels=ch, rate=rate, input=True,
                    frames_per_buffer=frames, input_device_index=index,
                )
                return stream, rate, ch, fmt, frames
            except Exception as e:
                last = e
        raise RuntimeError(f"无法打开音频设备 [{index}] {dev.get('name')}：{last}")

    def stop(self):
        for t in self.threads:
            t.stop()
        for t in self.threads:
            t.join(timeout=1.0)
        for s in self.streams:
            try:
                s.stop_stream()
                s.close()
            except Exception:
                pass
        self.streams.clear()
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
