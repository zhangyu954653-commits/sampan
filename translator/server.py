"""本地网页服务：提供字幕界面 + WebSocket 实时推送。"""
from __future__ import annotations

import asyncio
import json
import secrets
from pathlib import Path

from aiohttp import WSMsgType, web

WEB_DIR = Path(__file__).resolve().parent / "web"


class Server:
    def __init__(self, cfg: dict):
        self.cfg = cfg["server"]
        self.clients: set[web.WebSocketResponse] = set()
        self.pipeline = None       # 由 main 注入
        self.static_status: dict | None = None  # 演示模式下用，没有 pipeline 也能显示状态
        self._minutes_task: asyncio.Task | None = None
        # 公网分享：口令每次启动重新生成，旧链接自动失效
        self.share_token = ""
        self.tunnel = None
        self.viewers: set[web.WebSocketResponse] = set()   # 只读的远端观众
        self.app = web.Application()
        self.app.add_routes([
            web.get("/", self.index),
            web.get("/ws", self.ws),
            web.static("/static", str(WEB_DIR)),
        ])

    # ------------------------------------------------------------------
    def _is_local(self, request: web.Request) -> bool:
        """本机开的窗口 vs 从公网隧道进来的人。

        cloudflared 是在本机连过来的，所以看源 IP 分不出来；
        但 Host 头会带上 trycloudflare 的域名，本机窗口则是 127.0.0.1:8765。
        """
        host = (request.host or "").split(":")[0].lower()
        return host in ("127.0.0.1", "localhost", "::1", "[::1]")

    def _check(self, request: web.Request) -> bool:
        """远端必须带对口令。本机不用。"""
        if self._is_local(request):
            return True
        if not self.share_token:
            return False                      # 没开分享就谁也别想进
        got = request.query.get("k") or request.cookies.get("k") or ""
        return secrets.compare_digest(got, self.share_token)

    async def index(self, request: web.Request) -> web.StreamResponse:
        if not self._check(request):
            return web.Response(status=403, content_type="text/html",
                                text="<meta charset=utf-8>"
                                     "<h3>链接无效或口令不对</h3>"
                                     "<p>请向会议发起人要一个新链接。</p>")
        resp = web.FileResponse(WEB_DIR / "index.html")
        if not self._is_local(request):
            # 把口令种进 cookie，页面里的 /ws 请求就不用再拼参数
            resp.set_cookie("k", self.share_token, max_age=86400,
                            samesite="Lax")
        return resp

    async def ws(self, request: web.Request) -> web.StreamResponse:
        if not self._check(request):
            return web.Response(status=403, text="forbidden")
        viewer = not self._is_local(request)
        ws = web.WebSocketResponse(heartbeat=25)
        await ws.prepare(request)
        self.clients.add(ws)
        if viewer:
            # 远端只能看。不能暂停你的会、不能开按秒计费的极速模式、
            # 不能改你的字、不能清屏——这些都是会影响到你这边的操作。
            self.viewers.add(ws)
            await ws.send_json({"type": "viewer", "on": True})

        # 新连接：补一份当前状态和历史字幕。
        #
        # **远端观众从"现在"开始看，不补历史。** 你多半是开会到一半才
        # 把链接发给客户或供应商的，而这之前很可能刚讨论完报价底线、
        # 内部分歧这些不该给对方看的内容。本机窗口照旧补最近 40 句，
        # 那是你自己的屏幕。
        if self.pipeline is not None:
            await ws.send_json({"type": "status", **self.pipeline.public_status()})
            if not viewer:
                for item in self.pipeline.history[-40:]:
                    await ws.send_json({"type": "utterance", "item": item})
        elif self.static_status:
            await ws.send_json({"type": "status", **self.static_status})
        await ws.send_json(self.share_status())
        if viewer:
            # 有人进来了，告诉本机那边现在几个人在看
            await self.broadcast(self.share_status())

        try:                                   # 让界面上的置顶按钮和真实状态对上
            from .window import is_topmost

            await ws.send_json({"type": "topmost", "on": is_topmost()})
        except Exception:
            pass

        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                # 只读拦在服务端，不能只靠界面把按钮藏起来——
                # 藏起来只是看不见，自己拼一条消息照样能发过来。
                if viewer:
                    continue
                await self._handle(data)
        finally:
            self.clients.discard(ws)
            self.viewers.discard(ws)
        return ws

    async def _handle(self, data: dict):
        action = data.get("action")

        # 置顶不依赖 pipeline（演示模式下也能用），所以放在前面
        if action == "topmost":
            from .window import is_topmost, set_topmost

            want = bool(data.get("on"))
            set_topmost(want)
            await self.broadcast({"type": "topmost", "on": is_topmost()})
            return

        if self.pipeline is None:
            return
        if action == "precheck":
            # 让 pipeline 知道分享有没有开着——会前检查要提醒这个
            self.pipeline._server_share = bool(self.share_token and self.tunnel)
            await self.broadcast(await self.pipeline.precheck())
        elif action == "share":
            await self.set_share(bool(data.get("value")))
        elif action == "fast":
            await self.pipeline.set_fast(bool(data.get("value")))
            await self.broadcast({"type": "status", **self.pipeline.public_status()})
        elif action == "pause":
            self.pipeline.set_paused(bool(data.get("value")))
            await self.broadcast({"type": "status", **self.pipeline.public_status()})
        elif action == "mute":
            self.pipeline.set_muted(data.get("channel", ""), bool(data.get("value")))
            await self.broadcast({"type": "status", **self.pipeline.public_status()})
        elif action == "clear":
            self.pipeline.clear_history()
            await self.broadcast({"type": "cleared"})
        elif action == "lock":
            self.pipeline.set_lock(data.get("channel", ""), data.get("lang"))
            await self.broadcast({"type": "status", **self.pipeline.public_status()})
        elif action == "retranslate":
            try:
                await self.pipeline.retranslate(int(data.get("id")),
                                                str(data.get("text", "")))
            except Exception as e:
                print(f"[纠错] 失败：{e}")
        elif action == "minutes":
            # 整理一份纪要要几十秒到几分钟，不能占着消息循环——
            # 占住了字幕就停了。丢到后台任务里跑，进度另行推送。
            if getattr(self, "_minutes_task", None) and \
                    not self._minutes_task.done():
                await self.broadcast({"type": "minutes", "state": "busy",
                                      "message": "纪要正在生成中，请稍候"})
                return
            self._minutes_task = asyncio.create_task(self._make_minutes())
        elif action == "open_folder":
            import os
            import subprocess
            from pathlib import Path

            target = Path(str(data.get("path") or ""))
            base = Path(__file__).resolve().parent.parent
            try:
                # 只允许打开程序自己目录下的东西，别让页面指使我们开任意路径
                target = target.resolve()
                target.relative_to(base.resolve())
            except Exception:
                return
            folder = target if target.is_dir() else target.parent
            if not folder.is_dir():
                return
            try:
                if os.name == "nt":
                    subprocess.Popen(["explorer", str(folder)])
                else:
                    subprocess.Popen(["xdg-open", str(folder)])
            except Exception as e:
                print(f"[纪要] 打不开文件夹：{e}")

    async def _make_minutes(self):
        loop = asyncio.get_running_loop()

        def progress(msg):
            # 从工作线程/回调里投递，不能直接 await
            asyncio.run_coroutine_threadsafe(
                self.broadcast({"type": "minutes", "state": "working",
                                "message": msg}), loop)

        await self.broadcast({"type": "minutes", "state": "working",
                              "message": "正在整理会议纪要…"})
        try:
            res = await self.pipeline.make_minutes(progress)
        except Exception as e:
            print(f"[纪要] 失败：{e}")
            await self.broadcast({"type": "minutes", "state": "error",
                                  "message": str(e)})
            return
        await self.broadcast({
            "type": "minutes", "state": "done",
            "message": "会议纪要已生成",
            "md": res.get("md", ""), "docx": res.get("docx", ""),
            "markdown": res.get("markdown", ""),
        })

    # ------------------------------------------------------------------
    def _share_allowed(self) -> bool:
        p = self.pipeline
        if p is not None:
            return bool(p.cfg.get("share_enabled", True))
        return True

    async def set_share(self, value: bool):
        """开/关公网分享。开的时候会拉起隧道，可能要等几秒。"""
        from pathlib import Path as _P

        from .tunnel import Tunnel, make_token

        # config.json 里关掉了就是真的关掉，不只是界面上看不见
        if value and not self._share_allowed():
            print("[分享] config.json 里 share_enabled 是 false，没开这个功能")
            return

        if not value:
            if self.tunnel is not None:
                self.tunnel.stop()
                self.tunnel = None
            self.share_token = ""
            # 把已经连进来的远端踢掉：链接一关就该立刻失效
            for ws in list(self.viewers):
                try:
                    await ws.close()
                except Exception:
                    pass
            self.viewers.clear()
            print("[分享] 已关闭，链接立即失效")
            await self._push_share()
            return

        if self.tunnel is not None and self.tunnel.alive:
            return
        await self._push_share(busy=True)
        self.share_token = make_token()
        t = Tunnel(int(self.cfg["port"]), _P(__file__).resolve().parent.parent)
        ok, why = await t.start()
        if not ok:
            self.share_token = ""
            self.tunnel = None
            print(f"[分享] 开不了：{why}")
            await self._push_share(error=why)
            return
        self.tunnel = t
        link = f"{t.url}/?k={self.share_token}"
        print("\n[分享] 公网字幕链接已开启，把下面这条发给对方：")
        print(f"        {link}")
        print("        对方只能看，不能操作你这边。关掉分享链接立即失效。\n")
        await self._push_share()

    def share_status(self, busy=False, error="") -> dict:
        url = ""
        if self.tunnel is not None and self.tunnel.url and self.share_token:
            url = f"{self.tunnel.url}/?k={self.share_token}"
        return {"type": "share", "on": bool(url), "url": url,
                "busy": busy, "error": error,
                "viewers": len(self.viewers)}

    async def _push_share(self, busy=False, error=""):
        await self.broadcast(self.share_status(busy, error))

    async def broadcast(self, event: dict):
        if not self.clients:
            return
        payload = json.dumps(event, ensure_ascii=False)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_str(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    # ------------------------------------------------------------------
    async def run(self) -> str:
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.cfg["host"], int(self.cfg["port"]))
        await site.start()
        self._runner = runner
        return f"http://{self.cfg['host']}:{self.cfg['port']}"

    async def close(self):
        if self.tunnel is not None:
            self.tunnel.stop()      # 隧道是独立进程，不收掉会一直留着
            self.tunnel = None
        for ws in list(self.clients):
            await ws.close()
        runner = getattr(self, "_runner", None)
        if runner:
            await runner.cleanup()
