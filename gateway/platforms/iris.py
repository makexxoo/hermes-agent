"""
IRIS websocket platform adapter.

Connects Hermes gateway directly to an IRIS websocket endpoint, receives
normalized IRIS `message` events, and sends `reply` events with text/image
content parts.
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult

logger = logging.getLogger(__name__)


def check_iris_requirements() -> bool:
    """Check if IRIS adapter dependencies are available and minimally configured."""
    return AIOHTTP_AVAILABLE


class IrisAdapter(BasePlatformAdapter):
    """IRIS websocket adapter."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.IRIS)
        extra = config.extra or {}

        self._ws_url: str = str(extra.get("ws_url") or os.getenv("IRIS_WS_URL", "")).strip()
        self._auth_token: str = str(extra.get("auth_token") or os.getenv("IRIS_WS_TOKEN", "")).strip()
        self._reconnect_delay: float = float(extra.get("reconnect_delay_seconds", 5.0))
        self._connect_timeout: float = float(extra.get("connect_timeout_seconds", 30.0))
        # Dynamic streaming gate:
        # - True: enable GatewayStreamConsumer (progressive stream push)
        # - False: send only final complete response
        _streaming_env = (os.getenv("IRIS_STREAMING_ENABLED", "") or "").strip().lower()
        _streaming_cfg = extra.get("streaming_enabled")
        if _streaming_cfg is None:
            self._streaming_push_enabled = _streaming_env in ("1", "true", "yes", "on")
        else:
            self._streaming_push_enabled = bool(_streaming_cfg)
        # Gateway runner checks this flag before enabling stream consumer.
        self.SUPPORTS_MESSAGE_EDITING = bool(self._streaming_push_enabled)

        self._session: Optional["aiohttp.ClientSession"] = None
        self._ws: Optional["aiohttp.ClientWebSocketResponse"] = None
        self._listen_task: Optional[asyncio.Task] = None
        self._send_lock = asyncio.Lock()

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            logger.warning("[%s] aiohttp not installed", self.name)
            return False
        if not self._ws_url:
            logger.warning("[%s] IRIS websocket URL is not configured (IRIS_WS_URL)", self.name)
            return False

        self._running = True
        self._listen_task = asyncio.create_task(self._run_client_loop())
        self._mark_connected()
        logger.info(
            "[%s] Connecting to %s (streaming_push=%s)",
            self.name,
            self._ws_url,
            self._streaming_push_enabled,
        )
        return True

    async def disconnect(self) -> None:
        self._running = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None
        await self._close_ws()
        self._mark_disconnected()
        logger.info("[%s] Disconnected", self.name)

    async def _run_client_loop(self) -> None:
        while self._running:
            try:
                await self._connect_ws()
                await self._listen_messages()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("[%s] WS loop error: %s", self.name, e)
            finally:
                await self._close_ws()
            if self._running:
                await asyncio.sleep(self._reconnect_delay)

    async def _connect_ws(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self._connect_timeout)
        self._session = aiohttp.ClientSession(timeout=timeout)
        self._ws = await self._session.ws_connect(self._ws_url, heartbeat=30)
        if self._auth_token and self._ws:
            await self._ws.send_json({"type": "auth", "token": self._auth_token})
        logger.info("[%s] WS connected", self.name)

    async def _close_ws(self) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = None
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _listen_messages(self) -> None:
        if not self._ws:
            return
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_ws_text(msg.data)
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break

    async def _handle_ws_text(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return

        if not isinstance(payload, dict) or payload.get("type") != "message":
            return

        session_id = str(payload.get("sessionId") or "")
        content = payload.get("content") or {}
        text = ""
        if isinstance(content, dict):
            text = str(content.get("text") or "")
        if not text:
            return

        user_id = str(payload.get("channelUserId") or payload.get("userId") or "") or None
        source = self.build_source(
            chat_id=session_id,
            chat_name=session_id or "iris-session",
            chat_type="dm",
            user_id=user_id,
            user_name=user_id,
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.COMMAND if text.startswith("/") else MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=str(payload.get("id") or ""),
        )
        await self.handle_message(event)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._ws or self._ws.closed:
            return SendResult(success=False, error="IRIS websocket is not connected")
        try:
            message_id = str(uuid.uuid4())
            payload = {
                "type": "reply",
                "sessionId": chat_id,
                "messageId": message_id,
                **({"replyTo": reply_to, "requestId": reply_to} if reply_to else {}),
                "content": [{"type": "text", "text": content}],
            }
            async with self._send_lock:
                await self._ws.send_json(payload)
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            logger.error("[%s] Failed to send text reply: %s", self.name, e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
    ) -> SendResult:
        """Progressive stream push for IRIS when enabled."""
        if not self._streaming_push_enabled:
            return SendResult(success=False, error="IRIS streaming push is disabled")
        if not self._ws or self._ws.closed:
            return SendResult(success=False, error="IRIS websocket is not connected")
        try:
            payload = {
                "type": "reply_update",
                "sessionId": chat_id,
                "messageId": message_id,
                "requestId": message_id,
                "content": [{"type": "text", "text": content}],
            }
            async with self._send_lock:
                await self._ws.send_json(payload)
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            logger.error("[%s] Failed to stream edit: %s", self.name, e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._ws or self._ws.closed:
            return SendResult(success=False, error="IRIS websocket is not connected")
        try:
            parts = []
            if caption:
                parts.append({"type": "text", "text": caption})
            parts.append({"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}})
            message_id = str(uuid.uuid4())
            async with self._send_lock:
                await self._ws.send_json(
                    {
                        "type": "reply",
                        "sessionId": chat_id,
                        "messageId": message_id,
                        **({"replyTo": reply_to, "requestId": reply_to} if reply_to else {}),
                        "content": parts,
                    }
                )
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            logger.error("[%s] Failed to send image URL: %s", self.name, e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self._ws or self._ws.closed:
            return SendResult(success=False, error="IRIS websocket is not connected")
        try:
            p = Path(image_path)
            if not p.exists():
                return SendResult(success=False, error=f"Image file not found: {image_path}")
            mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
            data_url = f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"
            parts = []
            if caption:
                parts.append({"type": "text", "text": caption})
            parts.append({"type": "image_url", "image_url": {"url": data_url, "detail": p.name}})
            message_id = str(uuid.uuid4())
            async with self._send_lock:
                await self._ws.send_json(
                    {
                        "type": "reply",
                        "sessionId": chat_id,
                        "messageId": message_id,
                        **({"replyTo": reply_to, "requestId": reply_to} if reply_to else {}),
                        "content": parts,
                    }
                )
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            logger.error("[%s] Failed to send image file: %s", self.name, e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}
