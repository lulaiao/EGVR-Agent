"""Minimal client for an external asynchronous tool service."""

from __future__ import annotations

import os
import time
from typing import Any

import requests


class ToolServerError(RuntimeError):
    """Raised when an external tool service cannot be contacted."""


class ToolServerClient:
    """Submit tool jobs and poll their status through a small HTTP contract."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        submit_timeout_sec: float = 10.0,
        poll_timeout_sec: float = 10.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("EGVR_TOOL_SERVER_URL")
            or "http://127.0.0.1:8001"
        ).rstrip("/")
        self.submit_timeout_sec = submit_timeout_sec
        self.poll_timeout_sec = poll_timeout_sec
        self.session = requests.Session()
        self.session.trust_env = False

    def health(self) -> dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=self.poll_timeout_sec,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise ToolServerError(f"Tool service health check failed: {exc}") from exc

    def run(
        self,
        tool: str,
        payload: dict[str, Any],
        *,
        action: str = "default",
        timeout_sec: float = 300.0,
    ) -> dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.base_url}/run/{tool}/{action}",
                json=payload,
                timeout=self.submit_timeout_sec,
            )
            response.raise_for_status()
            submitted = response.json()
        except requests.RequestException as exc:
            return {"success": False, "error": f"Tool submission failed: {exc}"}

        job_id = submitted.get("job_id")
        if not job_id:
            return {
                "success": False,
                "error": submitted.get("error") or "Tool service returned no job_id.",
            }

        deadline = time.monotonic() + timeout_sec
        interval = 0.5
        while time.monotonic() < deadline:
            try:
                response = self.session.get(
                    f"{self.base_url}/job/{job_id}",
                    timeout=self.poll_timeout_sec,
                )
                response.raise_for_status()
                status = response.json()
            except requests.RequestException as exc:
                return {"success": False, "error": f"Tool polling failed: {exc}"}

            state = status.get("status")
            if state == "running":
                time.sleep(interval)
                interval = min(interval * 1.5, 10.0)
                continue
            if state == "failed":
                return {
                    "success": False,
                    "error": f"Tool execution failed: {status.get('data')}",
                }
            if state == "finished":
                result = status.get("data")
                if isinstance(result, dict):
                    return result
                return {"success": False, "error": "Tool returned a non-object result."}
            return {"success": False, "error": f"Unknown job status: {state!r}"}

        return {"success": False, "error": f"Tool timed out after {timeout_sec:.0f}s."}


def run_external_tool(
    tool: str,
    payload: dict[str, Any],
    *,
    action: str = "default",
    timeout_sec: float = 300.0,
) -> dict[str, Any]:
    """Run one tool using environment-configured external infrastructure."""

    return ToolServerClient().run(
        tool,
        payload,
        action=action,
        timeout_sec=timeout_sec,
    )
