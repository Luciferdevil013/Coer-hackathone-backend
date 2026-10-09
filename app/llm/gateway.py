"""Model gateway (TDD M1). One place for model choice, timeouts, refusal handling and cost logging.

Returns None on any failure so every caller can fall back to its deterministic path.
"""
from __future__ import annotations

import json
import logging
import time

from app.config import Settings, get_settings
from app.db import session_scope
from app.models import LLMCall

log = logging.getLogger("dataguardian.llm")


class LLMGateway:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self._client = None

    @property
    def enabled(self) -> bool:
        return self.s.llm_enabled and self.s.anthropic_key_present

    def status(self) -> dict:
        return {"enabled": self.enabled, "model": self.s.llm_model if self.enabled else None,
                "mode": "claude" if self.enabled else "deterministic"}

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(timeout=self.s.llm_timeout_s, max_retries=2)
        return self._client

    def _record(self, purpose: str, workspace_id: str | None, ok: bool, t0: float, resp=None, error: str = "") -> None:
        try:
            usage = getattr(resp, "usage", None)
            with session_scope() as db:
                db.add(LLMCall(
                    workspace_id=workspace_id, purpose=purpose, model=self.s.llm_model, ok=ok,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    input_tokens=getattr(usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(usage, "output_tokens", 0) or 0,
                    error=error[:500],
                ))
        except Exception:  # observability must never break the request
            log.exception("failed to record llm call")

    def _call(self, *, purpose: str, system: str, user: str, max_tokens: int, fmt: dict | None,
              workspace_id: str | None) -> str | None:
        if not self.enabled:
            return None
        import anthropic

        t0 = time.perf_counter()
        output_config: dict = {"effort": self.s.llm_effort}
        if fmt:
            output_config["format"] = fmt
        try:
            resp = self._get_client().beta.messages.create(
                model=self.s.llm_model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config=output_config,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as e:
            self._record(purpose, workspace_id, False, t0, error=f"rate limited: {e.message}")
            return None
        except anthropic.APIStatusError as e:
            self._record(purpose, workspace_id, False, t0, error=f"{e.status_code}: {e.message}")
            return None
        except anthropic.APIConnectionError as e:
            self._record(purpose, workspace_id, False, t0, error=f"connection: {e}")
            return None
        if resp.stop_reason in ("refusal", "max_tokens"):
            self._record(purpose, workspace_id, False, t0, resp, error=f"stop_reason={resp.stop_reason}")
            return None
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        self._record(purpose, workspace_id, bool(text), t0, resp)
        return text or None

    def structured(self, *, purpose: str, system: str, user: str, schema: dict,
                   workspace_id: str | None = None, max_tokens: int = 8000) -> dict | None:
        text = self._call(purpose=purpose, system=system, user=user, max_tokens=max_tokens,
                          fmt={"type": "json_schema", "schema": schema}, workspace_id=workspace_id)
        if text is None:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.warning("structured output was not valid JSON for %s", purpose)
            return None

    def text(self, *, purpose: str, system: str, user: str, workspace_id: str | None = None,
             max_tokens: int = 2000) -> str | None:
        return self._call(purpose=purpose, system=system, user=user, max_tokens=max_tokens, fmt=None,
                          workspace_id=workspace_id)


_gateway: LLMGateway | None = None


def get_gateway() -> LLMGateway:
    global _gateway
    if _gateway is None:
        _gateway = LLMGateway()
    return _gateway
