"""LLM client used for synthetic query generation and answer synthesis."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import httpx

from geolytics.config import Settings, get_settings


class LLMClient(ABC):
    """Minimal text-in / text-out interface."""

    name: str = "base"

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str: ...

    def complete_json(
        self, prompt: str, system: str | None = None, temperature: float = 0.0
    ) -> Any:
        """Complete and parse JSON, tolerating a fenced code block around it."""
        raw = self.complete(prompt, system=system, temperature=temperature).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        # Small models sometimes prepend prose; salvage the outermost JSON value.
        start = min((i for i in (raw.find("["), raw.find("{")) if i != -1), default=-1)
        if start > 0:
            raw = raw[start:]
        return json.loads(raw)

    def describe(self) -> dict[str, Any]:
        return {"llm": self.name}


class OllamaClient(LLMClient):
    """Local model served by Ollama.

    Caveat to carry into the report: a local 7-8B model is adequate for
    generating candidate questions, but is a noisy judge of faithfulness and
    answer relevance. Where this client is used as a judge, validate it
    against a human-labelled sample and report the agreement rather than
    presenting judge scores as ground truth.
    """

    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: float = 120.0,
    ) -> None:
        self.name = f"ollama:{model}"
        self.url = url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout)

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        response = self._client.post(f"{self.url}/api/generate", json=payload)
        response.raise_for_status()
        return str(response.json().get("response", ""))

    def describe(self) -> dict[str, Any]:
        return {"llm": "ollama", "model": self.model}

    def close(self) -> None:
        self._client.close()


class EchoLLMClient(LLMClient):
    """Offline stub for tests. Returns a canned response, never a real one."""

    name = "echo"

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or []
        self.calls: list[str] = []
        self._i = 0

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        self.calls.append(prompt)
        if not self.responses:
            return ""
        out = self.responses[self._i % len(self.responses)]
        self._i += 1
        return out


def build_llm(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    if settings.llm_backend == "ollama":
        return OllamaClient(url=settings.ollama_url, model=settings.ollama_model)
    return EchoLLMClient()
