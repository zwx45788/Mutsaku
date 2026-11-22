"""Qwen OpenAI-compatible client with SHA-256 ReplayLedger and strict JSON output."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_BASE_URL = "https://maas.qianwenaiapi.com/compatible-mode/v1"
DEFAULT_LEDGER_PATH = Path("evoalign_os/outputs/replay_ledger.jsonl")
# Secrets live OUTSIDE the project tree on a Linux-native filesystem with 0600 permissions.
# Lookup order: environment variables -> $QWEN_ENV_FILE -> ~/.config/evoalign_os/qwen.env.
# (The Windows-mounted project drive cannot enforce POSIX permissions, so no key file is kept there.)
DEFAULT_SECRET_FILE = Path(os.path.expanduser("~/.config/evoalign_os/qwen.env"))
DEFAULT_ENV_LOCAL = Path(os.environ.get("QWEN_ENV_FILE", str(DEFAULT_SECRET_FILE)))


def _load_env_local(env_path: Path = DEFAULT_ENV_LOCAL) -> Dict[str, str]:
    vals: Dict[str, str] = {}
    if env_path.exists():
        mode = env_path.stat().st_mode & 0o777
        if mode & 0o077:
            raise PermissionError(f"Refusing to read secret file {env_path} with permissive mode {oct(mode)}; run chmod 600")
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip()
    return vals


def _extract_json_block(text: str) -> Any:
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fenced code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    # Outermost {...} or [...]
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        i = text.find(open_ch)
        j = text.rfind(close_ch)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except Exception:
                pass
    raise ValueError(f"Could not parse valid JSON from LLM response: {text[:300]}")


class ReplayLedger:
    """Append-only, SHA-256 keyed cache & audit ledger (stores zero secrets)."""

    def __init__(self, path: Path = DEFAULT_LEDGER_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if "request_hash" in rec:
                            self._cache[rec["request_hash"]] = rec
                    except Exception:
                        continue

    @staticmethod
    def hash_request(model: str, messages: List[Dict[str, str]], temperature: float, json_mode: bool) -> str:
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": round(float(temperature), 4),
                "json_mode": bool(json_mode),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, req_hash: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(req_hash)

    def put(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._cache[record["request_hash"]] = record
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def stats(self) -> Dict[str, Any]:
        total_prompt = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in self._cache.values())
        total_comp = sum(r.get("usage", {}).get("completion_tokens", 0) for r in self._cache.values())
        return {
            "entries": len(self._cache),
            "prompt_tokens": total_prompt,
            "completion_tokens": total_comp,
            "total_tokens": total_prompt + total_comp,
        }


class QwenClient:
    """Dual-mode Qwen client: online API with automatic ReplayLedger caching & offline replay."""

    def __init__(
        self,
        model: str = "qwen3.8-flash",
        base_url: Optional[str] = None,
        ledger_path: Path = DEFAULT_LEDGER_PATH,
        offline_only: bool = False,
        timeout: float = 90.0,
    ):
        env_local = _load_env_local()
        self.api_key = os.environ.get("QWEN_API_KEY") or env_local.get("QWEN_API_KEY", "")
        self.base_url = (
            base_url
            or os.environ.get("QWEN_BASE_URL")
            or env_local.get("QWEN_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model
        self.offline_only = offline_only
        self.timeout = timeout
        self.ledger = ReplayLedger(ledger_path)

    def chat(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
        tag: str = "general",
        use_cache: bool = True,
        retries: int = 3,
        enable_thinking: Optional[bool] = None,
    ) -> Dict[str, Any]:
        mdl = model or self.model
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        req_hash = self.ledger.hash_request(mdl, messages, temperature, json_mode)
        if use_cache:
            cached = self.ledger.get(req_hash)
            if cached is not None:
                return {**cached, "cache_hit": True}

        if self.offline_only:
            raise RuntimeError(f"Offline replay miss for tag={tag}, hash={req_hash[:12]}")
        if not self.api_key:
            raise RuntimeError("QWEN_API_KEY not found in environment, $QWEN_ENV_FILE or ~/.config/evoalign_os/qwen.env (mode 600)")

        body: Dict[str, Any] = {
            "model": mdl,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if enable_thinking is not None:
            body["enable_thinking"] = bool(enable_thinking)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Optional[Exception] = None
        for attempt in range(retries):
            t0 = time.perf_counter()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=body)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    rec = {
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "tag": tag,
                        "request_hash": req_hash,
                        "model": mdl,
                        "resolved_model": data.get("model", mdl),
                        "temperature": temperature,
                        "json_mode": json_mode,
                        "prompt_preview": user[:300],
                        "content": content,
                        "usage": usage,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    }
                    self.ledger.put(rec)
                    return {**rec, "cache_hit": False}
                # Fallback if json_mode unsupported on a specific model
                if resp.status_code == 400 and json_mode and "response_format" in body:
                    body.pop("response_format", None)
                    continue
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            except Exception as e:
                last_err = e
            time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"Qwen API call failed after {retries} retries: {last_err}")

    def chat_json(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        tag: str = "json_task",
        use_cache: bool = True,
        enable_thinking: Optional[bool] = None,
    ) -> Dict[str, Any]:
        rec = self.chat(
            system=system,
            user=user,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            tag=tag,
            use_cache=use_cache,
            enable_thinking=enable_thinking,
        )
        try:
            parsed = _extract_json_block(rec["content"])
        except ValueError:
            # Retry once uncached with higher max_tokens in case of truncated output
            rec = self.chat(
                system=system,
                user=user,
                model=model,
                temperature=temperature,
                max_tokens=max(max_tokens * 2, 4096),
                json_mode=True,
                tag=tag + "_retry",
                use_cache=False,
                enable_thinking=enable_thinking,
            )
            parsed = _extract_json_block(rec["content"])
        return {
            "data": parsed,
            "meta": {
                "tag": tag,
                "model": rec.get("resolved_model"),
                "request_hash": rec.get("request_hash"),
                "usage": rec.get("usage", {}),
                "cache_hit": rec.get("cache_hit", False),
                "latency_ms": rec.get("latency_ms", 0.0),
            },
        }
