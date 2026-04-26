from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import settings
from .query_frame import LegalIntentRegistry, QueryFrame


FORBIDDEN_OUTPUT_KEYS = {
    "answer",
    "short_answer",
    "citations",
    "citation",
    "unit_id",
    "legal_unit_id",
    "article_number",
    "article",
    "law_id",
    "source_url",
    "conclusion",
}

SYSTEM_MESSAGE = (
    "You decompose Romanian legal questions for retrieval. You do not answer legal "
    "questions. You do not cite laws. You do not invent article numbers, "
    "legal_unit_id, law_id, or source_url. Return strict JSON only."
)

MIN_CONFIDENCE = 0.60
MAX_RETRIEVAL_QUERIES = 6
MAX_QUERY_LENGTH = 240

_UNSAFE_TEXT_PATTERNS = (
    re.compile(r"\bart(?:icol|\.?)\s*\d+\b", re.IGNORECASE),
    re.compile(r"\b(?:unit_id|legal_unit_id|article_number|law_id|source_url)\b", re.IGNORECASE),
    re.compile(r"\bro\.codul", re.IGNORECASE),
    re.compile(r"\bcodul\s+muncii\b", re.IGNORECASE),
)


class UnsafeLLMQueryDecomposition(ValueError):
    def __init__(self, reason: str, forbidden_keys: Sequence[str] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.forbidden_keys = list(forbidden_keys or [])


class LLMQueryDecomposition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    domain: str | None = None
    intents: list[str] = Field(default_factory=list)
    meta_intents: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)
    qualifiers: list[str] = Field(default_factory=list)
    retrieval_queries: list[str] = Field(default_factory=list)
    required_evidence_concepts: list[str] = Field(default_factory=list)
    ambiguity_flags: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def reject_forbidden_keys(cls, data: Any) -> Any:
        forbidden_keys = forbidden_keys_detected(data)
        if forbidden_keys:
            raise ValueError(
                "forbidden_keys_detected:" + ",".join(forbidden_keys)
            )
        return data

    @model_validator(mode="after")
    def normalize_output(self) -> "LLMQueryDecomposition":
        self.domain = _bounded_text(self.domain, max_length=80)
        self.intents = _bounded_string_list(self.intents, limit=16)
        self.meta_intents = _bounded_string_list(self.meta_intents, limit=16)
        self.targets = _bounded_string_list(self.targets, limit=16)
        self.actors = _bounded_string_list(self.actors, limit=16)
        self.qualifiers = _bounded_string_list(self.qualifiers, limit=16)
        self.retrieval_queries = _bounded_string_list(
            self.retrieval_queries,
            limit=MAX_RETRIEVAL_QUERIES,
            max_length=MAX_QUERY_LENGTH,
        )
        self.required_evidence_concepts = _bounded_string_list(
            self.required_evidence_concepts,
            limit=16,
        )
        self.ambiguity_flags = _bounded_string_list(self.ambiguity_flags, limit=16)
        self.confidence = _clamp01(self.confidence)
        unsafe_field = unsafe_text_field(self.model_dump(mode="json"))
        if unsafe_field:
            raise ValueError(f"unsafe_text_detected:{unsafe_field}")
        return self


@dataclass
class LLMQueryDecomposerResult:
    decomposition: LLMQueryDecomposition | None = None
    debug: dict[str, Any] = field(default_factory=dict)


class LLMQueryDecomposer:
    def __init__(
        self,
        *,
        enabled: bool | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._enabled = enabled
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self.transport = transport

    @property
    def enabled(self) -> bool:
        if self._enabled is not None:
            return self._enabled
        return bool(settings.enable_llm_query_decomposer)

    @property
    def base_url(self) -> str | None:
        return self._base_url if self._base_url is not None else settings.llm_query_decomposer_base_url

    @property
    def api_key(self) -> str | None:
        return self._api_key if self._api_key is not None else settings.llm_query_decomposer_api_key

    @property
    def model(self) -> str | None:
        return self._model if self._model is not None else settings.llm_query_decomposer_model

    @property
    def timeout_seconds(self) -> float:
        if self._timeout_seconds is not None:
            return self._timeout_seconds
        return float(settings.llm_query_decomposer_timeout_seconds or 5.0)

    async def decompose(
        self,
        *,
        question: str,
        deterministic_query_frame: QueryFrame,
        known_intents: Sequence[str],
        allowed_domains: Sequence[str],
    ) -> LLMQueryDecomposerResult:
        debug = self._base_debug()
        if not self.enabled:
            return LLMQueryDecomposerResult(debug=debug)
        if not self.base_url or not self.model:
            debug["fallback_reason"] = "not_configured"
            return LLMQueryDecomposerResult(debug=debug)

        debug["attempted"] = True
        started_at = time.perf_counter()
        try:
            response_payload = await self._post_chat_completion(
                self._request_payload(
                    question=question,
                    deterministic_query_frame=deterministic_query_frame,
                    known_intents=known_intents,
                    allowed_domains=allowed_domains,
                )
            )
            raw_decomposition = self._extract_content_json(response_payload)
            forbidden_keys = forbidden_keys_detected(raw_decomposition)
            if forbidden_keys:
                raise UnsafeLLMQueryDecomposition(
                    "forbidden_keys",
                    forbidden_keys=forbidden_keys,
                )
            decomposition = LLMQueryDecomposition.model_validate(raw_decomposition)
        except httpx.TimeoutException:
            debug["fallback_reason"] = "timeout"
            return LLMQueryDecomposerResult(debug=self._with_latency(debug, started_at))
        except json.JSONDecodeError:
            debug["fallback_reason"] = "invalid_json"
            return LLMQueryDecomposerResult(debug=self._with_latency(debug, started_at))
        except UnsafeLLMQueryDecomposition as exc:
            debug["fallback_reason"] = exc.reason
            debug["forbidden_keys_detected"] = exc.forbidden_keys
            return LLMQueryDecomposerResult(debug=self._with_latency(debug, started_at))
        except Exception as exc:
            reason = "invalid_output"
            text = str(exc)
            if "forbidden_keys_detected:" in text:
                reason = "forbidden_keys"
                debug["forbidden_keys_detected"] = _keys_from_validation_message(text)
            elif "unsafe_text_detected:" in text:
                reason = "unsafe_output"
            debug["fallback_reason"] = reason
            return LLMQueryDecomposerResult(debug=self._with_latency(debug, started_at))

        debug["latency_ms"] = _elapsed_ms(started_at)
        debug["succeeded"] = decomposition.confidence >= MIN_CONFIDENCE
        if decomposition.confidence < MIN_CONFIDENCE:
            debug["fallback_reason"] = "low_confidence"
        return LLMQueryDecomposerResult(decomposition=decomposition, debug=debug)

    async def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            response = await client.post(
                self._endpoint_url(),
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    def _endpoint_url(self) -> str:
        base_url = (self.base_url or "").strip().rstrip("/")
        if not base_url:
            raise RuntimeError("LLM query decomposer endpoint is not configured")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def _request_payload(
        self,
        *,
        question: str,
        deterministic_query_frame: QueryFrame,
        known_intents: Sequence[str],
        allowed_domains: Sequence[str],
    ) -> dict[str, Any]:
        user_payload = {
            "question": question,
            "deterministic_query_frame": deterministic_query_frame.model_dump(
                mode="json"
            ),
            "known_intents": list(known_intents),
            "allowed_domains": list(allowed_domains),
            "instructions": [
                "Produce retrieval-oriented Romanian paraphrases and concept labels.",
                "Do not answer the legal question.",
                "Do not return citations, article numbers, unit IDs, law IDs, or source URLs.",
                "Return only the JSON object matching the requested schema.",
            ],
            "output_schema": {
                "domain": "string|null",
                "intents": ["known intent ids only"],
                "meta_intents": ["string"],
                "targets": ["string"],
                "actors": ["string"],
                "qualifiers": ["string"],
                "retrieval_queries": ["Romanian retrieval phrase"],
                "required_evidence_concepts": ["string"],
                "ambiguity_flags": ["string"],
                "confidence": "number 0..1",
            },
        }
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

    def _extract_content_json(self, response_payload: Mapping[str, Any]) -> dict[str, Any]:
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("missing_choices")
        message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise ValueError("missing_message_content")
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("decomposition_must_be_object")
        return payload

    def _base_debug(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "attempted": False,
            "succeeded": False,
            "fallback_reason": None,
            "forbidden_keys_detected": [],
            "latency_ms": None,
            "model": self.model,
        }

    def _with_latency(self, debug: dict[str, Any], started_at: float) -> dict[str, Any]:
        debug["latency_ms"] = _elapsed_ms(started_at)
        return debug


def merge_query_frames(
    deterministic: QueryFrame,
    llm: LLMQueryDecomposition,
    *,
    registry: LegalIntentRegistry | None = None,
) -> QueryFrame:
    if llm.confidence < MIN_CONFIDENCE:
        return deterministic.model_copy(
            update={
                "decomposition_source": "deterministic",
                "llm_confidence": llm.confidence,
            }
        )

    registry = registry or LegalIntentRegistry()
    valid_intents = {intent.id for intent in registry.all()}
    allowed_domains = {intent.domain for intent in registry.all() if intent.domain}
    llm_intents = [intent for intent in llm.intents if intent in valid_intents]

    deterministic_domain_has_priority = (
        deterministic.confidence >= 0.70 and deterministic.domain is not None
    )
    domain = deterministic.domain
    if not deterministic_domain_has_priority and llm.domain in allowed_domains:
        domain = llm.domain

    intents = _dedupe([*deterministic.intents, *llm_intents])
    retrieval_queries = _dedupe(
        [*deterministic.retrieval_queries, *llm.retrieval_queries]
    )
    required_evidence_concepts = _dedupe(
        [
            *deterministic.required_evidence_concepts,
            *llm.required_evidence_concepts,
        ]
    )
    used_llm = any(
        [
            retrieval_queries != deterministic.retrieval_queries,
            required_evidence_concepts != deterministic.required_evidence_concepts,
            intents != deterministic.intents,
            domain != deterministic.domain,
            any(value not in deterministic.meta_intents for value in llm.meta_intents),
            any(value not in deterministic.targets for value in llm.targets),
            any(value not in deterministic.actors for value in llm.actors),
            any(value not in deterministic.qualifiers for value in llm.qualifiers),
            any(value not in deterministic.ambiguity_flags for value in llm.ambiguity_flags),
        ]
    )
    decomposition_source = "deterministic"
    if used_llm:
        decomposition_source = (
            "llm"
            if not deterministic.domain and not deterministic.intents
            else "merged"
        )

    return deterministic.model_copy(
        update={
            "domain": domain,
            "intents": intents,
            "meta_intents": _dedupe([*deterministic.meta_intents, *llm.meta_intents]),
            "targets": _dedupe([*deterministic.targets, *llm.targets]),
            "actors": _dedupe([*deterministic.actors, *llm.actors]),
            "qualifiers": _dedupe([*deterministic.qualifiers, *llm.qualifiers]),
            "retrieval_queries": retrieval_queries,
            "required_evidence_concepts": required_evidence_concepts,
            "ambiguity_flags": _dedupe(
                [*deterministic.ambiguity_flags, *llm.ambiguity_flags]
            ),
            "decomposition_source": decomposition_source,
            "llm_confidence": llm.confidence,
        }
    )


def forbidden_keys_detected(payload: Any) -> list[str]:
    keys: set[str] = set()
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized_key = str(key).casefold()
            if normalized_key in FORBIDDEN_OUTPUT_KEYS:
                keys.add(normalized_key)
            keys.update(forbidden_keys_detected(value))
    elif isinstance(payload, list):
        for item in payload:
            keys.update(forbidden_keys_detected(item))
    return sorted(keys)


def unsafe_text_field(payload: Any, *, path: str = "") -> str | None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            found = unsafe_text_field(value, path=f"{path}.{key}" if path else str(key))
            if found:
                return found
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            found = unsafe_text_field(item, path=f"{path}[{index}]")
            if found:
                return found
    elif isinstance(payload, str):
        for pattern in _UNSAFE_TEXT_PATTERNS:
            if pattern.search(payload):
                return path or "<string>"
    return None


def _bounded_text(value: Any, *, max_length: int) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    if not text:
        return None
    return text[:max_length].strip()


def _bounded_string_list(
    values: Any,
    *,
    limit: int,
    max_length: int = MAX_QUERY_LENGTH,
) -> list[str]:
    if not isinstance(values, list):
        return []
    bounded: list[str] = []
    for value in values:
        text = _bounded_text(value, max_length=max_length)
        if not text or text in bounded:
            continue
        bounded.append(text)
        if len(bounded) >= limit:
            break
    return bounded


def _dedupe(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _elapsed_ms(started_at: float) -> int:
    return round((time.perf_counter() - started_at) * 1000)


def _keys_from_validation_message(message: str) -> list[str]:
    _, _, suffix = message.partition("forbidden_keys_detected:")
    keys = []
    for token in re.split(r"[^a-zA-Z0-9_]+", suffix):
        if token.casefold() in FORBIDDEN_OUTPUT_KEYS:
            keys.append(token.casefold())
    return sorted(set(keys))
