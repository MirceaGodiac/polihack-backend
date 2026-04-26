from __future__ import annotations

import inspect
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas.retrieval import (
    RawExactCitation,
    RawRetrievalRequest,
    RawRetrievalResponse,
    RetrievalCandidate,
)
from .query_frame import LegalIntentRegistry, QueryFrameBuilder
from .retrieval_scoring import ScoreBreakdown, reciprocal_rank_fusion, weighted_retrieval_score


DEFAULT_METHOD_LIMIT_MULTIPLIER = 3
FTS_METHOD = "fts"
EXACT_METHOD = "exact_citation"
DENSE_METHOD = "dense_optional"
INTENT_GOVERNING_RULE_METHOD = "intent_governing_rule_lookup"


class RawRetrievalStore(Protocol):
    async def exact_citation_lookup(
        self,
        citations: Sequence[dict[str, Any]],
        *,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        ...

    async def lexical_search(
        self,
        question: str,
        *,
        filters: Mapping[str, Any],
        limit: int,
        query_frame: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        ...

    async def dense_search(
        self,
        query_embedding: Sequence[float],
        *,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        ...

    async def intent_governing_rule_lookup(
        self,
        *,
        intent: str,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        ...


class PostgresRawRetrievalStore:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self._unit_columns: set[str] | None = None
        self.last_lexical_debug: dict[str, Any] = {}

    async def rollback_after_error(self) -> None:
        await self.session.rollback()

    async def exact_citation_lookup(
        self,
        citations: Sequence[dict[str, Any]],
        *,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        if not citations:
            return []
        available_columns = await self._get_unit_columns()
        select_columns = _unit_select_columns_sql(available_columns, alias="u")
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for citation in citations:
            clauses, params = _sql_filters(
                filters,
                alias="u",
                available_columns=available_columns,
            )
            law_id = _text_value(citation.get("law_id"))
            article_number = _text_value(citation.get("article_number"))
            if not law_id or not article_number:
                continue
            clauses.extend(
                [
                    _sql_text_equals(available_columns, "law_id", "law_id", alias="u"),
                    _sql_text_equals(
                        available_columns,
                        "article_number",
                        "article_number",
                        alias="u",
                    ),
                ]
            )
            params.update({"law_id": law_id, "article_number": article_number})
            for field_name in ("paragraph_number", "letter_number", "point_number"):
                value = _text_value(citation.get(field_name))
                if value:
                    clauses.append(
                        _sql_text_equals(
                            available_columns,
                            field_name,
                            field_name,
                            alias="u",
                        )
                    )
                    params[field_name] = value
            params["limit"] = max(1, limit - len(rows))
            order_by = _unit_order_by_sql(available_columns, alias="u")
            async with self.session.begin_nested():
                result = await self.session.execute(
                    text(
                        f"""
                        SELECT {select_columns}
                        FROM legal_units u
                        WHERE {' AND '.join(clauses)}
                        ORDER BY {order_by}
                        LIMIT :limit
                        """
                    ),
                    params,
                )
            for row in result.mappings().all():
                unit = _unit_from_row(row)
                if unit["id"] not in seen:
                    rows.append(unit)
                    seen.add(unit["id"])
            if len(rows) >= limit:
                break
        return rows

    async def lexical_search(
        self,
        question: str,
        *,
        filters: Mapping[str, Any],
        limit: int,
        query_frame: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        available_columns = await self._get_unit_columns()
        lexical_query = _build_lexical_query(
            question,
            filters,
            query_frame=query_frame,
        )
        self.last_lexical_debug = _lexical_debug_payload(
            lexical_query,
            fallback_used=False,
        )
        clauses, params = _sql_filters(
            filters,
            alias="u",
            available_columns=available_columns,
        )
        select_columns = _unit_select_columns_sql(available_columns, alias="u")
        search_document = _unit_search_text_sql(available_columns, alias="u")
        params.update({"question": question, "limit": limit})
        try:
            rows = await self._strict_fts_search(
                select_columns=select_columns,
                search_document=search_document,
                clauses=clauses,
                params=params,
            )
            if rows:
                return rows
        except Exception:
            pass

        self.last_lexical_debug = _lexical_debug_payload(
            lexical_query,
            fallback_used=True,
        )
        return await self._lexical_ilike_fallback(
            question,
            filters=filters,
            limit=limit,
            available_columns=available_columns,
            terms=lexical_query.expanded_terms,
            query_frame=query_frame,
            lexical_query=lexical_query,
        )

    async def _strict_fts_search(
        self,
        *,
        select_columns: str,
        search_document: str,
        clauses: Sequence[str],
        params: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        async with self.session.begin_nested():
            result = await self.session.execute(
                text(
                    f"""
                    WITH q AS (
                        SELECT websearch_to_tsquery('simple', :question) AS query
                    )
                    SELECT
                        {select_columns},
                        ts_rank_cd(
                            to_tsvector('simple', {search_document}),
                            q.query
                        ) AS bm25_score
                    FROM legal_units u, q
                    WHERE q.query @@ to_tsvector('simple', {search_document})
                      AND {' AND '.join(clauses)}
                    ORDER BY bm25_score DESC, u.id
                    LIMIT :limit
                    """
                ),
                dict(params),
            )
        return [_unit_from_row(row) for row in result.mappings().all()]

    async def _lexical_ilike_fallback(
        self,
        question: str,
        *,
        filters: Mapping[str, Any],
        limit: int,
        available_columns: set[str] | None = None,
        terms: Sequence[str] | None = None,
        query_frame: Mapping[str, Any] | None = None,
        lexical_query: _LexicalQuery | None = None,
    ) -> list[dict[str, Any]]:
        lexical_query = lexical_query or _build_lexical_query(
            question,
            filters,
            query_frame=query_frame,
        )
        base_terms = terms if terms is not None else lexical_query.expanded_terms
        search_terms = _fallback_search_terms(
            base_terms,
            lexical_query.fallback_intent,
        )
        weighted_terms = _weighted_fallback_terms(search_terms)
        if not weighted_terms:
            return []
        if available_columns is None:
            available_columns = await self._get_unit_columns()
        clauses, params = _sql_filters(
            filters,
            alias="u",
            available_columns=available_columns,
        )
        select_columns = _unit_select_columns_sql(available_columns, alias="u")
        search_document = _unit_search_text_sql(available_columns, alias="u")
        if search_document == "''":
            return []
        match_clauses: list[str] = []
        score_parts: list[str] = []
        phrase_parts: list[str] = []
        for index, weighted_term in enumerate(weighted_terms):
            param_name = f"term_{index}"
            params[param_name] = f"%{weighted_term.term}%"
            condition = f"{search_document} ILIKE :{param_name}"
            match_clauses.append(condition)
            score_parts.append(
                f"CASE WHEN {condition} THEN {weighted_term.weight:.3f} ELSE 0.0 END"
            )
            indicator = f"CASE WHEN {condition} THEN 1 ELSE 0 END"
            if weighted_term.is_phrase:
                phrase_parts.append(indicator)
        params["limit"] = limit
        total_weight = sum(weighted_term.weight for weighted_term in weighted_terms)
        weighted_score = " + ".join(score_parts)
        phrase_match_count = " + ".join(phrase_parts) if phrase_parts else "0"
        base_weighted_score = f"(({weighted_score}) / {total_weight:.3f})"
        if lexical_query.fallback_intent:
            group_sql = _intent_group_score_sql(
                lexical_query.fallback_intent,
                weighted_terms,
                condition_by_term={
                    weighted_term.term: (
                        f"{search_document} ILIKE :term_{index}"
                    )
                    for index, weighted_term in enumerate(weighted_terms)
                },
            )
            final_score = _intent_final_score_sql(group_sql)
            extra_select = f"""
                    ({group_sql["core_score"]}) AS core_score,
                    ({group_sql["target_score"]}) AS target_score,
                    ({group_sql["actor_score"]}) AS actor_score,
                    ({group_sql["qualifier_score"]}) AS qualifier_score,
                    ({group_sql["generic_score"]}) AS generic_score,
                    ({group_sql["distractor_score"]}) AS distractor_score,
                    ({group_sql["core_match_count"]}) AS central_match_count,
            """
            order_by = "bm25_score DESC, core_score DESC, phrase_match_count DESC, u.id"
        else:
            final_score = base_weighted_score
            extra_select = ""
            order_by = "bm25_score DESC, phrase_match_count DESC, u.id"
        result = await self.session.execute(
            text(
                f"""
                SELECT
                    {select_columns},
                    ({final_score}) AS bm25_score,
                    {extra_select}
                    ({phrase_match_count}) AS phrase_match_count
                FROM legal_units u
                WHERE ({' OR '.join(match_clauses)})
                  AND {' AND '.join(clauses)}
                ORDER BY {order_by}
                LIMIT :limit
                """
            ),
            params,
        )
        return [_unit_from_row(row) for row in result.mappings().all()]

    async def dense_search(
        self,
        query_embedding: Sequence[float],
        *,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        available_columns = await self._get_unit_columns()
        clauses, params = _sql_filters(
            filters,
            alias="u",
            available_columns=available_columns,
        )
        select_columns = _unit_select_columns_sql(available_columns, alias="u")
        params.update(
            {
                "embedding": _vector_literal(query_embedding),
                "limit": limit,
            }
        )
        async with self.session.begin_nested():
            result = await self.session.execute(
                text(
                    f"""
                    SELECT
                        {select_columns},
                        greatest(0.0, least(1.0, 1.0 - (e.embedding <=> :embedding::vector))) AS dense_score
                    FROM legal_embeddings e
                    JOIN legal_units u ON u.id = e.legal_unit_id
                    WHERE {' AND '.join(clauses)}
                    ORDER BY e.embedding <=> :embedding::vector, u.id
                    LIMIT :limit
                    """
                ),
                params,
            )
        return [_unit_from_row(row) for row in result.mappings().all()]

    async def intent_governing_rule_lookup(
        self,
        *,
        intent: str,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        if intent != "labor_contract_modification":
            return []
        units = getattr(self, "units", None)
        if units is not None and not hasattr(self, "session"):
            return _intent_governing_rule_rows_from_units(
                units,
                intent=intent,
                filters=filters,
                limit=limit,
            )
        available_columns = await self._get_unit_columns()
        clauses, params = _sql_filters(
            filters,
            alias="u",
            available_columns=available_columns,
        )
        select_columns = _unit_select_columns_sql(available_columns, alias="u")
        search_document = _unit_search_text_sql(available_columns, alias="u")
        if search_document == "''":
            return []
        normalized_search_document = _sql_normalized_text(search_document)
        sql_parts, sql_params = _labor_contract_modification_governing_rule_sql(
            normalized_search_document
        )
        params.update(sql_params)
        params["limit"] = limit
        async with self.session.begin_nested():
            result = await self.session.execute(
                text(
                    f"""
                    SELECT
                        {select_columns},
                        ({sql_parts["score"]}) AS intent_governing_rule_score,
                        ({sql_parts["role"]}) AS intent_governing_rule_role
                    FROM legal_units u
                    WHERE {' AND '.join(clauses)}
                      AND ({sql_parts["match"]})
                    ORDER BY intent_governing_rule_score DESC, u.id
                    LIMIT :limit
                    """
                ),
                params,
            )
        return [_unit_from_row(row) for row in result.mappings().all()]

    async def _get_unit_columns(self) -> set[str]:
        if self._unit_columns is None:
            async with self.session.begin_nested():
                result = await self.session.execute(
                    text(
                        """
                        SELECT a.attname AS column_name
                        FROM pg_attribute a
                        WHERE a.attrelid = 'legal_units'::regclass
                          AND a.attnum > 0
                          AND NOT a.attisdropped
                        """
                    )
                )
            self._unit_columns = {
                str(row["column_name"]) for row in result.mappings().all()
            }
        return self._unit_columns


class EmptyRawRetrievalStore:
    async def exact_citation_lookup(
        self,
        citations: Sequence[dict[str, Any]],
        *,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        return []

    async def lexical_search(
        self,
        question: str,
        *,
        filters: Mapping[str, Any],
        limit: int,
        query_frame: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return []

    async def dense_search(
        self,
        query_embedding: Sequence[float],
        *,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        return []


@dataclass
class _CandidateAccumulator:
    unit: dict[str, Any]
    bm25: float = 0.0
    dense: float = 0.0
    intent_governing_rule: float = 0.0
    exact_citation_boost: float = 0.0
    methods: set[str] = field(default_factory=set)
    rrf: float = 0.0


@dataclass(frozen=True)
class _FallbackIntent:
    name: str
    triggers: tuple[str, ...]
    core_phrases: tuple[str, ...]
    target_terms: tuple[str, ...]
    actor_terms: tuple[str, ...]
    qualifier_terms: tuple[str, ...]
    generic_terms: tuple[str, ...]
    distractor_terms: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class _LexicalQuery:
    lexical_terms: list[str]
    expanded_terms: list[str]
    fallback_intent: _FallbackIntent | None = None
    registry_expanded_terms: list[str] = field(default_factory=list)
    decomposition_retrieval_queries: list[str] = field(default_factory=list)
    decomposition_expanded_terms: list[str] = field(default_factory=list)
    decomposition_source: str | None = None
    query_frame_intents: list[str] = field(default_factory=list)
    query_frame_confidence: float | None = None
    fallback_intent_source: str | None = None


@dataclass(frozen=True)
class _WeightedLexicalTerm:
    term: str
    weight: float
    is_phrase: bool
    is_generic: bool
    is_central: bool
    is_salary_or_employer: bool


@dataclass(frozen=True)
class _IntentGoverningRuleMatch:
    matched: bool
    score: float = 0.0
    role: str | None = None
    matched_terms: tuple[str, ...] = ()


class RawRetriever:
    def __init__(
        self,
        store: RawRetrievalStore,
        *,
        initial_warnings: Sequence[str] | None = None,
    ) -> None:
        self.store = store
        self.initial_warnings = list(initial_warnings or [])

    async def retrieve(self, request: RawRetrievalRequest) -> RawRetrievalResponse:
        top_k = request.top_k
        method_limit = max(top_k, top_k * DEFAULT_METHOD_LIMIT_MULTIPLIER)
        filters = _normalized_filters(request)
        warnings = list(self.initial_warnings)
        retrieval_methods = [EXACT_METHOD, FTS_METHOD, DENSE_METHOD]
        rankings: dict[str, list[str]] = {}
        candidates: dict[str, _CandidateAccumulator] = {}
        governing_rule_intent = _intent_governing_rule_intent(request.query_frame)
        governing_rule_terms = (
            _intent_governing_rule_terms(governing_rule_intent)
            if governing_rule_intent
            else {}
        )

        citation_filters = _citation_filters(request)
        exact_rows = await self._call_store(
            self.store.exact_citation_lookup(
                citation_filters,
                filters=filters,
                limit=method_limit,
            ),
            warnings=warnings,
            warning_code="exact_citation_lookup_failed",
            debug=request.debug,
        )
        rankings[EXACT_METHOD] = [row["id"] for row in exact_rows]
        for row in exact_rows:
            accumulator = candidates.setdefault(
                row["id"],
                _CandidateAccumulator(unit=row),
            )
            accumulator.exact_citation_boost = 1.0
            accumulator.methods.add(EXACT_METHOD)

        fts_rows = await self._call_store(
            self._lexical_search(
                request.question,
                filters=filters,
                limit=method_limit,
                query_frame=request.query_frame,
            ),
            warnings=warnings,
            warning_code="fts_retrieval_failed",
            debug=request.debug,
        )
        max_bm25 = max([_float(row.get("bm25_score")) for row in fts_rows] or [0.0])
        rankings[FTS_METHOD] = [row["id"] for row in fts_rows]
        for row in fts_rows:
            accumulator = candidates.setdefault(
                row["id"],
                _CandidateAccumulator(unit=row),
            )
            raw_score = _float(row.get("bm25_score"))
            accumulator.bm25 = max(accumulator.bm25, _normalize_score(raw_score, max_bm25))
            accumulator.methods.add(FTS_METHOD)

        governing_rule_rows: list[dict[str, Any]] = []
        if governing_rule_intent:
            governing_rule_rows = await self._call_store(
                self._intent_governing_rule_lookup(
                    intent=governing_rule_intent,
                    filters=filters,
                    limit=method_limit,
                ),
                warnings=warnings,
                warning_code="intent_governing_rule_lookup_failed",
                debug=request.debug,
            )
            rankings[INTENT_GOVERNING_RULE_METHOD] = [
                row["id"] for row in governing_rule_rows
            ]
            if INTENT_GOVERNING_RULE_METHOD not in retrieval_methods:
                retrieval_methods.insert(2, INTENT_GOVERNING_RULE_METHOD)
        for row in governing_rule_rows:
            accumulator = candidates.setdefault(
                row["id"],
                _CandidateAccumulator(unit=row),
            )
            accumulator.intent_governing_rule = max(
                accumulator.intent_governing_rule,
                _clamp01(_float(row.get("intent_governing_rule_score"))),
            )
            accumulator.methods.add(INTENT_GOVERNING_RULE_METHOD)

        dense_rows: list[dict[str, Any]] = []
        if request.query_embedding:
            dense_rows = await self._call_store(
                self.store.dense_search(
                    request.query_embedding,
                    filters=filters,
                    limit=method_limit,
                ),
                warnings=warnings,
                warning_code="dense_retrieval_failed_or_unavailable",
                debug=request.debug,
            )
        else:
            warnings.append("dense_retrieval_skipped_no_query_embedding")
        rankings[DENSE_METHOD] = [row["id"] for row in dense_rows]
        for row in dense_rows:
            accumulator = candidates.setdefault(
                row["id"],
                _CandidateAccumulator(unit=row),
            )
            accumulator.dense = max(accumulator.dense, _clamp01(_float(row.get("dense_score"))))
            accumulator.methods.add(DENSE_METHOD)

        domain_rank = [
            unit_id
            for unit_id, candidate in sorted(candidates.items())
            if _domain_match(filters.get("legal_domain"), candidate.unit.get("legal_domain")) > 0.0
        ]
        if domain_rank:
            rankings["domain_filtered_search"] = domain_rank
        rrf_scores = reciprocal_rank_fusion(rankings)
        for unit_id, score in rrf_scores.items():
            if unit_id in candidates:
                candidates[unit_id].rrf = score

        dense_available = bool(dense_rows)
        ranked_candidates = [
            self._to_candidate(
                accumulator,
                filters=filters,
                question=request.question,
                query_frame=request.query_frame,
                dense_available=dense_available,
            )
            for accumulator in candidates.values()
        ]
        ranked_candidates.sort(
            key=lambda candidate: (
                candidate.retrieval_score,
                candidate.score_breakdown.get("exact_citation_boost", 0.0),
                candidate.score_breakdown.get("rrf", 0.0),
            ),
            reverse=True,
        )
        ranked_candidates = ranked_candidates[:top_k]
        for rank, candidate in enumerate(ranked_candidates, start=1):
            candidate.rank = rank

        debug_payload = None
        if request.debug:
            lexical_debug = _store_lexical_debug(
                self.store,
                question=request.question,
                filters=filters,
                query_frame=request.query_frame,
            )
            debug_payload = {
                "candidate_count_before_top_k": len(candidates),
                "candidate_count": len(ranked_candidates),
                "top_k": top_k,
                "filters": filters,
                "lexical_terms": lexical_debug["lexical_terms"],
                "expanded_terms": lexical_debug["expanded_terms"],
                "query_frame_intents": lexical_debug.get("query_frame_intents"),
                "query_frame_confidence": lexical_debug.get("query_frame_confidence"),
                "registry_expanded_terms": lexical_debug.get("registry_expanded_terms"),
                "intent_governing_rule_candidate_count": len(governing_rule_rows),
                "intent_governing_rule_terms": governing_rule_terms,
                "decomposition_retrieval_queries": lexical_debug.get(
                    "decomposition_retrieval_queries"
                ),
                "decomposition_expanded_terms": lexical_debug.get(
                    "decomposition_expanded_terms"
                ),
                "decomposition_source": lexical_debug.get("decomposition_source"),
                "fts_fallback_used": lexical_debug["fts_fallback_used"],
                "fallback_intent": lexical_debug.get("fallback_intent"),
                "fallback_intent_source": lexical_debug.get("fallback_intent_source"),
                "scoring_strategy": lexical_debug.get("scoring_strategy"),
                "group_scores": _debug_group_scores(
                    ranked_candidates,
                    lexical_debug.get("lexical_query"),
                ),
                "dense_available": dense_available,
                "rankings": rankings,
                "rrf_scores": rrf_scores,
                "warnings": warnings,
            }

        return RawRetrievalResponse(
            candidates=ranked_candidates,
            retrieval_methods=retrieval_methods,
            warnings=warnings,
            debug=debug_payload,
        )

    async def _lexical_search(
        self,
        question: str,
        *,
        filters: Mapping[str, Any],
        limit: int,
        query_frame: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        lexical_search = self.store.lexical_search
        parameters = inspect.signature(lexical_search).parameters
        if "query_frame" in parameters:
            return await lexical_search(
                question,
                filters=filters,
                limit=limit,
                query_frame=query_frame,
            )
        return await lexical_search(question, filters=filters, limit=limit)

    async def _intent_governing_rule_lookup(
        self,
        *,
        intent: str,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        units = getattr(self.store, "units", None)
        if units is not None and not hasattr(self.store, "session"):
            return _intent_governing_rule_rows_from_units(
                units,
                intent=intent,
                filters=filters,
                limit=limit,
            )
        lookup = getattr(self.store, "intent_governing_rule_lookup", None)
        if lookup is None:
            return []
        return await lookup(intent=intent, filters=filters, limit=limit)

    async def _call_store(
        self,
        store_call,
        *,
        warnings: list[str],
        warning_code: str,
        debug: bool,
    ) -> list[dict[str, Any]]:
        try:
            return await store_call
        except Exception as exc:
            warnings.append(_debug_warning(warning_code, exc) if debug else warning_code)
            rollback = getattr(self.store, "rollback_after_error", None)
            if rollback is not None:
                try:
                    await rollback()
                except Exception:
                    pass
            return []

    def _to_candidate(
        self,
        accumulator: _CandidateAccumulator,
        *,
        filters: Mapping[str, Any],
        question: str,
        query_frame: Mapping[str, Any] | None,
        dense_available: bool,
    ) -> RetrievalCandidate:
        unit = accumulator.unit
        breakdown = ScoreBreakdown(
            bm25=accumulator.bm25,
            dense=accumulator.dense,
            rrf=accumulator.rrf,
            domain_match=_domain_match(filters.get("legal_domain"), unit.get("legal_domain")),
            metadata_validity=_metadata_validity(unit),
            exact_citation_boost=accumulator.exact_citation_boost,
        )
        score = weighted_retrieval_score(breakdown, dense_available=dense_available)
        score = max(
            score,
            _intent_governing_rule_retrieval_score(
                accumulator.intent_governing_rule,
                domain_match=breakdown.domain_match,
                metadata_validity=breakdown.metadata_validity,
            ),
        )
        return RetrievalCandidate(
            unit_id=unit["id"],
            unit=unit,
            rank=0,
            retrieval_score=round(_clamp01(score), 6),
            score_breakdown={
                "bm25": round(breakdown.bm25, 6),
                "dense": round(breakdown.dense, 6),
                "domain_match": round(breakdown.domain_match, 6),
                "metadata_validity": round(breakdown.metadata_validity, 6),
                "exact_citation_boost": round(breakdown.exact_citation_boost, 6),
                "intent_governing_rule": round(
                    accumulator.intent_governing_rule,
                    6,
                ),
                "rrf": round(breakdown.rrf, 6),
            },
            matched_terms=_matched_terms(question, unit, query_frame=query_frame),
            why_retrieved=_why_retrieved(accumulator.methods),
        )


def _normalized_filters(request: RawRetrievalRequest) -> dict[str, Any]:
    filters = dict(request.retrieval_filters or {})
    filters.update(request.filters or {})
    filters.pop("exact_citation_filters", None)
    if filters.get("legal_domain"):
        filters["legal_domain"] = _canonical_domain(filters["legal_domain"])
    return filters


def _citation_filters(request: RawRetrievalRequest) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    for raw in request.retrieval_filters.get("exact_citation_filters", []) or []:
        if isinstance(raw, dict):
            filters.append(_canonical_citation_filter(raw))
    for citation in request.exact_citations:
        payload = (
            citation.model_dump(exclude_none=True)
            if isinstance(citation, RawExactCitation)
            else dict(citation)
        )
        filters.append(_canonical_citation_filter(payload))
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for item in filters:
        key = tuple(sorted((name, str(value)) for name, value in item.items() if value))
        if key and key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def _debug_warning(warning_code: str, exc: Exception) -> str:
    return f"{warning_code}:{type(exc).__name__}:{_sanitize_exception_message(exc)}"


def _sanitize_exception_message(exc: Exception) -> str:
    message = next((line.strip() for line in str(exc).splitlines() if line.strip()), "")
    if not message:
        message = repr(exc)
    for pattern, replacement in _WARNING_REDACTIONS:
        message = pattern.sub(replacement, message)
    if len(message) > _MAX_WARNING_MESSAGE_LENGTH:
        message = message[: _MAX_WARNING_MESSAGE_LENGTH - 3].rstrip() + "..."
    return message


def _canonical_citation_filter(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "law_id": _text_value(payload.get("law_id") or payload.get("law_id_hint")),
        "article_number": _text_value(payload.get("article_number") or payload.get("article")),
        "paragraph_number": _text_value(payload.get("paragraph_number") or payload.get("paragraph")),
        "letter_number": _text_value(payload.get("letter_number") or payload.get("letter")),
        "point_number": _text_value(payload.get("point_number") or payload.get("point")),
    }


def _sql_filters(
    filters: Mapping[str, Any],
    *,
    alias: str,
    available_columns: set[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    clauses = ["1 = 1"]
    params: dict[str, Any] = {}
    legal_domain = _text_value(filters.get("legal_domain"))
    if legal_domain:
        if _has_column(available_columns, "legal_domain"):
            clauses.append(f"lower({alias}.legal_domain::text) = lower(:legal_domain)")
            params["legal_domain"] = legal_domain
        else:
            clauses.append("1 = 0")
    status = _text_value(filters.get("status"))
    if status:
        if _has_column(available_columns, "status"):
            clauses.append(f"{alias}.status::text = :status")
            params["status"] = status
        else:
            clauses.append("1 = 0")
    law_id = _text_value(filters.get("law_id"))
    if law_id:
        if _has_column(available_columns, "law_id"):
            clauses.append(f"{alias}.law_id::text = :filter_law_id")
            params["filter_law_id"] = law_id
        else:
            clauses.append("1 = 0")
    return clauses, params


def _has_column(available_columns: set[str] | None, column: str) -> bool:
    return available_columns is None or column in available_columns


def _sql_text_equals(
    available_columns: set[str],
    column: str,
    param_name: str,
    *,
    alias: str,
) -> str:
    if column not in available_columns:
        return "1 = 0"
    return f"{alias}.{column}::text = :{param_name}"


def _unit_select_columns_sql(available_columns: set[str], *, alias: str) -> str:
    select_parts = []
    for column in _UNIT_COLUMNS:
        if column in available_columns:
            select_parts.append(f"{alias}.{column} AS {column}")
        else:
            select_parts.append(f"NULL AS {column}")
    return ", ".join(select_parts)


def _unit_search_text_sql(available_columns: set[str], *, alias: str) -> str:
    text_parts = [
        f"coalesce({alias}.{column}::text, '')"
        for column in ("raw_text", "normalized_text")
        if column in available_columns
    ]
    if not text_parts:
        return "''"
    return " || ' ' || ".join(text_parts)


def _unit_order_by_sql(available_columns: set[str], *, alias: str) -> str:
    order_parts = []
    if "status" in available_columns:
        order_parts.append(
            f"""
            CASE WHEN {alias}.status::text = 'active' THEN 0
                 WHEN {alias}.status::text = 'unknown' THEN 1
                 ELSE 2 END
            """
        )
    order_parts.append(f"{alias}.id")
    return ", ".join(order_parts)


def _unit_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    unit = {column: row.get(column) for column in _UNIT_COLUMNS}
    unit["legal_concepts"] = unit.get("legal_concepts") or []
    unit["hierarchy_path"] = unit.get("hierarchy_path") or []
    if "bm25_score" in row:
        unit["bm25_score"] = _float(row.get("bm25_score"))
    if "dense_score" in row:
        unit["dense_score"] = _float(row.get("dense_score"))
    if "intent_governing_rule_score" in row:
        unit["intent_governing_rule_score"] = _float(
            row.get("intent_governing_rule_score")
        )
    if "intent_governing_rule_role" in row:
        unit["intent_governing_rule_role"] = _text_value(
            row.get("intent_governing_rule_role")
        )
    return unit


def _store_lexical_debug(
    store: RawRetrievalStore,
    *,
    question: str,
    filters: Mapping[str, Any],
    query_frame: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    store_debug = getattr(store, "last_lexical_debug", None)
    if isinstance(store_debug, Mapping):
        lexical_terms = store_debug.get("lexical_terms")
        expanded_terms = store_debug.get("expanded_terms")
        fallback_used = store_debug.get("fts_fallback_used")
        if isinstance(lexical_terms, list) and isinstance(expanded_terms, list):
            lexical_query = _build_lexical_query(
                question,
                filters,
                query_frame=query_frame,
            )
            return {
                "lexical_terms": lexical_terms,
                "expanded_terms": expanded_terms,
                "query_frame_intents": store_debug.get("query_frame_intents") or [],
                "query_frame_confidence": store_debug.get("query_frame_confidence"),
                "registry_expanded_terms": store_debug.get("registry_expanded_terms")
                or [],
                "decomposition_retrieval_queries": store_debug.get(
                    "decomposition_retrieval_queries"
                )
                or lexical_query.decomposition_retrieval_queries,
                "decomposition_expanded_terms": store_debug.get(
                    "decomposition_expanded_terms"
                )
                or lexical_query.decomposition_expanded_terms,
                "decomposition_source": store_debug.get("decomposition_source")
                or lexical_query.decomposition_source,
                "fts_fallback_used": bool(fallback_used),
                "fallback_intent": store_debug.get("fallback_intent"),
                "fallback_intent_source": store_debug.get("fallback_intent_source"),
                "scoring_strategy": store_debug.get("scoring_strategy"),
                "lexical_query": lexical_query,
            }
    lexical_query = _build_lexical_query(
        question,
        filters,
        query_frame=query_frame,
    )
    payload = _lexical_debug_payload(lexical_query, fallback_used=False)
    payload["lexical_query"] = lexical_query
    return payload


def _lexical_debug_payload(
    lexical_query: _LexicalQuery,
    *,
    fallback_used: bool,
) -> dict[str, Any]:
    return {
        "lexical_terms": lexical_query.lexical_terms,
        "expanded_terms": lexical_query.expanded_terms,
        "query_frame_intents": lexical_query.query_frame_intents,
        "query_frame_confidence": lexical_query.query_frame_confidence,
        "registry_expanded_terms": lexical_query.registry_expanded_terms,
        "decomposition_retrieval_queries": lexical_query.decomposition_retrieval_queries,
        "decomposition_expanded_terms": lexical_query.decomposition_expanded_terms,
        "decomposition_source": lexical_query.decomposition_source,
        "fts_fallback_used": fallback_used,
        "fallback_intent": (
            lexical_query.fallback_intent.name if lexical_query.fallback_intent else None
        ),
        "fallback_intent_source": lexical_query.fallback_intent_source,
        "scoring_strategy": _lexical_scoring_strategy(lexical_query, fallback_used),
    }


def _build_lexical_query(
    question: str,
    filters: Mapping[str, Any] | None = None,
    *,
    query_frame: Mapping[str, Any] | None = None,
) -> _LexicalQuery:
    fallback_intent = _detect_fallback_intent(
        question,
        filters or {},
        query_frame=query_frame,
    )
    query_frame_intents = _query_frame_list(query_frame, "intents")
    query_frame_confidence = _query_frame_float(query_frame, "confidence")
    decomposition_retrieval_queries = _query_frame_retrieval_queries(query_frame)
    decomposition_source = _query_frame_decomposition_source(query_frame)
    expanded_terms = _expanded_query_terms(
        question,
        filters or {},
        query_frame=query_frame,
    )
    legacy_terms = _expanded_query_terms(question, filters or {}, query_frame=None)
    registry_expanded_terms = [
        term for term in expanded_terms if term not in set(legacy_terms)
    ]
    return _LexicalQuery(
        lexical_terms=_query_terms(question),
        expanded_terms=expanded_terms,
        fallback_intent=fallback_intent,
        registry_expanded_terms=registry_expanded_terms,
        decomposition_retrieval_queries=decomposition_retrieval_queries,
        decomposition_expanded_terms=[
            term
            for term in decomposition_retrieval_queries
            if term in set(expanded_terms)
        ],
        decomposition_source=decomposition_source,
        query_frame_intents=query_frame_intents,
        query_frame_confidence=query_frame_confidence,
        fallback_intent_source=fallback_intent.source if fallback_intent else None,
    )


def _lexical_scoring_strategy(
    lexical_query: _LexicalQuery,
    fallback_used: bool,
) -> str:
    if not fallback_used:
        return "strict_fts"
    if lexical_query.fallback_intent:
        if lexical_query.fallback_intent_source == "query_frame_registry":
            return "registry_intent_grouped_lexical_fallback"
        return "legacy_intent_grouped_lexical_fallback"
    return "weighted_lexical_fallback"


def _detect_fallback_intent(
    question: str,
    filters: Mapping[str, Any] | None = None,
    *,
    query_frame: Mapping[str, Any] | None = None,
) -> _FallbackIntent | None:
    registry_intent = _registry_fallback_intent(query_frame)
    if registry_intent:
        return registry_intent

    legal_domain = _canonical_domain(filters.get("legal_domain")) if filters else None
    if legal_domain != "munca":
        return None
    normalized_question = _normalize_text(question)
    terms = set(_query_terms(question))
    intent = LABOR_FALLBACK_INTENTS["labor_contract_modification"]
    if any(trigger in normalized_question for trigger in intent.triggers):
        return intent
    if {"act", "aditional"}.issubset(terms):
        return intent
    if terms & {"scada", "scade"} and terms & {"salariu", "salariul"}:
        return intent
    return None


def _expanded_query_terms(
    question: str,
    filters: Mapping[str, Any] | None = None,
    *,
    query_frame: Mapping[str, Any] | None = None,
) -> list[str]:
    lexical_terms = _query_terms(question)
    normalized_question = _normalize_text(question)
    legal_domain = _canonical_domain(filters.get("legal_domain")) if filters else None
    fallback_intent = _detect_fallback_intent(
        question,
        filters or {},
        query_frame=query_frame,
    )
    term_set = set(lexical_terms)
    expanded: list[str] = list(lexical_terms)

    def add(*values: str) -> None:
        for value in values:
            normalized = _normalize_text(value).strip()
            if normalized and normalized not in expanded:
                expanded.append(normalized)

    salary_query = bool(term_set & _SALARY_TERMS)
    act_aditional_query = (
        "act aditional" in normalized_question
        or {"act", "aditional"}.issubset(term_set)
    )
    employer_query = bool(term_set & {"angajator", "angajatorul"})
    salary_reduction_query = salary_query and bool(
        term_set & {"scada", "scade", "scad", "reducere", "reduca", "micsoreze"}
    )
    labor_context = legal_domain == "munca" or act_aditional_query or bool(
        term_set & _LABOR_CONTEXT_TERMS
    )

    if employer_query:
        add("angajator", "angajatorul")

    if salary_query and labor_context:
        add("salariu", "salariul", "salarizare")

    if act_aditional_query and labor_context:
        add(
            "modificare contract",
            "modificarea contractului",
            "modificarea contractului individual de munca",
            "modificare contract individual munca",
            "contractului individual de munca",
            "modificat",
            "modificare",
            "contract",
            "contractul",
            "contractului",
            "contract individual de munca",
            "contract individual munca",
            "contractul individual de munca",
            "acordul partilor",
            "acord parti",
            "acord partilor",
            "numai prin acordul partilor",
            "poate fi modificat",
            "modificat numai prin acordul partilor",
            "acord",
            "acordul",
            "parti",
            "partilor",
        )

    if salary_reduction_query and labor_context:
        add(
            "salariul",
            "salariu",
            "modificarea contractului",
            "modificare contract",
            "contract",
        )
    if fallback_intent:
        add(
            *fallback_intent.triggers,
            *fallback_intent.core_phrases,
            *fallback_intent.target_terms,
            *fallback_intent.actor_terms,
            *fallback_intent.qualifier_terms,
            *fallback_intent.generic_terms,
        )
    decomposition_retrieval_queries = _query_frame_retrieval_queries(query_frame)
    if decomposition_retrieval_queries:
        add(*decomposition_retrieval_queries)

    return _dedupe_terms(expanded)


def _registry_fallback_intent(
    query_frame: Mapping[str, Any] | None,
) -> _FallbackIntent | None:
    intent_ids = _query_frame_list(query_frame, "intents")
    if not intent_ids:
        return None
    registry = LegalIntentRegistry()
    builder = QueryFrameBuilder(registry=registry)
    frame_terms = _query_frame_list(query_frame, "normalized_terms")
    decomposition_retrieval_queries = _query_frame_retrieval_queries(query_frame)
    for intent_id in intent_ids:
        legal_intent = registry.get(intent_id)
        if legal_intent is None:
            continue
        core_aliases = [
            alias
            for concept in legal_intent.core_concepts
            for alias in builder._concept_terms(concept)
        ]
        target_aliases = [
            alias
            for concept in [
                *legal_intent.target_concepts,
                *_query_frame_list(query_frame, "targets"),
            ]
            for alias in builder._concept_terms(concept)
        ]
        actor_aliases = [
            alias
            for concept in [
                *legal_intent.actor_concepts,
                *_query_frame_list(query_frame, "actors"),
            ]
            for alias in builder._concept_terms(concept)
        ]
        qualifier_aliases = [
            alias
            for concept in [
                *legal_intent.qualifier_concepts,
                *_query_frame_list(query_frame, "qualifiers"),
            ]
            for alias in builder._concept_terms(concept)
        ]
        return _FallbackIntent(
            name=legal_intent.id,
            triggers=tuple(_dedupe_terms(legal_intent.triggers)),
            core_phrases=tuple(
                _dedupe_terms([*legal_intent.core_phrases, *core_aliases])
            ),
            target_terms=tuple(
                _dedupe_terms([*legal_intent.target_terms, *target_aliases])
            ),
            actor_terms=tuple(
                _dedupe_terms([*legal_intent.actor_terms, *actor_aliases])
            ),
            qualifier_terms=tuple(
                _dedupe_terms([*legal_intent.qualifier_terms, *qualifier_aliases])
            ),
            generic_terms=tuple(
                _dedupe_terms([*frame_terms, *decomposition_retrieval_queries])
            ),
            distractor_terms=tuple(_dedupe_terms(legal_intent.distractor_terms)),
            source="query_frame_registry",
        )
    return None


def _query_terms(question: str) -> list[str]:
    terms = []
    for raw in _repair_split_legal_terms(
        re.split(r"[^a-z0-9_]+", _normalize_text(question).replace("_", " "))
    ):
        token = raw.strip(".,;:!?()[]{}\"'")
        if len(token) >= 3 and token not in _STOP_WORDS and token not in terms:
            terms.append(token)
    return terms


def _repair_split_legal_terms(tokens: Sequence[str]) -> list[str]:
    repaired: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        next_token = tokens[index + 1] if index + 1 < len(tokens) else None
        if token == "adi" and next_token in {"ional", "ionala"}:
            repaired.append(f"adit{next_token}")
            index += 2
            continue
        repaired.append(token)
        index += 1
    return repaired


def _dedupe_terms(terms: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    for term in terms:
        normalized = _normalize_text(str(term)).strip()
        if normalized and normalized not in _STOP_WORDS and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _fallback_search_terms(
    terms: Sequence[str],
    intent: _FallbackIntent | None,
) -> list[str]:
    expanded = list(terms)
    if intent:
        expanded.extend(intent.triggers)
        expanded.extend(intent.core_phrases)
        expanded.extend(intent.target_terms)
        expanded.extend(intent.actor_terms)
        expanded.extend(intent.qualifier_terms)
        expanded.extend(intent.generic_terms)
        expanded.extend(intent.distractor_terms)
    return _dedupe_terms(expanded)[:80]


def _weighted_fallback_terms(terms: Sequence[str]) -> list[_WeightedLexicalTerm]:
    weighted_terms: list[_WeightedLexicalTerm] = []
    for term in _dedupe_terms(terms):
        weight = _fallback_term_weight(term)
        if weight <= 0.0:
            continue
        weighted_terms.append(
            _WeightedLexicalTerm(
                term=term,
                weight=weight,
                is_phrase=" " in term,
                is_generic=term in _GENERIC_FALLBACK_TERMS,
                is_central=term in _CENTRAL_CONTRACT_CHANGE_TERMS,
                is_salary_or_employer=term in _SALARY_EMPLOYER_OR_GENERIC_TERMS,
            )
        )
    return weighted_terms


def _fallback_term_weight(term: str) -> float:
    if term in _FALLBACK_TERM_WEIGHTS:
        return _FALLBACK_TERM_WEIGHTS[term]
    if " " in term:
        if "contract" in term and "munca" in term:
            return 6.0
        return 3.0
    if term in _GENERIC_FALLBACK_TERMS:
        return 0.25
    return 1.0


def _intent_group_score_sql(
    intent: _FallbackIntent,
    weighted_terms: Sequence[_WeightedLexicalTerm],
    *,
    condition_by_term: Mapping[str, str],
) -> dict[str, str]:
    core_score, core_count = _intent_group_sql_parts(
        intent.core_phrases,
        weighted_terms,
        condition_by_term=condition_by_term,
    )
    target_score, target_count = _intent_group_sql_parts(
        intent.target_terms,
        weighted_terms,
        condition_by_term=condition_by_term,
    )
    actor_score, actor_count = _intent_group_sql_parts(
        intent.actor_terms,
        weighted_terms,
        condition_by_term=condition_by_term,
    )
    qualifier_score, qualifier_count = _intent_group_sql_parts(
        intent.qualifier_terms,
        weighted_terms,
        condition_by_term=condition_by_term,
    )
    generic_score, generic_count = _intent_group_sql_parts(
        intent.generic_terms,
        weighted_terms,
        condition_by_term=condition_by_term,
    )
    distractor_score, distractor_count = _intent_group_sql_parts(
        intent.distractor_terms,
        weighted_terms,
        condition_by_term=condition_by_term,
    )
    return {
        "core_score": core_score,
        "target_score": target_score,
        "actor_score": actor_score,
        "qualifier_score": qualifier_score,
        "generic_score": generic_score,
        "distractor_score": distractor_score,
        "core_match_count": core_count,
        "target_match_count": target_count,
        "actor_match_count": actor_count,
        "qualifier_match_count": qualifier_count,
        "generic_match_count": generic_count,
        "distractor_match_count": distractor_count,
    }


def _intent_group_sql_parts(
    group_terms: Sequence[str],
    weighted_terms: Sequence[_WeightedLexicalTerm],
    *,
    condition_by_term: Mapping[str, str],
) -> tuple[str, str]:
    group_set = set(_dedupe_terms(group_terms))
    weighted_group_terms = [
        weighted_term
        for weighted_term in weighted_terms
        if weighted_term.term in group_set and weighted_term.term in condition_by_term
    ]
    if not weighted_group_terms:
        return "0.0", "0"
    score_parts = [
        (
            f"CASE WHEN {condition_by_term[weighted_term.term]} "
            f"THEN {weighted_term.weight:.3f} ELSE 0.0 END"
        )
        for weighted_term in weighted_group_terms
    ]
    count_parts = [
        f"CASE WHEN {condition_by_term[weighted_term.term]} THEN 1 ELSE 0 END"
        for weighted_term in weighted_group_terms
    ]
    total_weight = sum(weighted_term.weight for weighted_term in weighted_group_terms)
    return f"(({' + '.join(score_parts)}) / {total_weight:.3f})", " + ".join(count_parts)


def _intent_final_score_sql(group_sql: Mapping[str, str]) -> str:
    intent_score = (
        f"(0.55 * ({group_sql['core_score']}) "
        f"+ 0.20 * ({group_sql['target_score']}) "
        f"+ 0.10 * ({group_sql['actor_score']}) "
        f"+ 0.10 * ({group_sql['qualifier_score']}) "
        f"+ 0.05 * ({group_sql['generic_score']}) "
        f"- 0.25 * ({group_sql['distractor_score']}))"
    )
    return f"""
        GREATEST(0.0,
            CASE WHEN ({group_sql['distractor_score']}) >= 0.7
                       AND ({group_sql['core_score']}) < 0.25
                 THEN LEAST(0.35,
                     CASE WHEN ({group_sql['core_score']}) > 0
                          THEN LEAST(1.0, 0.70 + (0.30 * {intent_score}))
                          WHEN ({group_sql['core_score']}) = 0
                               AND (({group_sql['target_score']}) > 0
                                    OR ({group_sql['actor_score']}) > 0
                                    OR ({group_sql['qualifier_score']}) > 0)
                          THEN LEAST({intent_score}, 0.55)
                          WHEN ({group_sql['core_score']}) = 0
                               AND ({group_sql['target_score']}) = 0
                               AND ({group_sql['actor_score']}) = 0
                               AND ({group_sql['qualifier_score']}) = 0
                               AND ({group_sql['generic_score']}) > 0
                          THEN LEAST({intent_score}, 0.25)
                          ELSE {intent_score}
                     END)
                 WHEN ({group_sql['core_score']}) > 0
                 THEN LEAST(1.0, 0.70 + (0.30 * {intent_score}))
                 WHEN ({group_sql['core_score']}) = 0
                      AND (({group_sql['target_score']}) > 0
                           OR ({group_sql['actor_score']}) > 0
                           OR ({group_sql['qualifier_score']}) > 0)
                 THEN LEAST({intent_score}, 0.55)
                 WHEN ({group_sql['core_score']}) = 0
                      AND ({group_sql['target_score']}) = 0
                      AND ({group_sql['actor_score']}) = 0
                      AND ({group_sql['qualifier_score']}) = 0
                      AND ({group_sql['generic_score']}) > 0
                 THEN LEAST({intent_score}, 0.25)
                 ELSE {intent_score}
            END
        )
    """


def _lexical_ilike_score_for_text(
    text_value: str,
    weighted_terms: Sequence[_WeightedLexicalTerm],
    *,
    intent: _FallbackIntent | None = None,
) -> float:
    if not weighted_terms:
        return 0.0
    haystack = _normalize_text(text_value)
    if intent:
        return _intent_score_for_text(haystack, weighted_terms, intent)
    matched_weight = 0.0
    for weighted_term in weighted_terms:
        if weighted_term.term not in haystack:
            continue
        matched_weight += weighted_term.weight
    if matched_weight <= 0.0:
        return 0.0
    total_weight = sum(weighted_term.weight for weighted_term in weighted_terms)
    return matched_weight / total_weight


def _intent_score_for_text(
    haystack: str,
    weighted_terms: Sequence[_WeightedLexicalTerm],
    intent: _FallbackIntent,
) -> float:
    group_scores = _intent_group_scores_for_text(haystack, weighted_terms, intent)
    return _intent_final_score_for_groups(group_scores)


def _intent_final_score_for_groups(group_scores: Mapping[str, float]) -> float:
    core_score = group_scores["core_score"]
    target_score = group_scores["target_score"]
    actor_score = group_scores["actor_score"]
    qualifier_score = group_scores["qualifier_score"]
    generic_score = group_scores["generic_score"]
    distractor_score = group_scores["distractor_score"]
    intent_score = (
        0.55 * core_score
        + 0.20 * target_score
        + 0.10 * actor_score
        + 0.10 * qualifier_score
        + 0.05 * generic_score
        - 0.25 * distractor_score
    )
    if core_score > 0.0:
        final_score = min(1.0, 0.70 + 0.30 * intent_score)
    elif target_score > 0.0 or actor_score > 0.0 or qualifier_score > 0.0:
        final_score = min(intent_score, 0.55)
    elif generic_score > 0.0:
        final_score = min(intent_score, 0.25)
    else:
        final_score = intent_score
    if distractor_score >= 0.7 and core_score < 0.25:
        final_score = min(final_score, 0.35)
    return _clamp01(final_score)


def _intent_group_scores_for_text(
    haystack: str,
    weighted_terms: Sequence[_WeightedLexicalTerm],
    intent: _FallbackIntent,
) -> dict[str, float]:
    return {
        "core_score": _intent_group_score_for_text(
            haystack,
            weighted_terms,
            intent.core_phrases,
        ),
        "target_score": _intent_group_score_for_text(
            haystack,
            weighted_terms,
            intent.target_terms,
        ),
        "actor_score": _intent_group_score_for_text(
            haystack,
            weighted_terms,
            intent.actor_terms,
        ),
        "qualifier_score": _intent_group_score_for_text(
            haystack,
            weighted_terms,
            intent.qualifier_terms,
        ),
        "generic_score": _intent_group_score_for_text(
            haystack,
            weighted_terms,
            intent.generic_terms,
        ),
        "distractor_score": _intent_group_score_for_text(
            haystack,
            weighted_terms,
            intent.distractor_terms,
        ),
    }


def _intent_group_score_for_text(
    haystack: str,
    weighted_terms: Sequence[_WeightedLexicalTerm],
    group_terms: Sequence[str],
) -> float:
    group_set = set(_dedupe_terms(group_terms))
    weighted_group_terms = [
        weighted_term for weighted_term in weighted_terms if weighted_term.term in group_set
    ]
    if not weighted_group_terms:
        return 0.0
    matched_weight = sum(
        weighted_term.weight
        for weighted_term in weighted_group_terms
        if weighted_term.term in haystack
    )
    total_weight = sum(weighted_term.weight for weighted_term in weighted_group_terms)
    return matched_weight / total_weight if total_weight > 0.0 else 0.0


def _matched_terms(
    question: str,
    unit: Mapping[str, Any],
    *,
    query_frame: Mapping[str, Any] | None = None,
) -> list[str]:
    haystack = _normalize_text(
        f"{unit.get('raw_text') or ''} {unit.get('normalized_text') or ''}"
    )
    matches = [
        term
        for term in _expanded_query_terms(
            question,
            {"legal_domain": unit.get("legal_domain")},
            query_frame=query_frame,
        )
        if term in haystack
    ]
    normalized_question = _normalize_text(question)
    if "act aditional" in normalized_question and "act" in matches and "aditional" in matches:
        matches.append("act adițional")
    return matches[:10]


def _debug_group_scores(
    ranked_candidates: Sequence[RetrievalCandidate],
    lexical_query: Any,
) -> dict[str, dict[str, float]]:
    if not isinstance(lexical_query, _LexicalQuery) or not lexical_query.fallback_intent:
        return {}
    weighted_terms = _weighted_fallback_terms(
        _fallback_search_terms(
            lexical_query.expanded_terms,
            lexical_query.fallback_intent,
        )
    )
    scores: dict[str, dict[str, float]] = {}
    for candidate in ranked_candidates[:10]:
        unit = candidate.unit or {}
        haystack = _normalize_text(
            f"{unit.get('raw_text') or ''} {unit.get('normalized_text') or ''}"
        )
        group_scores = _intent_group_scores_for_text(
            haystack,
            weighted_terms,
            lexical_query.fallback_intent,
        )
        scores[candidate.unit_id] = {
            name: round(value, 6) for name, value in group_scores.items()
        }
    return scores


def build_registry_lexical_query(
    question: str,
    filters: Mapping[str, Any] | None = None,
    *,
    query_frame: Mapping[str, Any] | None = None,
) -> _LexicalQuery:
    return _build_lexical_query(
        question,
        filters or {},
        query_frame=query_frame,
    )


def score_unit_for_fallback(
    unit: Mapping[str, Any],
    lexical_query: _LexicalQuery,
) -> dict[str, Any]:
    weighted_terms = _weighted_fallback_terms(
        _fallback_search_terms(
            lexical_query.expanded_terms,
            lexical_query.fallback_intent,
        )
    )
    haystack = _normalize_text(
        f"{unit.get('raw_text') or ''} {unit.get('normalized_text') or ''}"
    )
    if lexical_query.fallback_intent:
        group_scores = _intent_group_scores_for_text(
            haystack,
            weighted_terms,
            lexical_query.fallback_intent,
        )
        fallback_score = _intent_final_score_for_groups(group_scores)
    else:
        group_scores = {
            "core_score": 0.0,
            "target_score": 0.0,
            "actor_score": 0.0,
            "qualifier_score": 0.0,
            "generic_score": 0.0,
            "distractor_score": 0.0,
        }
        fallback_score = _lexical_ilike_score_for_text(haystack, weighted_terms)
    return {
        "unit_id": str(unit.get("id") or ""),
        "fallback_score": round(_clamp01(fallback_score), 6),
        "group_scores": {
            name: round(value, 6) for name, value in group_scores.items()
        },
    }


def evaluate_fallback_candidates(
    *,
    question: str,
    filters: Mapping[str, Any] | None = None,
    query_frame: Mapping[str, Any] | None = None,
    candidate_units: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lexical_query = build_registry_lexical_query(
        question,
        filters or {},
        query_frame=query_frame,
    )
    ranked = []
    for unit in candidate_units:
        score_row = score_unit_for_fallback(unit, lexical_query)
        if score_row["fallback_score"] <= 0.0:
            continue
        ranked.append(
            {
                **score_row,
                "unit": dict(unit),
            }
        )
    ranked.sort(
        key=lambda row: (
            -row["fallback_score"],
            -row["group_scores"].get("core_score", 0.0),
            row["unit_id"],
        ),
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    debug_payload = _lexical_debug_payload(lexical_query, fallback_used=True)
    return {
        "lexical_query": lexical_query,
        "lexical_terms": debug_payload["lexical_terms"],
        "expanded_terms": debug_payload["expanded_terms"],
        "registry_expanded_terms": debug_payload["registry_expanded_terms"],
        "fallback_intent": debug_payload["fallback_intent"],
        "fallback_intent_source": debug_payload["fallback_intent_source"],
        "scoring_strategy": debug_payload["scoring_strategy"],
        "ranked": ranked,
    }


def _query_frame_list(
    query_frame: Mapping[str, Any] | None,
    field_name: str,
) -> list[str]:
    if not query_frame:
        return []
    value = query_frame.get(field_name)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _query_frame_retrieval_queries(
    query_frame: Mapping[str, Any] | None,
) -> list[str]:
    queries: list[str] = []
    for value in _query_frame_list(query_frame, "retrieval_queries")[:6]:
        normalized = _normalize_text(value[:240]).strip()
        if not normalized or _unsafe_decomposition_term(normalized):
            continue
        queries.append(normalized)
    return _dedupe_terms(queries)


def _query_frame_decomposition_source(
    query_frame: Mapping[str, Any] | None,
) -> str | None:
    if not query_frame:
        return None
    source = _normalize_text(str(query_frame.get("decomposition_source") or ""))
    if source in {"deterministic", "llm", "merged"}:
        return source
    return None


def _unsafe_decomposition_term(term: str) -> bool:
    return bool(
        re.search(r"\bart(?:icol|\.?)\s*\d+\b", term)
        or re.search(r"\b(?:unit_id|legal_unit_id|article_number|law_id|source_url)\b", term)
        or re.search(r"\bro\.codul", term)
        or re.search(r"\bcodul\s+muncii\b", term)
    )


def _intent_governing_rule_intent(
    query_frame: Mapping[str, Any] | None,
) -> str | None:
    if _query_frame_float(query_frame, "confidence") is None:
        return None
    if (_query_frame_float(query_frame, "confidence") or 0.0) < 0.70:
        return None
    intents = set(_query_frame_list(query_frame, "intents"))
    if "labor_contract_modification" in intents:
        return "labor_contract_modification"
    return None


def _query_frame_float(
    query_frame: Mapping[str, Any] | None,
    field_name: str,
) -> float | None:
    if not query_frame:
        return None
    try:
        return float(query_frame.get(field_name))
    except (TypeError, ValueError):
        return None


def _domain_match(filter_domain: Any, unit_domain: Any) -> float:
    if not filter_domain:
        return 0.5
    return 1.0 if _normalize_text(str(filter_domain)) == _normalize_text(str(unit_domain or "")) else 0.0


def _metadata_validity(unit: Mapping[str, Any]) -> float:
    status = _text_value(unit.get("status")) or "unknown"
    if status in {"active", "unknown"}:
        return 1.0
    if status == "historical":
        return 0.7
    if status == "repealed":
        return 0.4
    return 0.7


def _intent_governing_rule_rows_from_units(
    units: Sequence[Mapping[str, Any]],
    *,
    intent: str,
    filters: Mapping[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    if intent != "labor_contract_modification":
        return []
    rows: list[dict[str, Any]] = []
    for unit in units:
        if not _unit_matches_filters(unit, filters):
            continue
        match = _labor_contract_modification_governing_rule_match(
            f"{unit.get('raw_text') or ''} {unit.get('normalized_text') or ''}"
        )
        if not match.matched:
            continue
        rows.append(
            {
                **dict(unit),
                "intent_governing_rule_score": match.score,
                "intent_governing_rule_role": match.role,
                "intent_governing_rule_terms": list(match.matched_terms),
            }
        )
    rows.sort(
        key=lambda row: (
            -_float(row.get("intent_governing_rule_score")),
            _intent_governing_rule_role_order(
                _text_value(row.get("intent_governing_rule_role"))
            ),
            str(row.get("id") or ""),
        )
    )
    return rows[:limit]


def _unit_matches_filters(
    unit: Mapping[str, Any],
    filters: Mapping[str, Any],
) -> bool:
    legal_domain = _text_value(filters.get("legal_domain"))
    if legal_domain and _domain_match(legal_domain, unit.get("legal_domain")) <= 0.0:
        return False
    status = _text_value(filters.get("status"))
    if status and _text_value(unit.get("status")) != status:
        return False
    law_id = _text_value(filters.get("law_id"))
    if law_id and _text_value(unit.get("law_id")) != law_id:
        return False
    return True


def _labor_contract_modification_governing_rule_match(
    text_value: str,
) -> _IntentGoverningRuleMatch:
    haystack = _normalize_text(text_value)
    if _contains_any(haystack, _LABOR_CONTRACT_MODIFICATION_DISTRACTOR_TERMS):
        return _IntentGoverningRuleMatch(matched=False)

    agreement_terms = _matched_term_values(
        haystack,
        (
            _LABOR_CONTRACT_PHRASES,
            _LABOR_MODIFICATION_TERMS,
            _LABOR_AGREEMENT_TERMS,
        ),
    )
    if agreement_terms:
        return _IntentGoverningRuleMatch(
            matched=True,
            score=1.0,
            role="agreement_rule",
            matched_terms=tuple(agreement_terms),
        )

    salary_scope_terms = _matched_term_values(
        haystack,
        (
            _LABOR_MODIFICATION_SCOPE_PHRASES,
            _SALARY_TERMS,
        ),
    )
    if salary_scope_terms:
        return _IntentGoverningRuleMatch(
            matched=True,
            score=0.92,
            role="salary_scope",
            matched_terms=tuple(salary_scope_terms),
        )

    target_terms = [
        term for term in _SALARY_TERMS if _phrase_in_haystack(term, haystack)
    ]
    if target_terms and len(haystack) <= 120:
        return _IntentGoverningRuleMatch(
            matched=True,
            score=0.28,
            role="target_element",
            matched_terms=tuple(target_terms),
        )

    return _IntentGoverningRuleMatch(matched=False)


def _matched_term_values(
    haystack: str,
    groups: Sequence[Sequence[str]],
) -> list[str]:
    matched: list[str] = []
    for group in groups:
        group_match = next(
            (term for term in group if _phrase_in_haystack(term, haystack)),
            None,
        )
        if not group_match:
            return []
        matched.append(group_match)
    return matched


def _contains_any(haystack: str, terms: Sequence[str]) -> bool:
    return any(_phrase_in_haystack(term, haystack) for term in terms)


def _phrase_in_haystack(term: str, haystack: str) -> bool:
    normalized = _normalize_text(term)
    if not normalized:
        return False
    if " " in normalized:
        return normalized in haystack
    return normalized in set(re.split(r"[^a-z0-9_]+", haystack))


def _intent_governing_rule_retrieval_score(
    governing_rule_score: float,
    *,
    domain_match: float,
    metadata_validity: float,
) -> float:
    if governing_rule_score <= 0.0:
        return 0.0
    return _clamp01(
        0.40
        + 0.35 * governing_rule_score
        + 0.15 * domain_match
        + 0.10 * metadata_validity
    )


def _intent_governing_rule_role_order(role: str | None) -> int:
    order = {
        "agreement_rule": 0,
        "salary_scope": 1,
        "target_element": 2,
    }
    return order.get(role or "", 99)


def _intent_governing_rule_terms(intent: str | None) -> dict[str, list[str]]:
    if intent != "labor_contract_modification":
        return {}
    return {
        "agreement_rule": [
            *_LABOR_CONTRACT_PHRASES,
            *_LABOR_MODIFICATION_TERMS,
            *_LABOR_AGREEMENT_TERMS,
        ],
        "salary_scope": [
            *_LABOR_MODIFICATION_SCOPE_PHRASES,
            *_SALARY_TERMS,
        ],
        "target_element": list(_SALARY_TERMS),
        "distractors": list(_LABOR_CONTRACT_MODIFICATION_DISTRACTOR_TERMS),
    }


def _sql_normalized_text(expression: str) -> str:
    return (
        "lower(translate("
        f"{expression}, "
        "'ăâîșşțţĂÂÎȘŞȚŢ', "
        "'aaissttaaisstt'"
        "))"
    )


def _labor_contract_modification_governing_rule_sql(
    normalized_search_document: str,
) -> tuple[dict[str, str], dict[str, str]]:
    params: dict[str, str] = {}
    agreement_contract = _sql_any_like(
        normalized_search_document,
        _LABOR_CONTRACT_PHRASES,
        "gov_agreement_contract",
        params,
    )
    agreement_modification = _sql_any_like(
        normalized_search_document,
        _LABOR_MODIFICATION_TERMS,
        "gov_agreement_modification",
        params,
    )
    agreement_parties = _sql_any_like(
        normalized_search_document,
        _LABOR_AGREEMENT_TERMS,
        "gov_agreement_parties",
        params,
    )
    salary_scope = _sql_any_like(
        normalized_search_document,
        _LABOR_MODIFICATION_SCOPE_PHRASES,
        "gov_salary_scope",
        params,
    )
    salary = _sql_any_like(
        normalized_search_document,
        tuple(sorted(_SALARY_TERMS)),
        "gov_salary",
        params,
    )
    distractors = _sql_any_like(
        normalized_search_document,
        _LABOR_CONTRACT_MODIFICATION_DISTRACTOR_TERMS,
        "gov_distractor",
        params,
    )
    target_element = f"({salary} AND length({normalized_search_document}) <= 120)"
    agreement_rule = (
        f"({agreement_contract} AND {agreement_modification} AND {agreement_parties})"
    )
    salary_scope_rule = f"({salary_scope} AND {salary})"
    no_distractors = f"NOT ({distractors})"
    match = (
        f"{no_distractors} AND "
        f"({agreement_rule} OR {salary_scope_rule} OR {target_element})"
    )
    score = (
        "CASE "
        f"WHEN {no_distractors} AND {agreement_rule} THEN 1.0 "
        f"WHEN {no_distractors} AND {salary_scope_rule} THEN 0.92 "
        f"WHEN {no_distractors} AND {target_element} THEN 0.28 "
        "ELSE 0.0 END"
    )
    role = (
        "CASE "
        f"WHEN {no_distractors} AND {agreement_rule} THEN 'agreement_rule' "
        f"WHEN {no_distractors} AND {salary_scope_rule} THEN 'salary_scope' "
        f"WHEN {no_distractors} AND {target_element} THEN 'target_element' "
        "ELSE NULL END"
    )
    return {"match": match, "score": score, "role": role}, params


def _sql_any_like(
    expression: str,
    terms: Sequence[str],
    prefix: str,
    params: dict[str, str],
) -> str:
    clauses: list[str] = []
    for index, term in enumerate(_dedupe_terms(terms)):
        name = f"{prefix}_{index}"
        params[name] = f"%{term}%"
        clauses.append(f"{expression} LIKE :{name}")
    return "(" + " OR ".join(clauses or ["FALSE"]) + ")"


def _why_retrieved(methods: set[str]) -> str:
    labels = []
    if EXACT_METHOD in methods:
        labels.append("exact citation")
    if FTS_METHOD in methods:
        labels.append("lexical match")
    if INTENT_GOVERNING_RULE_METHOD in methods:
        labels.append("intent_governing_rule_lookup:labor_contract_modification")
    if DENSE_METHOD in methods:
        labels.append("dense similarity")
    return " + ".join(labels) if labels else "domain filtered search"


def _normalize_score(value: float, maximum: float) -> float:
    if maximum <= 0.0:
        return 0.0
    return _clamp01(value / maximum)


def _clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _text_value(value: Any) -> str | None:
    if value is None:
        return None
    text_value = str(value).strip()
    return text_value or None


def _normalize_text(value: str) -> str:
    value = (
        value.replace("Äƒ", "a")
        .replace("È™", "s")
        .replace("È›", "t")
        .replace("Å£", "t")
        .replace("ÅŸ", "s")
    )
    value = value.translate(_ROMANIAN_DIACRITIC_TRANSLATION)
    normalized = unicodedata.normalize("NFKD", value.lower())
    stripped = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return _repair_legal_replacement_marks(stripped)


def _repair_legal_replacement_marks(value: str) -> str:
    for pattern, replacement in _LEGAL_REPLACEMENT_MARK_REPAIRS:
        value = pattern.sub(replacement, value)
    return value


def _canonical_domain(value: Any) -> str:
    return _normalize_text(str(value)).replace(" ", "_")


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


_UNIT_COLUMNS = (
    "id",
    "canonical_id",
    "source_id",
    "law_id",
    "law_title",
    "act_type",
    "act_number",
    "publication_date",
    "effective_date",
    "version_start",
    "version_end",
    "status",
    "hierarchy_path",
    "article_number",
    "paragraph_number",
    "letter_number",
    "point_number",
    "raw_text",
    "normalized_text",
    "legal_domain",
    "legal_concepts",
    "source_url",
    "parent_id",
    "parser_warnings",
    "created_at",
    "updated_at",
)
_MAX_WARNING_MESSAGE_LENGTH = 240
_WARNING_REDACTIONS = (
    (
        re.compile(r"(?i)\b(postgres(?:ql)?(?:\+asyncpg)?://)([^:@/\s]+):([^@/\s]+)@"),
        r"\1\2:***@",
    ),
    (re.compile(r"(?i)(\bpassword=)([^&\s;]+)"), r"\1***"),
    (re.compile(r"(?i)\bDATABASE_URL=\S+"), "<redacted_database_url>"),
    (
        re.compile(
            r"(?is)((?:['\"])?(?:raw_text|normalized_text|embedding_text)(?:['\"])?\s*[:=]\s*)(['\"])(.*?)(\2)"
        ),
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>{match.group(4)}",
    ),
    (re.compile(r"(?i)\b(raw_text|normalized_text|embedding_text)\b"), "<text_field>"),
    (
        re.compile(r"(?is)((?:['\"])?embedding(?:['\"])?\s*[:=]\s*)\[[^\]]*\]"),
        r"\1[<redacted>]",
    ),
    (
        re.compile(
            r"\[(?:\s*-?\d+(?:\.\d+)?(?:e[+-]?\d+)?\s*,){2,}\s*-?\d+(?:\.\d+)?(?:e[+-]?\d+)?\s*\]",
            re.IGNORECASE,
        ),
        "[<redacted_vector>]",
    ),
)
_ROMANIAN_DIACRITIC_TRANSLATION = str.maketrans(
    {
        "\u0103": "a",
        "\u00e2": "a",
        "\u00ee": "i",
        "\u0219": "s",
        "\u015f": "s",
        "\u021b": "t",
        "\u0163": "t",
        "\u0102": "A",
        "\u00c2": "A",
        "\u00ce": "I",
        "\u0218": "S",
        "\u015e": "S",
        "\u021a": "T",
        "\u0162": "T",
    }
)
_LEGAL_REPLACEMENT_MARK_REPAIRS = (
    (
        re.compile(r"\badi[?\ufffd]ional(a?)\b"),
        lambda match: f"aditional{match.group(1)}",
    ),
)
LABOR_FALLBACK_INTENTS = {
    "labor_contract_modification": _FallbackIntent(
        name="labor_contract_modification",
        triggers=(
            "act aditional",
            "modificare contract",
            "modificarea contractului",
            "scada salariul",
            "scade salariul",
            "salariul fara act aditional",
        ),
        core_phrases=(
            "modificarea contractului individual de munca",
            "modificare contract individual munca",
            "contractului individual de munca",
            "contract individual de munca",
            "contract individual munca",
            "contractul individual de munca",
            "modificarea contractului",
            "modificare contract",
            "acordul partilor",
            "acord partilor",
            "acord parti",
            "numai prin acordul partilor",
            "poate fi modificat",
            "modificat numai prin acordul partilor",
        ),
        target_terms=(
            "salariu",
            "salariul",
            "salarizare",
        ),
        actor_terms=(
            "angajator",
            "angajatorul",
            "salariat",
            "salariatul",
        ),
        qualifier_terms=(
            "fara act aditional",
            "fara acord",
            "fara acordul partilor",
            "unilateral",
        ),
        generic_terms=(
            "act",
            "contract",
            "contractul",
            "contractului",
            "acord",
            "acordul",
            "parti",
            "partilor",
        ),
        distractor_terms=(
            "remuneratie restanta",
            "persoane angajate ilegal",
            "neplata salariului",
            "intarzierea platii salariului",
            "salariul minim",
            "confidentialitatea salariului",
            "registrul general de evidenta",
            "formare profesionala",
            "drepturile si obligatiile partilor",
            "durata formarii profesionale",
            "semnatura electronica",
            "delegarea",
            "detasarea",
            "locul muncii poate fi modificat unilateral",
            "recuperarea contravalorii pagubei",
            "nota de constatare",
            "clauza de neconcurenta",
            "munca temporara",
            "acord scris pentru evidenta orelor",
        ),
        source="legacy",
    )
}
_LABOR_CONTRACT_MODIFICATION_INTENT = LABOR_FALLBACK_INTENTS[
    "labor_contract_modification"
]
_GENERIC_FALLBACK_TERMS = set(_LABOR_CONTRACT_MODIFICATION_INTENT.generic_terms)
_CENTRAL_CONTRACT_CHANGE_TERMS = set(
    _LABOR_CONTRACT_MODIFICATION_INTENT.core_phrases
)
_SALARY_EMPLOYER_OR_GENERIC_TERMS = {
    *_GENERIC_FALLBACK_TERMS,
    *_LABOR_CONTRACT_MODIFICATION_INTENT.target_terms,
    *_LABOR_CONTRACT_MODIFICATION_INTENT.actor_terms,
    "salarii",
    "salarial",
}
_FALLBACK_TERM_WEIGHTS = {
    "modificarea contractului individual de munca": 9.0,
    "modificare contract individual munca": 8.0,
    "contractului individual de munca": 7.5,
    "contractul individual de munca": 7.5,
    "contract individual de munca": 7.0,
    "contract individual munca": 6.5,
    "numai prin acordul partilor": 6.5,
    "modificat numai prin acordul partilor": 6.0,
    "acordul partilor": 5.5,
    "poate fi modificat": 4.5,
    "acord partilor": 4.5,
    "acord parti": 4.0,
    "modificarea contractului": 4.5,
    "modificare contract": 4.0,
    "salariul": 3.0,
    "salariu": 3.0,
    "salarizare": 2.5,
    "modificat": 1.5,
    "modificare": 1.5,
    "scada": 1.0,
    "scade": 1.0,
    "angajator": 1.0,
    "angajatorul": 1.0,
    "aditional": 1.0,
}
_STOP_WORDS = {
    "care",
    "este",
    "sunt",
    "fara",
    "fără",
    "poate",
    "prin",
    "din",
    "pentru",
    "angajatorul",
    "imi",
    "îmi",
    "sa",
    "să",
}
_STOP_WORDS.update(
    {
        "a",
        "al",
        "ale",
        "am",
        "ar",
        "as",
        "ce",
        "cu",
        "de",
        "eu",
        "fi",
        "in",
        "la",
        "mai",
        "mi",
        "nu",
        "pe",
        "sau",
        "se",
        "si",
        "un",
        "unei",
        "unui",
    }
)
_STOP_WORDS.discard("angajatorul")
_SALARY_TERMS = {"salariu", "salariul", "salarizare", "salarial", "salarii"}
_LABOR_CONTRACT_PHRASES = (
    "contract individual de munca",
    "contractul individual de munca",
    "contractului individual de munca",
    "contractele individuale de munca",
)
_LABOR_MODIFICATION_TERMS = (
    "modificat",
    "modifica",
    "modificare",
    "modificarea",
)
_LABOR_AGREEMENT_TERMS = (
    "acordul partilor",
    "acord partilor",
    "acord parti",
)
_LABOR_MODIFICATION_SCOPE_PHRASES = (
    "modificarea contractului individual de munca",
    "modificare contract individual munca",
)
_LABOR_CONTRACT_MODIFICATION_DISTRACTOR_TERMS = (
    "formare profesionala",
    "durata formarii profesionale",
    "semnatura electronica",
    "delegarea",
    "detasarea",
    "recuperarea pagubei",
    "recuperarea contravalorii pagubei",
    "nota de constatare",
    "clauza de neconcurenta",
    "salariul minim",
    "munca temporara",
    "evidenta orelor",
)
_LABOR_CONTEXT_TERMS = {
    "act",
    "aditional",
    "angajator",
    "angajatorul",
    "contract",
    "contractul",
    "munca",
    "salariu",
    "salariul",
    "salarizare",
}
