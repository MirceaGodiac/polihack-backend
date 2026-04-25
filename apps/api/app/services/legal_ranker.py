from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Any

from ..schemas import (
    ExpandedCandidate,
    GraphExpansionResult,
    LegalRankerResult,
    QueryPlan,
    RankedCandidate,
    RankerFeatureBreakdown,
    RawRetrievalResponse,
    RetrievalCandidate,
)

LEGAL_RANKER_NO_CANDIDATES = "legal_ranker_no_candidates"

RANKER_WEIGHTS: dict[str, float] = {
    "bm25_score": 0.16,
    "dense_score": 0.16,
    "exact_citation_match": 0.10,
    "domain_match": 0.10,
    "graph_proximity": 0.10,
    "concept_overlap": 0.08,
    "legal_term_overlap": 0.07,
    "temporal_validity": 0.07,
    "source_reliability": 0.05,
    "parent_relevance": 0.05,
    "is_exception": 0.03,
    "is_definition": 0.02,
    "is_sanction": 0.01,
}

FEATURE_NAMES = tuple(RANKER_WEIGHTS.keys())

RAW_BM25_KEYS = ("bm25_score", "bm25", "lexical", "lexical_score")
RAW_DENSE_KEYS = ("dense_score", "dense", "vector", "vector_score")

STOPWORDS = {
    "a",
    "al",
    "ale",
    "alin",
    "art",
    "cu",
    "de",
    "din",
    "este",
    "fara",
    "in",
    "la",
    "mi",
    "nr",
    "pe",
    "sa",
    "sau",
    "se",
    "si",
}


@dataclass
class _CandidateBundle:
    unit_id: str
    retrieval_candidate: RetrievalCandidate | None = None
    expanded_candidate: ExpandedCandidate | None = None
    unit: dict[str, Any] | None = None
    score_breakdown: dict[str, float] = field(default_factory=dict)
    graph_proximity: float = 0.0
    retrieval_score: float | None = None
    retrieval_rank: int | None = None
    sources: set[str] = field(default_factory=set)
    reasons: list[str] = field(default_factory=list)


class LegalRanker:
    def rank(
        self,
        *,
        question: str,
        plan: QueryPlan,
        retrieval_response: RawRetrievalResponse,
        graph_expansion: GraphExpansionResult,
        debug: bool = False,
    ) -> LegalRankerResult:
        bundles = self._merge_candidates(
            retrieval_response=retrieval_response,
            graph_expansion=graph_expansion,
        )
        input_candidate_count = len(bundles)
        if not bundles:
            return self._fallback_result(debug=debug)

        raw_feature_rows = [
            self._extract_features(
                question=question,
                plan=plan,
                bundle=bundle,
            )
            for bundle in bundles
        ]
        normalized_feature_rows = self._normalize_rows(raw_feature_rows)
        scored_rows = [
            (
                bundle,
                RankerFeatureBreakdown(**normalized_features),
                self._weighted_score(normalized_features),
            )
            for bundle, normalized_features in zip(
                bundles,
                normalized_feature_rows,
                strict=True,
            )
        ]
        scored_rows.sort(key=self._sort_key)

        ranked_candidates = [
            self._ranked_candidate(
                rank=index,
                bundle=bundle,
                score_breakdown=score_breakdown,
                rerank_score=rerank_score,
                plan=plan,
            )
            for index, (bundle, score_breakdown, rerank_score) in enumerate(
                scored_rows,
                start=1,
            )
        ]

        result = LegalRankerResult(ranked_candidates=ranked_candidates)
        if debug:
            result.debug = self._debug_payload(
                fallback_used=False,
                input_candidate_count=input_candidate_count,
                ranked_candidates=ranked_candidates,
                warnings=[],
            )
        return result

    def _fallback_result(self, *, debug: bool) -> LegalRankerResult:
        result = LegalRankerResult(
            ranked_candidates=[],
            warnings=[LEGAL_RANKER_NO_CANDIDATES],
            debug=None,
        )
        if debug:
            result.debug = self._debug_payload(
                fallback_used=True,
                input_candidate_count=0,
                ranked_candidates=[],
                warnings=result.warnings,
            )
        return result

    def _merge_candidates(
        self,
        *,
        retrieval_response: RawRetrievalResponse,
        graph_expansion: GraphExpansionResult,
    ) -> list[_CandidateBundle]:
        merged: dict[str, _CandidateBundle] = {}

        for candidate in retrieval_response.candidates:
            bundle = merged.setdefault(
                candidate.unit_id,
                _CandidateBundle(unit_id=candidate.unit_id),
            )
            self._merge_retrieval_candidate(bundle, candidate)

        for expanded in (
            graph_expansion.seed_candidates + graph_expansion.expanded_candidates
        ):
            bundle = merged.setdefault(
                expanded.unit_id,
                _CandidateBundle(unit_id=expanded.unit_id),
            )
            self._merge_expanded_candidate(bundle, expanded)
            if expanded.retrieval_candidate:
                self._merge_retrieval_candidate(
                    bundle,
                    expanded.retrieval_candidate,
                )

        return list(merged.values())

    def _merge_retrieval_candidate(
        self,
        bundle: _CandidateBundle,
        candidate: RetrievalCandidate,
    ) -> None:
        if (
            bundle.retrieval_candidate is None
            or self._unit_info_score(candidate.unit)
            > self._unit_info_score(bundle.retrieval_candidate.unit)
        ):
            bundle.retrieval_candidate = candidate
        if self._unit_info_score(candidate.unit) > self._unit_info_score(bundle.unit):
            bundle.unit = candidate.unit
        if candidate.retrieval_score is not None:
            bundle.retrieval_score = max(
                bundle.retrieval_score or 0.0,
                candidate.retrieval_score,
            )
        bundle.retrieval_rank = (
            candidate.rank
            if bundle.retrieval_rank is None
            else min(bundle.retrieval_rank, candidate.rank)
        )
        bundle.score_breakdown.update(candidate.score_breakdown)
        bundle.sources.add("raw_retrieval")
        if candidate.why_retrieved:
            bundle.reasons.append(candidate.why_retrieved)

    def _merge_expanded_candidate(
        self,
        bundle: _CandidateBundle,
        candidate: ExpandedCandidate,
    ) -> None:
        bundle.expanded_candidate = candidate
        bundle.graph_proximity = max(
            bundle.graph_proximity,
            self._clamp(candidate.graph_proximity),
        )
        bundle.score_breakdown.update(candidate.score_breakdown)
        bundle.sources.add(candidate.source)
        if candidate.expansion_reason:
            bundle.reasons.append(candidate.expansion_reason)
        if candidate.expansion_edge_type:
            bundle.reasons.append(candidate.expansion_edge_type)

    def _extract_features(
        self,
        *,
        question: str,
        plan: QueryPlan,
        bundle: _CandidateBundle,
    ) -> dict[str, float]:
        unit = bundle.unit or {}
        return {
            "bm25_score": self._score_from_breakdown(
                bundle.score_breakdown,
                RAW_BM25_KEYS,
            ),
            "dense_score": self._score_from_breakdown(
                bundle.score_breakdown,
                RAW_DENSE_KEYS,
            ),
            "exact_citation_match": self._exact_citation_match(plan, bundle),
            "domain_match": self._domain_match(plan, unit),
            "graph_proximity": self._graph_proximity(bundle),
            "concept_overlap": self._concept_overlap(question, plan, unit),
            "legal_term_overlap": self._legal_term_overlap(question, plan, unit),
            "temporal_validity": self._temporal_validity(unit),
            "source_reliability": self._source_reliability(unit),
            "parent_relevance": self._parent_relevance(unit),
            "is_exception": self._indicator(
                unit=unit,
                bundle=bundle,
                terms=(
                    "exception",
                    "exceptie",
                    "except",
                    "cu exceptia",
                    "exception_to",
                ),
            ),
            "is_definition": self._indicator(
                unit=unit,
                bundle=bundle,
                terms=(
                    "definition",
                    "defineste",
                    "in sensul",
                    "se intelege",
                    "defines",
                ),
            ),
            "is_sanction": self._indicator(
                unit=unit,
                bundle=bundle,
                terms=(
                    "sanction",
                    "sanctiune",
                    "amenda",
                    "contraventie",
                    "sanctions",
                ),
            ),
        }

    def _normalize_rows(
        self,
        raw_feature_rows: list[dict[str, float]],
    ) -> list[dict[str, float]]:
        normalized = [{name: 0.0 for name in FEATURE_NAMES} for _ in raw_feature_rows]
        for feature_name in FEATURE_NAMES:
            values = [row.get(feature_name, 0.0) for row in raw_feature_rows]
            if any(value > 1.0 for value in values):
                normalized_values = self._min_max_normalize(values)
            else:
                normalized_values = [self._clamp(value) for value in values]
            for row, value in zip(normalized, normalized_values, strict=True):
                row[feature_name] = value
        return normalized

    def _min_max_normalize(self, values: list[float]) -> list[float]:
        if all(value == 0.0 for value in values):
            return [0.0 for _ in values]
        minimum = min(values)
        maximum = max(values)
        if minimum == maximum:
            return [0.5 if minimum != 0.0 else 0.0 for _ in values]
        return [self._clamp((value - minimum) / (maximum - minimum)) for value in values]

    def _weighted_score(self, features: dict[str, float]) -> float:
        score = sum(
            RANKER_WEIGHTS[feature_name] * features.get(feature_name, 0.0)
            for feature_name in FEATURE_NAMES
        )
        return round(self._clamp(score), 6)

    def _sort_key(
        self,
        row: tuple[_CandidateBundle, RankerFeatureBreakdown, float],
    ) -> tuple[float, float, float, float, int, str]:
        bundle, score_breakdown, rerank_score = row
        return (
            -rerank_score,
            -score_breakdown.exact_citation_match,
            -score_breakdown.domain_match,
            -score_breakdown.graph_proximity,
            bundle.retrieval_rank or 1_000_000,
            bundle.unit_id,
        )

    def _ranked_candidate(
        self,
        *,
        rank: int,
        bundle: _CandidateBundle,
        score_breakdown: RankerFeatureBreakdown,
        rerank_score: float,
        plan: QueryPlan,
    ) -> RankedCandidate:
        return RankedCandidate(
            unit_id=bundle.unit_id,
            rank=rank,
            rerank_score=rerank_score,
            retrieval_score=bundle.retrieval_score,
            unit=bundle.unit,
            score_breakdown=score_breakdown,
            why_ranked=self._why_ranked(plan, bundle, score_breakdown),
            source=self._source(bundle),
        )

    def _why_ranked(
        self,
        plan: QueryPlan,
        bundle: _CandidateBundle,
        score_breakdown: RankerFeatureBreakdown,
    ) -> list[str]:
        reasons: list[str] = []
        if score_breakdown.bm25_score >= 0.70:
            reasons.append("high_bm25_score")
        if score_breakdown.dense_score >= 0.70:
            reasons.append("high_dense_score")
        if score_breakdown.exact_citation_match > 0:
            reasons.append("exact_citation_match")
        if score_breakdown.domain_match == 1.0 and plan.legal_domain:
            reasons.append(f"domain_match:{self._normal_label(plan.legal_domain)}")
        if score_breakdown.graph_proximity > 0 and bundle.expanded_candidate:
            reasons.append("graph_proximity_from_expansion")
        if score_breakdown.concept_overlap > 0:
            reasons.append("concept_overlap")
        if score_breakdown.legal_term_overlap > 0:
            reasons.append("legal_term_overlap")
        if score_breakdown.temporal_validity == 1.0:
            reasons.append("active_legal_unit")
        if score_breakdown.source_reliability == 1.0:
            reasons.append("has_source_url")
        if score_breakdown.is_exception > 0:
            reasons.append("exception_candidate")
        if score_breakdown.is_definition > 0:
            reasons.append("definition_candidate")
        if score_breakdown.is_sanction > 0:
            reasons.append("sanction_candidate")
        if score_breakdown.parent_relevance > 0:
            reasons.append("parent_context_candidate")
        for reason in bundle.reasons:
            normalized_reason = self._normal_label(reason)
            if normalized_reason and normalized_reason not in reasons:
                reasons.append(normalized_reason)
        return reasons

    def _score_from_breakdown(
        self,
        score_breakdown: dict[str, float],
        keys: tuple[str, ...],
    ) -> float:
        for key in keys:
            if key in score_breakdown:
                return float(score_breakdown[key])
        return 0.0

    def _exact_citation_match(
        self,
        plan: QueryPlan,
        bundle: _CandidateBundle,
    ) -> float:
        if bundle.score_breakdown.get("exact_citation_boost", 0.0) > 0:
            return 1.0
        if bundle.score_breakdown.get("exact_citation_match", 0.0) > 0:
            return 1.0
        if not plan.exact_citations:
            return 0.0

        haystack = self._candidate_haystack(bundle)
        for citation in plan.exact_citations:
            article_match = True
            if citation.article:
                article = self._normalize_text(citation.article)
                article_match = any(
                    marker in haystack
                    for marker in (
                        f"art {article}",
                        f"art_{article}",
                        f"art-{article}",
                        f"article {article}",
                    )
                )

            paragraph_match = True
            if citation.paragraph:
                paragraph = self._normalize_text(citation.paragraph)
                paragraph_match = any(
                    marker in haystack
                    for marker in (
                        f"alin {paragraph}",
                        f"alin_{paragraph}",
                        f"paragraph {paragraph}",
                    )
                )

            law_match = True
            if citation.law_id_hint:
                law = self._normalize_text(citation.law_id_hint)
                law_match = law in haystack
            elif citation.act_hint:
                law = self._normalize_text(citation.act_hint)
                law_match = law in haystack

            if article_match and paragraph_match and law_match:
                return 1.0
        return 0.0

    def _domain_match(self, plan: QueryPlan, unit: dict[str, Any]) -> float:
        if not plan.legal_domain:
            return 0.0
        unit_domain = self._unit_value(unit, "legal_domain", "domain")
        if not unit_domain:
            return 0.3
        plan_domain = self._normalize_text(plan.legal_domain)
        normalized_unit_domain = self._normalize_text(str(unit_domain))
        if (
            plan_domain == normalized_unit_domain
            or plan_domain in normalized_unit_domain
            or normalized_unit_domain in plan_domain
        ):
            return 1.0
        return 0.0

    def _graph_proximity(self, bundle: _CandidateBundle) -> float:
        if bundle.graph_proximity > 0:
            return bundle.graph_proximity
        if bundle.retrieval_candidate:
            return 1.0
        return 0.0

    def _concept_overlap(
        self,
        question: str,
        plan: QueryPlan,
        unit: dict[str, Any],
    ) -> float:
        concepts = unit.get("legal_concepts")
        if not concepts:
            return 0.0
        concept_tokens = self._tokens_from_value(concepts)
        if not concept_tokens:
            return 0.0
        query_tokens = self._tokenize(question) | {
            self._normalize_text(query_type) for query_type in plan.query_types
        }
        return len(concept_tokens & query_tokens) / len(concept_tokens)

    def _legal_term_overlap(
        self,
        question: str,
        plan: QueryPlan,
        unit: dict[str, Any],
    ) -> float:
        text = self._unit_value(unit, "normalized_text", "raw_text", "text")
        if not text:
            return 0.0
        question_tokens = self._tokenize(plan.normalized_question or question)
        unit_tokens = self._tokenize(str(text))
        if not question_tokens or not unit_tokens:
            return 0.0
        return len(question_tokens & unit_tokens) / len(question_tokens)

    def _temporal_validity(self, unit: dict[str, Any]) -> float:
        status = self._normalize_text(str(unit.get("status") or "unknown"))
        if status == "active":
            return 1.0
        if status in {"unknown", ""}:
            return 0.6
        if status == "historical":
            return 0.2
        if status == "repealed":
            return 0.0
        return 0.6

    def _source_reliability(self, unit: dict[str, Any]) -> float:
        if unit.get("source_url"):
            return 1.0
        if any(unit.get(key) for key in ("law_id", "law_title", "raw_text")):
            return 0.8
        return 0.6

    def _parent_relevance(self, unit: dict[str, Any]) -> float:
        unit_type = self._normalize_text(str(unit.get("type") or unit.get("unit_type") or ""))
        if unit.get("parent_id") and unit_type in {"paragraf", "paragraph", "alineat", "litera", "letter", "punct", "point"}:
            return 0.6
        if unit_type in {"articol", "article"}:
            return 0.3
        return 0.0

    def _indicator(
        self,
        *,
        unit: dict[str, Any],
        bundle: _CandidateBundle,
        terms: tuple[str, ...],
    ) -> float:
        haystack = self._candidate_haystack(bundle)
        if unit:
            haystack = f"{haystack} {self._normalize_text(str(unit))}"
        return 1.0 if any(self._normalize_text(term) in haystack for term in terms) else 0.0

    def _debug_payload(
        self,
        *,
        fallback_used: bool,
        input_candidate_count: int,
        ranked_candidates: list[RankedCandidate],
        warnings: list[str],
    ) -> dict[str, Any]:
        rows = [candidate.model_dump(mode="json") for candidate in ranked_candidates]
        return {
            "fallback_used": fallback_used,
            "input_candidate_count": input_candidate_count,
            "ranked_candidate_count": len(ranked_candidates),
            "weights": RANKER_WEIGHTS,
            "ranked_candidates": rows,
            "rows": rows,
            "warnings": warnings,
        }

    def _candidate_haystack(self, bundle: _CandidateBundle) -> str:
        values: list[str] = [bundle.unit_id]
        if bundle.unit:
            values.extend(str(value) for value in bundle.unit.values())
        values.extend(bundle.reasons)
        if bundle.expanded_candidate:
            values.extend(
                str(value)
                for value in (
                    bundle.expanded_candidate.source,
                    bundle.expanded_candidate.expansion_edge_type,
                    bundle.expanded_candidate.expansion_reason,
                )
                if value
            )
        return self._normalize_text(" ".join(values))

    def _unit_value(self, unit: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = unit.get(key)
            if value is not None:
                return value
        return None

    def _unit_info_score(self, unit: dict[str, Any] | None) -> int:
        if not unit:
            return 0
        return sum(1 for value in unit.values() if value not in (None, "", [], {}))

    def _tokens_from_value(self, value: Any) -> set[str]:
        if isinstance(value, list | tuple | set):
            return {
                token
                for item in value
                for token in self._tokenize(str(item))
            }
        return self._tokenize(str(value))

    def _tokenize(self, text: str) -> set[str]:
        normalized = self._normalize_text(text)
        return {
            token
            for token in re.split(r"[^a-z0-9_]+", normalized)
            if len(token) > 1 and token not in STOPWORDS
        }

    def _normalize_text(self, text: str) -> str:
        replacements = {
            "Äƒ": "ă",
            "Ä‚": "Ă",
            "È™": "ș",
            "È˜": "Ș",
            "È›": "ț",
            "Èš": "Ț",
            "Ã¢": "â",
            "Ã‚": "Â",
            "Ã®": "î",
            "ÃŽ": "Î",
            "ÅŸ": "ș",
            "Åž": "Ș",
            "Å£": "ț",
            "Å¢": "Ț",
        }
        for broken, fixed in replacements.items():
            text = text.replace(broken, fixed)
        normalized = unicodedata.normalize("NFD", text.casefold())
        stripped = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        return " ".join(stripped.replace(".", " ").replace("-", "_").split())

    def _normal_label(self, text: str) -> str:
        return self._normalize_text(text).replace(" ", "_")

    def _source(self, bundle: _CandidateBundle) -> str | None:
        if not bundle.sources:
            return None
        if "graph_expansion" in bundle.sources:
            return "graph_expansion"
        if "seed" in bundle.sources:
            return "seed"
        if "raw_retrieval" in bundle.sources:
            return "raw_retrieval"
        return sorted(bundle.sources)[0]

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))
