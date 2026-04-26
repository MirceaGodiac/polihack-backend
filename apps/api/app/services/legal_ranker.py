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
from .query_frame import LegalIntent, LegalIntentRegistry, QueryFrame

LEGAL_RANKER_NO_CANDIDATES = "legal_ranker_no_candidates"
LEGAL_RANKER_V1_SCORING = "legal_ranker_v1_linear"
LEGAL_RANKER_V2_SCORING = "legal_ranker_v2_query_frame_gated"
QUERY_FRAME_CONFIDENCE_THRESHOLD = 0.35

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

V2_RANKER_WEIGHTS: dict[str, float] = {
    "core_issue_score": 0.22,
    "concept_overlap": 0.14,
    "legal_term_overlap": 0.12,
    "retrieval_score_feature": 0.10,
    "target_object_score": 0.09,
    "qualifier_score": 0.08,
    "actor_score": 0.07,
    "exact_citation_match": 0.06,
    "graph_proximity": 0.05,
    "domain_match": 0.03,
    "temporal_validity": 0.02,
    "structural_fit": 0.02,
    "source_reliability": 0.01,
    "distractor_penalty": -0.15,
    "target_without_core_penalty": -0.12,
    "context_only_penalty": -0.08,
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
    def __init__(self, intent_registry: LegalIntentRegistry | None = None) -> None:
        self.intent_registry = intent_registry or LegalIntentRegistry()

    def rank(
        self,
        *,
        question: str,
        plan: QueryPlan,
        retrieval_response: RawRetrievalResponse,
        graph_expansion: GraphExpansionResult,
        query_frame: QueryFrame | None = None,
        debug: bool = False,
    ) -> LegalRankerResult:
        bundles = self._merge_candidates(
            retrieval_response=retrieval_response,
            graph_expansion=graph_expansion,
        )
        input_candidate_count = len(bundles)
        if not bundles:
            return self._fallback_result(debug=debug)

        use_v2_scoring = self._use_v2_scoring(query_frame)
        if use_v2_scoring:
            scored_rows = [
                self._v2_scored_row(
                    question=question,
                    plan=plan,
                    query_frame=query_frame,
                    bundle=bundle,
                )
                for bundle in bundles
            ]
        else:
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
        scored_rows = [
            (
                bundle,
                score_breakdown,
                rerank_score,
            )
            for bundle, score_breakdown, rerank_score in scored_rows
        ]
        scored_rows.sort(key=self._sort_key)

        ranked_candidates = [
            self._ranked_candidate(
                rank=index,
                bundle=bundle,
                score_breakdown=score_breakdown,
                rerank_score=rerank_score,
                plan=plan,
                use_v2_scoring=use_v2_scoring,
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
                query_frame=query_frame,
                scoring_version=LEGAL_RANKER_V2_SCORING
                if use_v2_scoring
                else LEGAL_RANKER_V1_SCORING,
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
                query_frame=None,
                scoring_version=LEGAL_RANKER_V1_SCORING,
            )
        return result

    def _use_v2_scoring(self, query_frame: QueryFrame | None) -> bool:
        if query_frame is None:
            return False
        return query_frame.confidence >= QUERY_FRAME_CONFIDENCE_THRESHOLD

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

    def _v2_scored_row(
        self,
        *,
        question: str,
        plan: QueryPlan,
        query_frame: QueryFrame,
        bundle: _CandidateBundle,
    ) -> tuple[_CandidateBundle, RankerFeatureBreakdown, float]:
        features = self._extract_v2_features(
            question=question,
            plan=plan,
            query_frame=query_frame,
            bundle=bundle,
        )
        score = self._v2_weighted_score(features, query_frame=query_frame)
        return (
            bundle,
            RankerFeatureBreakdown(**features),
            score,
        )

    def _extract_v2_features(
        self,
        *,
        question: str,
        plan: QueryPlan,
        query_frame: QueryFrame,
        bundle: _CandidateBundle,
    ) -> dict[str, float]:
        unit = bundle.unit or {}
        haystack = self._candidate_text_haystack(bundle)
        bm25_score = self._clamp(
            self._score_from_breakdown(bundle.score_breakdown, RAW_BM25_KEYS)
        )
        dense_score = self._clamp(
            self._score_from_breakdown(bundle.score_breakdown, RAW_DENSE_KEYS)
        )
        exact_citation_match = self._exact_citation_match(plan, bundle)
        core_issue_score = self._core_issue_score(query_frame, bundle, haystack)
        target_object_score = self._term_group_score(
            self._target_terms(query_frame),
            haystack,
        )
        actor_score = self._term_group_score(self._actor_terms(query_frame), haystack)
        qualifier_score = self._term_group_score(
            self._qualifier_terms(query_frame),
            haystack,
        )
        concept_overlap = self._v2_concept_overlap(query_frame, unit, haystack)
        legal_term_overlap = self._v2_legal_term_overlap(query_frame, haystack)
        graph_proximity = self._v2_graph_proximity(bundle)
        domain_match = self._domain_match(plan, unit)
        temporal_validity = self._temporal_validity(unit)
        source_reliability = self._source_reliability(unit)
        structural_fit = self._structural_fit(unit, core_issue_score, haystack)
        distractor_penalty = self._distractor_penalty(query_frame, haystack)
        target_without_core_penalty = (
            target_object_score
            if target_object_score > 0.0 and core_issue_score < 0.25
            else 0.0
        )
        support_role_hint_score = self._support_role_hint_score(
            query_frame=query_frame,
            features={
                "core_issue_score": core_issue_score,
                "target_object_score": target_object_score,
                "distractor_penalty": distractor_penalty,
            },
            haystack=haystack,
        )
        context_only_penalty = self._context_only_penalty(
            query_frame=query_frame,
            unit=unit,
            core_issue_score=core_issue_score,
            target_object_score=target_object_score,
            actor_score=actor_score,
            distractor_penalty=distractor_penalty,
        )
        is_exception = self._indicator(
            unit=unit,
            bundle=bundle,
            terms=(
                "exception",
                "exceptie",
                "except",
                "cu exceptia",
                "derogare",
                "deroga",
                "situatie speciala",
                "exception_to",
            ),
        )

        return {
            "bm25_score": bm25_score,
            "dense_score": dense_score,
            "core_issue_score": core_issue_score,
            "target_object_score": target_object_score,
            "actor_score": actor_score,
            "qualifier_score": qualifier_score,
            "retrieval_score_feature": self._retrieval_score_feature(
                bundle,
                bm25_score=bm25_score,
                dense_score=dense_score,
            ),
            "exact_citation_match": exact_citation_match,
            "domain_match": domain_match,
            "graph_proximity": graph_proximity,
            "concept_overlap": concept_overlap,
            "legal_term_overlap": legal_term_overlap,
            "temporal_validity": temporal_validity,
            "structural_fit": structural_fit,
            "source_reliability": source_reliability,
            "distractor_penalty": distractor_penalty,
            "target_without_core_penalty": target_without_core_penalty,
            "context_only_penalty": context_only_penalty,
            "query_frame_confidence": query_frame.confidence,
            "support_role_hint_score": support_role_hint_score,
            "parent_relevance": self._parent_relevance(unit),
            "is_exception": is_exception,
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
            "gate_domain_mismatch": 0.0,
            "gate_low_core_issue": 0.0,
            "gate_distractor": 0.0,
            "gate_context_role": 0.0,
            "gate_irrelevant_exception": 0.0,
            "gate_core_issue_floor": 0.0,
            "gate_exact_citation_floor": 0.0,
        }

    def _v2_weighted_score(
        self,
        features: dict[str, float],
        *,
        query_frame: QueryFrame,
    ) -> float:
        score = sum(
            weight * features.get(feature_name, 0.0)
            for feature_name, weight in V2_RANKER_WEIGHTS.items()
        )
        final_score = score
        if features["domain_match"] == 0.0:
            final_score = min(final_score, 0.35)
            features["gate_domain_mismatch"] = 1.0
        if (
            query_frame.intents
            and features["core_issue_score"] < 0.25
            and features["exact_citation_match"] == 0.0
        ):
            final_score = min(final_score, 0.55)
            features["gate_low_core_issue"] = 1.0
        if (
            features["distractor_penalty"] >= 0.7
            and features["core_issue_score"] < 0.25
        ):
            final_score = min(final_score, 0.35)
            features["gate_distractor"] = 1.0
        if features["context_only_penalty"] >= 0.7:
            final_score = min(final_score, 0.55)
            features["gate_context_role"] = 1.0
        if features["is_exception"] > 0.0 and not self._asks_for_exception(query_frame):
            final_score = min(final_score, 0.65)
            features["gate_irrelevant_exception"] = 1.0
        if features["core_issue_score"] >= 0.70:
            final_score = max(final_score, 0.75)
            features["gate_core_issue_floor"] = 1.0
        if features["exact_citation_match"] >= 1.0:
            final_score = max(final_score, 0.82)
            features["gate_exact_citation_floor"] = 1.0
        if features["gate_domain_mismatch"] > 0.0:
            final_score = min(final_score, 0.35)
        if features["gate_distractor"] > 0.0:
            final_score = min(final_score, 0.35)
        if features["gate_context_role"] > 0.0:
            final_score = min(final_score, 0.55)
        if features["gate_irrelevant_exception"] > 0.0:
            final_score = min(final_score, 0.65)

        for key, value in list(features.items()):
            features[key] = self._clamp(value)
        return round(self._clamp(final_score), 6)

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
        use_v2_scoring: bool = False,
    ) -> RankedCandidate:
        return RankedCandidate(
            unit_id=bundle.unit_id,
            rank=rank,
            rerank_score=rerank_score,
            retrieval_score=bundle.retrieval_score,
            unit=bundle.unit,
            score_breakdown=score_breakdown,
            why_ranked=self._why_ranked(
                plan,
                bundle,
                score_breakdown,
                use_v2_scoring=use_v2_scoring,
            ),
            source=self._source(bundle),
        )

    def _why_ranked(
        self,
        plan: QueryPlan,
        bundle: _CandidateBundle,
        score_breakdown: RankerFeatureBreakdown,
        use_v2_scoring: bool = False,
    ) -> list[str]:
        reasons: list[str] = []
        if use_v2_scoring:
            if score_breakdown.core_issue_score >= 0.70:
                reasons.append("core_issue_match")
            if score_breakdown.target_object_score > 0:
                reasons.append("target_object_match")
            if score_breakdown.qualifier_score > 0:
                reasons.append("qualifier_match")
            if score_breakdown.actor_score > 0:
                reasons.append("actor_match")
            if score_breakdown.distractor_penalty >= 0.70:
                reasons.append("distractor_penalty")
            if score_breakdown.gate_low_core_issue > 0:
                reasons.append("capped_low_core_issue")
            if score_breakdown.gate_distractor > 0:
                reasons.append("capped_distractor")
            if score_breakdown.gate_context_role > 0:
                reasons.append("capped_context_role")
            if score_breakdown.gate_domain_mismatch > 0:
                reasons.append("capped_domain_mismatch")
            if score_breakdown.gate_core_issue_floor > 0:
                reasons.append("boosted_core_issue")
            if score_breakdown.gate_exact_citation_floor > 0:
                reasons.append("boosted_exact_citation")
            role_hint = self._support_role_hint_from_breakdown(score_breakdown)
            if role_hint:
                reasons.append(f"support_role_hint:{role_hint}")
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

    def _core_issue_score(
        self,
        query_frame: QueryFrame,
        bundle: _CandidateBundle,
        haystack: str,
    ) -> float:
        scores: list[float] = []
        for intent in self._intents_for_frame(query_frame):
            if intent.id == "labor_contract_modification":
                scores.append(
                    self._labor_contract_modification_core_issue_score(haystack)
                )
                continue
            scores.append(
                max(
                    self._core_phrase_score(intent, haystack),
                    self._core_concept_score(intent, bundle, haystack),
                )
            )
        if scores:
            return self._clamp(max(scores))
        if not query_frame.normalized_terms:
            return 0.0
        matched = sum(
            1
            for term in query_frame.normalized_terms
            if self._phrase_present(term, haystack)
        )
        return self._clamp(matched / len(query_frame.normalized_terms))

    def _labor_contract_modification_core_issue_score(self, haystack: str) -> float:
        if (
            "contractul individual de munca poate fi modificat numai prin acordul partilor"
            in haystack
            or "contract individual de munca poate fi modificat numai prin acordul partilor"
            in haystack
        ):
            return 1.0
        if "poate fi modificat numai prin acordul partilor" in haystack:
            return 0.95

        has_contract_modification_scope = (
            "modificarea contractului individual de munca" in haystack
            or "modificare contract individual munca" in haystack
        )
        if has_contract_modification_scope and any(
            marker in haystack
            for marker in (
                "element",
                "elemente",
                "elementele",
                "poate privi",
                "se refera",
                "salariu",
                "salariul",
            )
        ):
            return 0.90
        if has_contract_modification_scope:
            return 0.75

        if (
            "contract individual de munca" in haystack
            or "contractul individual de munca" in haystack
        ) and (
            "poate fi modificat unilateral" in haystack
            or "locul muncii poate fi modificat" in haystack
        ):
            return 0.20
        return 0.0

    def _core_phrase_score(self, intent: LegalIntent, haystack: str) -> float:
        if not intent.core_phrases:
            return 0.0
        exact_matches = [
            phrase
            for phrase in intent.core_phrases
            if self._phrase_present(phrase, haystack)
        ]
        if exact_matches:
            longest = max(len(self._tokenize(phrase)) for phrase in exact_matches)
            if longest >= 5:
                return 1.0
            if longest >= 3:
                return 0.85
            return 0.70
        core_terms = self._dedupe(
            [
                token
                for phrase in intent.core_phrases
                for token in self._tokenize(phrase)
            ]
        )
        if not core_terms:
            return 0.0
        matched = sum(1 for term in core_terms if self._phrase_present(term, haystack))
        overlap = matched / len(core_terms)
        if overlap >= 0.75:
            return 0.85
        if overlap >= 0.50:
            return 0.70
        return self._clamp(overlap * 0.8)

    def _core_concept_score(
        self,
        intent: LegalIntent,
        bundle: _CandidateBundle,
        haystack: str,
    ) -> float:
        if not intent.core_concepts:
            return 0.0
        unit_concepts = self._tokens_from_value(
            (bundle.unit or {}).get("legal_concepts") or []
        )
        matched = 0
        for concept in intent.core_concepts:
            aliases = self._concept_aliases(concept)
            if (
                self._normalize_text(concept) in unit_concepts
                or any(self._normalize_text(alias) in unit_concepts for alias in aliases)
                or any(self._phrase_present(alias, haystack) for alias in aliases)
            ):
                matched += 1
        overlap = matched / len(intent.core_concepts)
        if overlap >= 1.0:
            return 0.8
        if overlap > 0:
            return 0.4 + 0.3 * overlap
        return 0.0

    def _target_terms(self, query_frame: QueryFrame) -> list[str]:
        terms = [
            term
            for intent in self._intents_for_frame(query_frame)
            for term in intent.target_terms
        ]
        for target in query_frame.targets:
            terms.extend(self._concept_aliases(target))
        return self._dedupe(terms)

    def _actor_terms(self, query_frame: QueryFrame) -> list[str]:
        terms = [
            term
            for intent in self._intents_for_frame(query_frame)
            for term in intent.actor_terms
        ]
        for actor in query_frame.actors:
            terms.extend(self._concept_aliases(actor))
        return self._dedupe(terms)

    def _qualifier_terms(self, query_frame: QueryFrame) -> list[str]:
        terms = [
            term
            for intent in self._intents_for_frame(query_frame)
            for term in intent.qualifier_terms
        ]
        for qualifier in query_frame.qualifiers:
            terms.extend(self._concept_aliases(qualifier))
        return self._dedupe(terms)

    def _term_group_score(self, terms: list[str], haystack: str) -> float:
        if not terms:
            return 0.0
        matched = sum(1 for term in terms if self._phrase_present(term, haystack))
        if matched == 0:
            return 0.0
        return self._clamp(matched / min(len(terms), 4))

    def _v2_concept_overlap(
        self,
        query_frame: QueryFrame,
        unit: dict[str, Any],
        haystack: str,
    ) -> float:
        desired: set[str] = set(query_frame.targets + query_frame.actors)
        for intent in self._intents_for_frame(query_frame):
            desired.update(intent.core_concepts)
            desired.update(intent.target_concepts)
            desired.update(intent.actor_concepts)
        if not desired:
            return 0.0
        unit_concepts = self._tokens_from_value(unit.get("legal_concepts") or [])
        matched = 0
        for concept in desired:
            aliases = self._concept_aliases(concept)
            if (
                self._normalize_text(concept) in unit_concepts
                or any(self._normalize_text(alias) in unit_concepts for alias in aliases)
                or any(self._phrase_present(alias, haystack) for alias in aliases)
            ):
                matched += 1
        return self._clamp(matched / len(desired))

    def _v2_legal_term_overlap(self, query_frame: QueryFrame, haystack: str) -> float:
        terms = self._dedupe(query_frame.normalized_terms)
        if not terms:
            return 0.0
        matched = sum(1 for term in terms if self._phrase_present(term, haystack))
        return self._clamp(matched / len(terms))

    def _distractor_penalty(self, query_frame: QueryFrame, haystack: str) -> float:
        terms = [
            term
            for intent in self._intents_for_frame(query_frame)
            for term in intent.distractor_terms
        ]
        if not terms:
            return 0.0
        matched = sum(1 for term in terms if self._phrase_present(term, haystack))
        if matched >= 2:
            return 1.0
        if matched == 1:
            return 0.7
        return 0.0

    def _context_only_penalty(
        self,
        *,
        query_frame: QueryFrame,
        unit: dict[str, Any],
        core_issue_score: float,
        target_object_score: float,
        actor_score: float,
        distractor_penalty: float,
    ) -> float:
        if self._normalize_text(str(unit.get("support_role") or "")) == "context":
            return 1.0
        if distractor_penalty >= 0.7:
            return 1.0
        if core_issue_score >= 0.25:
            return 0.0
        if target_object_score > 0.0 or actor_score > 0.0:
            return 0.8
        if query_frame.intents:
            return 0.4
        return 0.0

    def _retrieval_score_feature(
        self,
        bundle: _CandidateBundle,
        *,
        bm25_score: float,
        dense_score: float,
    ) -> float:
        if bundle.retrieval_score is not None:
            return self._clamp(bundle.retrieval_score)
        return self._clamp(max(bm25_score, dense_score))

    def _v2_graph_proximity(self, bundle: _CandidateBundle) -> float:
        if bundle.expanded_candidate and bundle.graph_proximity > 0:
            return self._clamp(bundle.graph_proximity)
        return 0.0

    def _structural_fit(
        self,
        unit: dict[str, Any],
        core_issue_score: float,
        haystack: str,
    ) -> float:
        if core_issue_score >= 0.70:
            if unit.get("paragraph_number") or unit.get("letter_number"):
                return 1.0
            if unit.get("article_number") or self._unit_type(unit) == "article":
                return 0.8
        if haystack:
            return 0.3
        return 0.0

    def _support_role_hint_score(
        self,
        *,
        query_frame: QueryFrame,
        features: dict[str, float],
        haystack: str,
    ) -> float:
        return 1.0 if self._support_role_hint(query_frame, features, haystack) else 0.0

    def _support_role_hint(
        self,
        query_frame: QueryFrame,
        features: dict[str, float],
        haystack: str,
    ) -> str | None:
        if self._is_labor_contract_modification_frame(query_frame):
            if (
                features["core_issue_score"] >= 0.70
                and features["distractor_penalty"] < 0.5
            ):
                return "direct_basis"
            if (
                features["distractor_penalty"] >= 0.5
                or features["target_object_score"] > 0
                or features.get("qualifier_score", 0.0) > 0
                or features.get("actor_score", 0.0) > 0
            ):
                return "context"
            return None
        if features["distractor_penalty"] >= 0.7:
            return "context"
        if features["core_issue_score"] >= 0.70:
            return "direct_basis"
        if self._phrase_present("definition", haystack) or self._phrase_present("defineste", haystack):
            return "definition"
        if "sanction" in query_frame.meta_intents or self._phrase_present("sanctiune", haystack):
            return "sanction"
        if "procedure" in query_frame.meta_intents or self._phrase_present("procedura", haystack):
            return "procedure"
        if "exception" in query_frame.meta_intents or self._phrase_present("exceptie", haystack):
            return "exception"
        if features["target_object_score"] > 0 and features["core_issue_score"] < 0.25:
            return "context"
        return None

    def _is_labor_contract_modification_frame(self, query_frame: QueryFrame) -> bool:
        return "labor_contract_modification" in query_frame.intents

    def _support_role_hint_from_breakdown(
        self,
        score_breakdown: RankerFeatureBreakdown,
    ) -> str | None:
        if score_breakdown.distractor_penalty >= 0.7:
            return "context"
        if (
            score_breakdown.core_issue_score >= 0.70
            and score_breakdown.distractor_penalty < 0.5
        ):
            return "direct_basis"
        if score_breakdown.is_definition > 0:
            return "definition"
        if score_breakdown.is_sanction > 0:
            return "sanction"
        if score_breakdown.is_exception > 0:
            return "exception"
        if (
            score_breakdown.target_object_score > 0
            and score_breakdown.core_issue_score < 0.25
        ):
            return "context"
        return None

    def _asks_for_exception(self, query_frame: QueryFrame) -> bool:
        return any(
            value in {"exception", "exceptie", "derogation"}
            for value in [*query_frame.meta_intents, *query_frame.intents]
        )

    def _intents_for_frame(self, query_frame: QueryFrame) -> list[LegalIntent]:
        return [
            intent
            for intent_id in query_frame.intents
            if (intent := self.intent_registry.get(intent_id)) is not None
        ]

    def _concept_aliases(self, concept: str) -> list[str]:
        aliases = {
            "salary": ["salary", "salariu", "salariul", "salariului", "salarizare"],
            "employer": ["employer", "angajator", "angajatorul"],
            "employee": ["employee", "salariat", "salariatul"],
            "without_addendum": ["without_addendum", "act aditional", "aditional"],
            "without_agreement": ["without_agreement", "fara acord", "acord", "acordul partilor"],
            "contract_modification": ["contract_modification", "contract", "modificare", "modificat", "modificarea contractului"],
            "agreement_of_parties": ["agreement_of_parties", "acord", "partilor", "acordul partilor"],
            "fine": ["amenda", "amenzii"],
            "prescription": ["prescriptie", "prescrie"],
            "limitation_period": ["termen", "prescriptie"],
            "tax_declaration": ["declaratie", "fiscala"],
            "tax_payment": ["plata", "impozit", "taxa"],
            "dismissal": ["concediere", "preaviz"],
            "working_time": ["program", "timp", "ore"],
            "leave": ["concediu", "concediul"],
        }
        return aliases.get(concept, [concept])

    def _phrase_present(self, phrase: str, haystack: str) -> bool:
        normalized = self._normalize_text(phrase)
        if not normalized:
            return False
        if " " in normalized or "_" in normalized:
            return normalized in haystack
        haystack_tokens = set(haystack.split())
        if normalized in haystack_tokens:
            return True
        if len(normalized) >= 6:
            return any(
                token.startswith(normalized[:6]) or normalized.startswith(token[:6])
                for token in haystack_tokens
                if len(token) >= 6
            )
        return False

    def _candidate_text_haystack(self, bundle: _CandidateBundle) -> str:
        values: list[str] = []
        if bundle.unit:
            for key in (
                "raw_text",
                "normalized_text",
                "text",
                "legal_concepts",
                "why_retrieved",
            ):
                value = bundle.unit.get(key)
                if value:
                    values.append(str(value))
        values.extend(bundle.reasons)
        if bundle.retrieval_candidate and bundle.retrieval_candidate.why_retrieved:
            values.append(bundle.retrieval_candidate.why_retrieved)
        return self._normalize_text(" ".join(values))

    def _unit_type(self, unit: dict[str, Any]) -> str:
        raw_type = self._normalize_text(str(unit.get("type") or unit.get("unit_type") or ""))
        mapping = {
            "articol": "article",
            "article": "article",
            "alineat": "paragraph",
            "paragraph": "paragraph",
            "litera": "letter",
            "letter": "letter",
        }
        return mapping.get(raw_type, raw_type)

    def _dedupe(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            if value and value not in deduped:
                deduped.append(value)
        return deduped

    def _debug_payload(
        self,
        *,
        fallback_used: bool,
        input_candidate_count: int,
        ranked_candidates: list[RankedCandidate],
        warnings: list[str],
        query_frame: QueryFrame | None,
        scoring_version: str,
    ) -> dict[str, Any]:
        rows = [candidate.model_dump(mode="json") for candidate in ranked_candidates]
        return {
            "fallback_used": fallback_used,
            "input_candidate_count": input_candidate_count,
            "ranked_candidate_count": len(ranked_candidates),
            "scoring_version": scoring_version,
            "query_frame": query_frame.model_dump(mode="json") if query_frame else None,
            "weights": V2_RANKER_WEIGHTS
            if scoring_version == LEGAL_RANKER_V2_SCORING
            else RANKER_WEIGHTS,
            "gates_applied": {
                candidate.unit_id: self._gates_applied(candidate.score_breakdown)
                for candidate in ranked_candidates
            },
            "ranked_candidates": rows,
            "rows": rows,
            "warnings": warnings,
        }

    def _gates_applied(self, score_breakdown: RankerFeatureBreakdown) -> list[str]:
        gate_names = [
            "gate_domain_mismatch",
            "gate_low_core_issue",
            "gate_distractor",
            "gate_context_role",
            "gate_irrelevant_exception",
            "gate_core_issue_floor",
            "gate_exact_citation_floor",
        ]
        return [
            name
            for name in gate_names
            if getattr(score_breakdown, name, 0.0) > 0.0
        ]

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
        stripped = re.sub(r"\badi[?\ufffd]ional(a?)\b", r"aditional\1", stripped)
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
