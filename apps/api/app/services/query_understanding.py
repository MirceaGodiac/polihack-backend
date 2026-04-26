import re
import unicodedata

from ..schemas import ExactCitation, QueryPlan, QueryRequest
from .domain_router import DomainRouter
from .exact_citation_detector import ExactCitationDetector


def normalize_ro_text(text: str) -> str:
    text = text.replace("ş", "ș").replace("Ş", "Ș")
    text = text.replace("ţ", "ț").replace("Ţ", "Ț")
    text = re.sub(r"\badi[?\ufffd]ional(a?)\b", r"aditional\1", text)
    return " ".join(text.lower().split())


def _strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _match_text(normalized_question: str) -> str:
    return _strip_diacritics(normalized_question).casefold()


def detect_query_types(normalized_question: str) -> list[str]:
    match_text = _match_text(normalized_question)
    query_types: list[str] = []

    if any(term in match_text for term in ("poate", "are voie")):
        query_types.append("right")
    if any(term in match_text for term in ("fara", "nu are voie", "interzis")):
        query_types.append("prohibition")
    if any(
        term in match_text
        for term in ("trebuie", "obligat", "obligatie", "act aditional")
    ):
        query_types.append("obligation")
    if any(term in match_text for term in ("amenda", "sanctiune", "pedeapsa")):
        query_types.append("sanction")
    if any(
        term in match_text
        for term in ("cum contest", "termen", "procedura", "plangere")
    ):
        query_types.append("procedure")
    if any(
        term in match_text
        for term in ("ce inseamna", "definitie", "este definit")
    ):
        query_types.append("definition")
    if any(term in match_text for term in ("exceptie", "cu exceptia", "exceptat")):
        query_types.append("exception")

    return query_types or ["right"]


def detect_temporal_context(question: str, request_date: str) -> str | None:
    if request_date == "current":
        return "current"
    if request_date:
        return request_date
    normalized_question = normalize_ro_text(question)
    year = _find_year(normalized_question)
    return year


def detect_safety_flags(normalized_question: str) -> list[str]:
    match_text = _match_text(normalized_question)
    abusive_terms = (
        "cum ascund",
        "cum falsific",
        "falsific acte",
        "evit ilegal",
        "frauda",
        "spalare de bani",
    )
    return ["abusive_or_illegal_instruction"] if any(
        term in match_text for term in abusive_terms
    ) else []


def build_retrieval_filters(
    legal_domain: str | None,
    temporal_context: str | None,
    exact_citations: list[ExactCitation],
) -> dict[str, object]:
    filters: dict[str, object] = {}
    if legal_domain:
        filters["legal_domain"] = legal_domain
    if temporal_context == "current":
        filters["status"] = "active"
    if temporal_context:
        filters["date_context"] = temporal_context
    if exact_citations:
        filters["exact_citation_filters"] = [
            citation.lookup_filters for citation in exact_citations
        ]
    return filters


def choose_expansion_policy(
    normalized_question: str,
    query_types: list[str],
    ambiguity_flags: list[str],
) -> dict[str, int | bool]:
    match_text = _match_text(normalized_question)
    asks_permission_or_exception = any(
        term in match_text for term in ("poate", "are voie", "fara")
    )
    return {
        "max_depth": 2,
        "max_expanded_nodes": 80,
        "include_parents": True,
        "include_children": True,
        "include_references": True,
        "include_definitions": "definition" in query_types or bool(ambiguity_flags),
        "include_exceptions": "exception" in query_types
        or asks_permission_or_exception,
        "include_sanctions": "sanction" in query_types,
    }


class QueryUnderstanding:
    def __init__(
        self,
        domain_router: DomainRouter | None = None,
        exact_citation_detector: ExactCitationDetector | None = None,
    ) -> None:
        self.domain_router = domain_router or DomainRouter()
        self.exact_citation_detector = (
            exact_citation_detector or ExactCitationDetector()
        )

    def build_plan(self, request: QueryRequest) -> QueryPlan:
        normalized_question = normalize_ro_text(request.question)
        route = self.domain_router.route(normalized_question)
        query_types = detect_query_types(normalized_question)
        temporal_context = detect_temporal_context(request.question, request.date)
        exact_citations = self._enrich_exact_citations(
            exact_citations=self.exact_citation_detector.detect(request.question),
            temporal_context=temporal_context,
        )
        ambiguity_flags = self._detect_ambiguity_flags(
            normalized_question=normalized_question,
            question=request.question,
            route_flags=route.ambiguity_flags,
            exact_citations=exact_citations,
        )
        safety_flags = detect_safety_flags(normalized_question)
        retrieval_filters = build_retrieval_filters(
            legal_domain=route.legal_domain,
            temporal_context=temporal_context,
            exact_citations=exact_citations,
        )
        expansion_policy = choose_expansion_policy(
            normalized_question=normalized_question,
            query_types=query_types,
            ambiguity_flags=ambiguity_flags,
        )

        return QueryPlan(
            question=request.question,
            normalized_question=normalized_question,
            legal_domain=route.legal_domain,
            domain_confidence=route.domain_confidence,
            domain_scores=route.domain_scores,
            query_types=query_types,
            exact_citations=exact_citations,
            temporal_context=temporal_context,
            ambiguity_flags=ambiguity_flags,
            safety_flags=safety_flags,
            should_refuse_early=bool(safety_flags),
            retrieval_filters=retrieval_filters,
            expansion_policy=expansion_policy,
        )

    def _detect_ambiguity_flags(
        self,
        normalized_question: str,
        question: str,
        route_flags: list[str],
        exact_citations: list[ExactCitation],
    ) -> list[str]:
        flags = list(route_flags)
        if len(normalized_question) < 10 and "too_short_question" not in flags:
            flags.append("too_short_question")
        if _find_year(normalized_question):
            flags.append("ambiguous_temporal_context")
        if any(citation.is_relative for citation in exact_citations):
            flags.append("relative_citation_needs_context")
        if not exact_citations and self._looks_like_exact_citation(question):
            flags.append("exact_citation_detection_pending")
        return flags

    def _looks_like_exact_citation(self, question: str) -> bool:
        match_text = _match_text(normalize_ro_text(question))
        citation_markers = ("art.", "art ", "alin.", "alin ", "lit.")
        return any(marker in match_text for marker in citation_markers)

    def _enrich_exact_citations(
        self,
        exact_citations: list[ExactCitation],
        temporal_context: str | None,
    ) -> list[ExactCitation]:
        enriched: list[ExactCitation] = []
        for citation in exact_citations:
            lookup_filters = dict(citation.lookup_filters)
            if temporal_context == "current":
                lookup_filters["status"] = "active"
            if temporal_context:
                lookup_filters["date_context"] = temporal_context
            enriched.append(citation.model_copy(update={"lookup_filters": lookup_filters}))
        return enriched


def _find_year(normalized_question: str) -> str | None:
    for token in normalized_question.replace(",", " ").replace(".", " ").split():
        if token.isdigit() and len(token) == 4:
            year = int(token)
            if 1900 <= year <= 2099:
                return str(year)
    return None
