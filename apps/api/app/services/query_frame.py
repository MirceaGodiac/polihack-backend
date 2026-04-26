from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field

from ..schemas import QueryPlan


class QueryFrame(BaseModel):
    domain: str | None = None
    intents: list[str] = Field(default_factory=list)
    meta_intents: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)
    qualifiers: list[str] = Field(default_factory=list)
    surface_phrases: list[str] = Field(default_factory=list)
    normalized_terms: list[str] = Field(default_factory=list)


class LegalIntent(BaseModel):
    id: str
    domain: str | None = None
    meta_intents: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    core_concepts: list[str] = Field(default_factory=list)
    core_phrases: list[str] = Field(default_factory=list)
    target_concepts: list[str] = Field(default_factory=list)
    target_terms: list[str] = Field(default_factory=list)
    actor_concepts: list[str] = Field(default_factory=list)
    actor_terms: list[str] = Field(default_factory=list)
    qualifier_concepts: list[str] = Field(default_factory=list)
    qualifier_terms: list[str] = Field(default_factory=list)
    distractor_terms: list[str] = Field(default_factory=list)


class LegalIntentRegistry:
    def __init__(self, intents: list[LegalIntent] | None = None) -> None:
        self._intents = {intent.id: intent for intent in (intents or self._defaults())}

    def get(self, intent_id: str) -> LegalIntent | None:
        return self._intents.get(intent_id)

    def all(self) -> list[LegalIntent]:
        return list(self._intents.values())

    def matching_intents(self, *, question: str, plan: QueryPlan) -> list[LegalIntent]:
        haystack = normalize_legal_text(
            " ".join(
                [
                    question,
                    plan.normalized_question,
                    " ".join(plan.query_types),
                    str(plan.retrieval_filters),
                ]
            )
        )
        matches: list[LegalIntent] = []
        for intent in self.all():
            if intent.domain and plan.legal_domain:
                if normalize_legal_text(intent.domain) != normalize_legal_text(plan.legal_domain):
                    continue
            if any(_phrase_in_text(trigger, haystack) for trigger in intent.triggers):
                matches.append(intent)
                continue
            target_hit = any(
                _phrase_in_text(term, haystack) for term in intent.target_terms
            )
            qualifier_hit = any(
                _phrase_in_text(term, haystack) for term in intent.qualifier_terms
            )
            actor_hit = any(_phrase_in_text(term, haystack) for term in intent.actor_terms)
            if target_hit and (qualifier_hit or actor_hit):
                matches.append(intent)
        return matches

    def _defaults(self) -> list[LegalIntent]:
        return [
            LegalIntent(
                id="labor_contract_modification",
                domain="munca",
                meta_intents=["modification", "permission"],
                triggers=[
                    "act aditional",
                    "scada salariul",
                    "scade salariul",
                    "modificare contract",
                    "schimbare contract",
                ],
                core_concepts=[
                    "contract_modification",
                    "agreement_of_parties",
                ],
                core_phrases=[
                    "contractul individual de munca poate fi modificat numai prin acordul partilor",
                    "modificarea contractului individual de munca",
                    "poate fi modificat numai prin acordul partilor",
                    "acordul partilor",
                ],
                target_concepts=["salary"],
                target_terms=[
                    "salariu",
                    "salariul",
                    "salarizare",
                ],
                actor_concepts=[
                    "employer",
                    "employee",
                ],
                actor_terms=[
                    "angajator",
                    "angajatorul",
                    "salariat",
                    "salariatul",
                ],
                qualifier_concepts=[
                    "without_addendum",
                    "without_agreement",
                ],
                qualifier_terms=[
                    "fara act aditional",
                    "fara acord",
                    "fara acordul partilor",
                    "unilateral",
                ],
                distractor_terms=[
                    "remuneratie restanta",
                    "persoane angajate ilegal",
                    "neplata salariului",
                    "intarzierea platii salariului",
                    "salariul de baza",
                    "informarea salariatului",
                    "persoana selectata in vederea angajarii",
                    "salariul minim",
                    "salariul este confidential",
                    "confidentialitatea salariului",
                    "registrul general de evidenta",
                ],
            )
        ]


class QueryFrameBuilder:
    def __init__(self, registry: LegalIntentRegistry | None = None) -> None:
        self.registry = registry or LegalIntentRegistry()

    def build(self, *, question: str, plan: QueryPlan) -> QueryFrame:
        matched_intents = self.registry.matching_intents(question=question, plan=plan)
        generic_labor = (
            not matched_intents
            and normalize_legal_text(str(plan.legal_domain or "")) == "munca"
        )
        intents = [intent.id for intent in matched_intents]
        if generic_labor:
            intents.append("generic_labor_query")

        meta_intents = self._meta_intents(question, plan, matched_intents)
        targets = self._matched_concepts(question, matched_intents, "target")
        actors = self._matched_concepts(question, matched_intents, "actor")
        qualifiers = self._matched_concepts(question, matched_intents, "qualifier")
        surface_phrases = self._surface_phrases(question, matched_intents)
        normalized_terms = self._normalized_terms(
            question=question,
            plan=plan,
            intents=matched_intents,
            targets=targets,
            actors=actors,
            qualifiers=qualifiers,
        )

        return QueryFrame(
            domain=plan.legal_domain,
            intents=self._dedupe(intents),
            meta_intents=self._dedupe(meta_intents),
            targets=self._dedupe(targets),
            actors=self._dedupe(actors),
            qualifiers=self._dedupe(qualifiers),
            surface_phrases=self._dedupe(surface_phrases),
            normalized_terms=self._dedupe(normalized_terms),
        )

    def _meta_intents(
        self,
        question: str,
        plan: QueryPlan,
        intents: list[LegalIntent],
    ) -> list[str]:
        haystack = normalize_legal_text(f"{question} {plan.normalized_question}")
        values = [meta for intent in intents for meta in intent.meta_intents]
        if any(term in haystack for term in ("poate", "are voie", "permis")):
            values.append("permission")
        if any(term in haystack for term in ("modificare", "modifica", "schimbare")):
            values.append("modification")
        values.extend(plan.query_types)
        return values

    def _matched_concepts(
        self,
        question: str,
        intents: list[LegalIntent],
        kind: str,
    ) -> list[str]:
        haystack = normalize_legal_text(question)
        concepts: list[str] = []
        for intent in intents:
            concept_values = getattr(intent, f"{kind}_concepts")
            term_values = getattr(intent, f"{kind}_terms")
            for concept in concept_values:
                if any(_phrase_in_text(term, haystack) for term in term_values):
                    concepts.append(concept)
        if "without_addendum" in concepts and "without_agreement" not in concepts:
            concepts.append("without_agreement")
        return concepts

    def _surface_phrases(
        self,
        question: str,
        intents: list[LegalIntent],
    ) -> list[str]:
        haystack = normalize_legal_text(question)
        phrases: list[str] = []
        for intent in intents:
            for phrase in [*intent.triggers, *intent.target_terms, *intent.qualifier_terms]:
                if _phrase_in_text(phrase, haystack):
                    phrases.append(normalize_legal_text(phrase))
        return phrases

    def _normalized_terms(
        self,
        *,
        question: str,
        plan: QueryPlan,
        intents: list[LegalIntent],
        targets: list[str],
        actors: list[str],
        qualifiers: list[str],
    ) -> list[str]:
        terms = list(self._tokens(question))
        terms.extend(self._tokens(plan.normalized_question))
        for intent in intents:
            terms.extend(self._tokens(" ".join(intent.core_phrases)))
            for concept in intent.core_concepts:
                terms.extend(self._concept_terms(concept))
            terms.extend(self._tokens(" ".join(intent.target_terms)))
            terms.extend(self._tokens(" ".join(intent.actor_terms)))
            terms.extend(self._tokens(" ".join(intent.qualifier_terms)))
        for concept in [*targets, *actors, *qualifiers]:
            terms.extend(self._concept_terms(concept))
        return [term for term in terms if len(term) > 2]

    def _concept_terms(self, concept: str) -> list[str]:
        aliases = {
            "salary": ["salariu", "salariul", "salarizare"],
            "employer": ["angajator", "angajatorul"],
            "employee": ["salariat", "salariatul"],
            "without_addendum": ["act", "aditional"],
            "without_agreement": ["acord", "parti"],
            "contract_modification": ["modificare", "modificat", "contract"],
            "agreement_of_parties": ["acord", "partilor"],
        }
        return aliases.get(concept, [normalize_legal_text(concept)])

    def _tokens(self, text: str) -> list[str]:
        return [
            token
            for token in re.split(r"[^a-z0-9_]+", normalize_legal_text(text))
            if token
        ]

    def _dedupe(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            if value and value not in deduped:
                deduped.append(value)
        return deduped


def normalize_legal_text(text: str) -> str:
    replacements = {
        "\u0103": "a",
        "\u0102": "a",
        "\u0219": "s",
        "\u0218": "s",
        "\u021b": "t",
        "\u021a": "t",
        "\u00e2": "a",
        "\u00c2": "a",
        "\u00ee": "i",
        "\u00ce": "i",
    }
    for broken, fixed in replacements.items():
        text = text.replace(broken, fixed)
    normalized = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return " ".join(stripped.replace(".", " ").replace("-", "_").split())


def _phrase_in_text(phrase: str, text: str) -> bool:
    normalized_phrase = normalize_legal_text(phrase)
    if " " in normalized_phrase:
        return normalized_phrase in text
    return normalized_phrase in set(re.split(r"[^a-z0-9_]+", text))
