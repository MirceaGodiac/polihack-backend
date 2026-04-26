from __future__ import annotations

import re
import unicodedata
from typing import Literal

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
    retrieval_queries: list[str] = Field(default_factory=list)
    required_evidence_concepts: list[str] = Field(default_factory=list)
    decomposition_source: Literal["deterministic", "llm", "merged"] = "deterministic"
    llm_confidence: float | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ambiguity_flags: list[str] = Field(default_factory=list)
    requires_clarification: bool = False


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
            explicit_trigger = any(
                _phrase_in_text(trigger, haystack) for trigger in intent.triggers
            )
            if explicit_trigger:
                matches.append(intent)
                continue
            if intent.domain and plan.legal_domain:
                if normalize_legal_text(intent.domain) != normalize_legal_text(plan.legal_domain):
                    continue
            target_hit = any(
                _phrase_in_text(term, haystack) for term in intent.target_terms
            )
            qualifier_hit = any(
                _phrase_in_text(term, haystack) for term in intent.qualifier_terms
            )
            actor_hit = any(_phrase_in_text(term, haystack) for term in intent.actor_terms)
            issue_hit = any(
                _phrase_in_text(term, haystack)
                for term in [
                    *intent.core_phrases,
                    *intent.qualifier_terms,
                    *intent.triggers,
                ]
            )
            if target_hit and (qualifier_hit or actor_hit) and issue_hit:
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
                    "elementele contractului",
                    "poate fi modificat numai prin acordul partilor",
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
                ],
            ),
            LegalIntent(
                id="labor_salary_payment",
                domain="munca",
                meta_intents=["payment", "obligation", "remedy"],
                triggers=[
                    "plata salariului",
                    "intarzie plata salariului",
                    "intarzierea platii salariului",
                    "neplata salariului",
                    "salariu neplatit",
                    "drepturi salariale restante",
                ],
                core_concepts=["salary_payment", "wage_due"],
                core_phrases=[
                    "plata salariului",
                    "drepturi salariale",
                    "salariul se plateste",
                    "salariu neplatit",
                ],
                target_concepts=["salary"],
                target_terms=["salariu", "salariul", "salariului", "drepturi salariale", "remuneratie"],
                actor_concepts=["employer", "employee"],
                actor_terms=["angajator", "angajatorul", "salariat", "salariatul"],
                qualifier_concepts=["delayed_payment", "unpaid_salary"],
                qualifier_terms=["intarzie", "intarziere", "neplata", "restant", "restante"],
                distractor_terms=[
                    "act aditional",
                    "modificare contract",
                    "confidentialitatea salariului",
                    "salariul minim",
                ],
            ),
            LegalIntent(
                id="labor_dismissal",
                domain="munca",
                meta_intents=["termination", "procedure", "obligation"],
                triggers=[
                    "concediere",
                    "preaviz",
                    "decizie de concediere",
                    "incetarea contractului de munca",
                    "dat afara",
                ],
                core_concepts=["dismissal", "notice"],
                core_phrases=[
                    "concediere",
                    "preaviz",
                    "decizie de concediere",
                    "incetarea contractului individual de munca",
                ],
                target_concepts=["dismissal", "notice"],
                target_terms=["concediere", "preaviz", "incetare", "demitere"],
                actor_concepts=["employer", "employee"],
                actor_terms=["angajator", "angajatorul", "salariat", "salariatul"],
                qualifier_concepts=["before_dismissal", "without_notice"],
                qualifier_terms=["inainte", "fara preaviz", "disciplinar", "colectiva"],
                distractor_terms=["demisie", "pensie", "salariu minim"],
            ),
            LegalIntent(
                id="labor_working_time",
                domain="munca",
                meta_intents=["working_time", "obligation", "limit"],
                triggers=[
                    "ore suplimentare",
                    "program de lucru",
                    "timp de munca",
                    "pontaj",
                    "repaus saptamanal",
                    "munca de noapte",
                ],
                core_concepts=["working_time", "overtime", "rest_period"],
                core_phrases=[
                    "timp de munca",
                    "ore suplimentare",
                    "program de lucru",
                    "repaus",
                    "munca de noapte",
                ],
                target_concepts=["working_time", "overtime"],
                target_terms=["ore", "program", "pontaj", "timp de munca", "repaus"],
                actor_concepts=["employer", "employee"],
                actor_terms=["angajator", "angajatorul", "salariat", "salariatul"],
                qualifier_concepts=["overtime", "night_work", "weekly_rest"],
                qualifier_terms=["suplimentare", "noapte", "saptamanal", "maxim"],
                distractor_terms=["concediu", "concediere", "salariu restant"],
            ),
            LegalIntent(
                id="labor_leave",
                domain="munca",
                meta_intents=["leave", "entitlement", "procedure"],
                triggers=[
                    "concediu de odihna",
                    "concediul de odihna",
                    "concediu medical",
                    "zile libere",
                    "indemnizatie de concediu",
                    "cerere de concediu",
                ],
                core_concepts=["leave", "paid_leave"],
                core_phrases=[
                    "concediu de odihna",
                    "concediu medical",
                    "zile libere",
                    "indemnizatie de concediu",
                ],
                target_concepts=["leave"],
                target_terms=["concediu", "concediul", "zile libere", "indemnizatie"],
                actor_concepts=["employer", "employee"],
                actor_terms=["angajator", "angajatorul", "salariat", "salariatul"],
                qualifier_concepts=["annual_leave", "medical_leave"],
                qualifier_terms=["odihna", "medical", "neefectuat", "aprobare"],
                distractor_terms=["concediere", "preaviz", "ore suplimentare"],
            ),
            LegalIntent(
                id="contravention_sanction",
                domain="contraventional",
                meta_intents=["sanction", "liability"],
                triggers=[
                    "sanctiune contraventionala",
                    "amenda pentru contraventie",
                    "ce amenda",
                    "proces verbal de contraventie",
                ],
                core_concepts=["contravention", "fine", "sanction"],
                core_phrases=["contraventie", "sanctiune contraventionala", "amenda"],
                target_concepts=["fine", "sanction"],
                target_terms=["amenda", "sanctiune", "contraventie"],
                actor_concepts=["offender", "public_authority"],
                actor_terms=["contravenient", "agent constatator", "autoritate"],
                qualifier_concepts=["contravention"],
                qualifier_terms=["contraventional", "proces verbal"],
                distractor_terms=["impozit", "salariu", "contract civil"],
            ),
            LegalIntent(
                id="contravention_challenge",
                domain="contraventional",
                meta_intents=["procedure", "challenge", "remedy"],
                triggers=[
                    "contest amenda",
                    "contesta amenda",
                    "cum contest",
                    "plangere contraventionala",
                    "anulare proces verbal",
                    "contest procesul verbal",
                ],
                core_concepts=["contravention_challenge", "complaint"],
                core_phrases=[
                    "plangere contraventionala",
                    "contestare amenda",
                    "anulare proces verbal",
                    "proces verbal contraventional",
                ],
                target_concepts=["fine", "report"],
                target_terms=["amenda", "proces verbal", "sanctiune"],
                actor_concepts=["offender", "public_authority"],
                actor_terms=["contravenient", "agent constatator", "autoritate"],
                qualifier_concepts=["challenge", "deadline"],
                qualifier_terms=["contest", "contesta", "plangere", "termen"],
                distractor_terms=["plata impozit", "salariu", "concediere"],
            ),
            LegalIntent(
                id="contravention_payment_deadline",
                domain="contraventional",
                meta_intents=["payment", "deadline", "procedure"],
                triggers=[
                    "termen plata amenda",
                    "termenul de plata a amenzii",
                    "platesc amenda",
                    "achit amenda",
                    "plata amenzii",
                    "jumatate din minim",
                    "plata amenzii contraventionale",
                ],
                core_concepts=["fine_payment", "deadline"],
                core_phrases=[
                    "plata amenzii",
                    "termen de plata",
                    "achitarea amenzii",
                    "jumatate din minim",
                ],
                target_concepts=["fine", "payment_deadline"],
                target_terms=["amenda", "amenzii", "plata", "termen", "achitare"],
                actor_concepts=["offender", "public_authority"],
                actor_terms=["contravenient", "autoritate", "trezorerie"],
                qualifier_concepts=["payment_deadline", "reduced_payment"],
                qualifier_terms=["termen", "jumatate", "minim", "rapid"],
                distractor_terms=["contest", "plangere", "salariu"],
            ),
            LegalIntent(
                id="civil_contract_validity",
                domain="civil",
                meta_intents=["validity", "condition"],
                triggers=[
                    "contract valabil",
                    "validitatea contractului",
                    "nulitate contract",
                    "consimtamant",
                    "capacitate de a contracta",
                ],
                core_concepts=["contract_validity", "consent", "capacity"],
                core_phrases=[
                    "validitatea contractului",
                    "contract valabil",
                    "nulitate",
                    "consimtamant",
                    "capacitate",
                ],
                target_concepts=["contract"],
                target_terms=["contract", "act juridic", "clauza"],
                actor_concepts=["party", "creditor", "debtor"],
                actor_terms=["parte", "parti", "creditor", "debitor"],
                qualifier_concepts=["validity", "nullity"],
                qualifier_terms=["valabil", "nulitate", "anulare", "consimtamant"],
                distractor_terms=["concediere", "amenda contraventionala", "anaf"],
            ),
            LegalIntent(
                id="civil_liability",
                domain="civil",
                meta_intents=["liability", "remedy", "damages"],
                triggers=[
                    "raspundere civila",
                    "daune",
                    "prejudiciu",
                    "despagubiri",
                    "vina",
                ],
                core_concepts=["civil_liability", "damage", "fault"],
                core_phrases=[
                    "raspundere civila",
                    "prejudiciu",
                    "despagubiri",
                    "repararea prejudiciului",
                ],
                target_concepts=["damages", "liability"],
                target_terms=["daune", "prejudiciu", "despagubiri", "raspundere"],
                actor_concepts=["injured_party", "liable_party"],
                actor_terms=["persoana vatamata", "autor", "debitor", "creditor"],
                qualifier_concepts=["fault", "causation"],
                qualifier_terms=["vina", "culpa", "cauzalitate", "fapta"],
                distractor_terms=["salariu", "amenda", "impozit"],
            ),
            LegalIntent(
                id="civil_prescription",
                domain="civil",
                meta_intents=["limitation_period", "deadline"],
                triggers=[
                    "se prescrie",
                    "prescriptie",
                    "dreptul la actiune",
                    "termen de prescriptie",
                    "datorie civila",
                ],
                core_concepts=["prescription", "limitation_period"],
                core_phrases=[
                    "prescriptie",
                    "dreptul la actiune",
                    "termen de prescriptie",
                    "datorie civila",
                ],
                target_concepts=["claim", "debt"],
                target_terms=["actiune", "drept", "datorie", "creanta"],
                actor_concepts=["creditor", "debtor"],
                actor_terms=["creditor", "debitor", "parte"],
                qualifier_concepts=["limitation_period"],
                qualifier_terms=["termen", "cat timp", "in cat timp", "prescrie"],
                distractor_terms=["concediere", "amenda", "declaratie fiscala"],
            ),
            LegalIntent(
                id="tax_payment_obligation",
                domain="fiscal",
                meta_intents=["payment", "obligation"],
                triggers=[
                    "plata impozit",
                    "plata taxei",
                    "cand platesc impozitul",
                    "obligatie de plata fiscala",
                    "taxe si impozite",
                ],
                core_concepts=["tax_payment", "tax_obligation"],
                core_phrases=[
                    "plata impozitului",
                    "obligatie fiscala",
                    "obligatie de plata",
                    "taxe si impozite",
                ],
                target_concepts=["tax", "payment"],
                target_terms=["impozit", "taxa", "tva", "contributii"],
                actor_concepts=["taxpayer", "tax_authority"],
                actor_terms=["contribuabil", "anaf", "organ fiscal"],
                qualifier_concepts=["due_date", "tax_obligation"],
                qualifier_terms=["scadenta", "termen", "datorat", "de plata"],
                distractor_terms=["salariu", "concediere", "amenda contraventionala"],
            ),
            LegalIntent(
                id="tax_declaration_obligation",
                domain="fiscal",
                meta_intents=["procedure", "obligation"],
                triggers=[
                    "declaratie fiscala",
                    "declaratia fiscala",
                    "depusa declaratia",
                    "depun declaratia",
                    "formular fiscal",
                ],
                core_concepts=["tax_declaration", "filing_obligation"],
                core_phrases=[
                    "declaratie fiscala",
                    "depunerea declaratiei",
                    "formular fiscal",
                    "anaf",
                ],
                target_concepts=["tax_declaration"],
                target_terms=["declaratie", "formular", "anaf", "fiscal"],
                actor_concepts=["taxpayer", "tax_authority"],
                actor_terms=["contribuabil", "anaf", "organ fiscal"],
                qualifier_concepts=["filing_deadline"],
                qualifier_terms=["cand", "termen", "depusa", "depunere"],
                distractor_terms=["plata salariului", "concediu", "contraventie"],
            ),
            LegalIntent(
                id="tax_penalty_interest",
                domain="fiscal",
                meta_intents=["sanction", "interest", "late_payment"],
                triggers=[
                    "dobanzi penalitati",
                    "penalitati fiscale",
                    "dobanda fiscala",
                    "majorari intarziere",
                    "intarziere plata impozit",
                ],
                core_concepts=["tax_penalty", "interest", "late_payment"],
                core_phrases=[
                    "penalitati fiscale",
                    "dobanzi",
                    "majorari de intarziere",
                    "intarziere la plata",
                ],
                target_concepts=["penalty", "interest"],
                target_terms=["penalitati", "dobanzi", "majorari", "intarziere"],
                actor_concepts=["taxpayer", "tax_authority"],
                actor_terms=["contribuabil", "anaf", "organ fiscal"],
                qualifier_concepts=["late_payment"],
                qualifier_terms=["intarziere", "neplata", "scadenta"],
                distractor_terms=["preaviz", "concediere", "amenda contraventionala"],
            ),
        ]


class QueryFrameBuilder:
    def __init__(self, registry: LegalIntentRegistry | None = None) -> None:
        self.registry = registry or LegalIntentRegistry()

    def build(self, *, question: str, plan: QueryPlan) -> QueryFrame:
        matched_intents = self.registry.matching_intents(question=question, plan=plan)
        matched_intents = self._ordered_intents(question, plan, matched_intents)
        domain = self._resolved_domain(plan, matched_intents)
        generic_labor = (
            not matched_intents
            and normalize_legal_text(str(domain or "")) == "munca"
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
        ambiguity_flags = self._ambiguity_flags(
            question=question,
            plan=plan,
            intents=matched_intents,
            domain=domain,
        )
        confidence = self._confidence(
            question=question,
            plan=plan,
            intents=matched_intents,
            domain=domain,
            targets=targets,
            actors=actors,
            qualifiers=qualifiers,
            generic_labor=generic_labor,
        )

        return QueryFrame(
            domain=domain,
            intents=self._dedupe(intents),
            meta_intents=self._dedupe(meta_intents),
            targets=self._dedupe(targets),
            actors=self._dedupe(actors),
            qualifiers=self._dedupe(qualifiers),
            surface_phrases=self._dedupe(surface_phrases),
            normalized_terms=self._dedupe(normalized_terms),
            confidence=confidence,
            ambiguity_flags=ambiguity_flags,
            requires_clarification=self._requires_clarification(
                confidence=confidence,
                ambiguity_flags=ambiguity_flags,
            ),
        )

    def _resolved_domain(
        self,
        plan: QueryPlan,
        intents: list[LegalIntent],
    ) -> str | None:
        if plan.legal_domain:
            return plan.legal_domain
        intent_domains = {
            intent.domain for intent in intents if intent.domain
        }
        if len(intent_domains) == 1:
            return next(iter(intent_domains))
        return None

    def _ordered_intents(
        self,
        question: str,
        plan: QueryPlan,
        intents: list[LegalIntent],
    ) -> list[LegalIntent]:
        return sorted(
            intents,
            key=lambda intent: (
                not self._intent_has_trigger(intent, question, plan),
                intent.id,
            ),
        )

    def _ambiguity_flags(
        self,
        *,
        question: str,
        plan: QueryPlan,
        intents: list[LegalIntent],
        domain: str | None,
    ) -> list[str]:
        flags = list(plan.ambiguity_flags)
        tokens = self._tokens(question)
        intent_domains = {
            intent.domain for intent in intents if intent.domain
        }
        if len(tokens) < 3:
            flags.append("too_short")
        if not domain and not intents:
            flags.append("unknown_legal_domain")
        if len(intent_domains) > 1:
            flags.append("multiple_possible_domains")
        if len(intents) > 1:
            flags.append("multiple_possible_intents")
        return self._dedupe(flags)

    def _confidence(
        self,
        *,
        question: str,
        plan: QueryPlan,
        intents: list[LegalIntent],
        domain: str | None,
        targets: list[str],
        actors: list[str],
        qualifiers: list[str],
        generic_labor: bool,
    ) -> float:
        if not intents:
            if generic_labor or domain:
                return 0.40
            return 0.20 if len(self._tokens(question)) >= 3 else 0.10

        explicit_trigger = any(
            self._intent_has_trigger(intent, question, plan) for intent in intents
        )
        domain_match = bool(
            domain
            and any(intent.domain == domain for intent in intents if intent.domain)
        )
        if explicit_trigger and domain_match:
            return 0.90
        if (targets and (actors or qualifiers)) and domain_match:
            return 0.75
        if explicit_trigger:
            return 0.60
        if domain_match:
            return 0.55
        return 0.30

    def _requires_clarification(
        self,
        *,
        confidence: float,
        ambiguity_flags: list[str],
    ) -> bool:
        if "too_short" in ambiguity_flags:
            return True
        if confidence < 0.35:
            return True
        return "unknown_legal_domain" in ambiguity_flags

    def _intent_has_trigger(
        self,
        intent: LegalIntent,
        question: str,
        plan: QueryPlan,
    ) -> bool:
        haystack = normalize_legal_text(
            " ".join([question, plan.normalized_question])
        )
        return any(_phrase_in_text(trigger, haystack) for trigger in intent.triggers)

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
                terms = [*term_values, *self._concept_terms(concept)]
                if any(_phrase_in_text(term, haystack) for term in terms):
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
            "salary": ["salariu", "salariul", "salariului", "salarizare"],
            "employer": ["angajator", "angajatorul"],
            "employee": ["salariat", "salariatul"],
            "without_addendum": ["act", "aditional"],
            "without_agreement": ["acord", "parti"],
            "contract_modification": ["modificare", "modificat", "contract"],
            "agreement_of_parties": ["acord", "partilor"],
            "delayed_payment": ["intarzie", "intarziere", "restant"],
            "unpaid_salary": ["neplata", "neplatit", "restant"],
            "dismissal": ["concediere", "concediat", "demitere"],
            "notice": ["preaviz"],
            "before_dismissal": ["inainte", "preaviz"],
            "without_notice": ["fara", "preaviz"],
            "working_time": ["program", "timp", "munca", "pontaj"],
            "overtime": ["ore", "suplimentare"],
            "rest_period": ["repaus"],
            "night_work": ["noapte"],
            "weekly_rest": ["saptamanal", "repaus"],
            "leave": ["concediu", "concediul", "zile", "libere"],
            "paid_leave": ["concediu", "concediul", "indemnizatie"],
            "annual_leave": ["odihna"],
            "medical_leave": ["medical"],
            "contravention": ["contraventie", "contraventional"],
            "contravention_challenge": ["contest", "contesta", "plangere"],
            "fine": ["amenda"],
            "sanction": ["sanctiune"],
            "offender": ["contravenient"],
            "public_authority": ["autoritate", "agent", "constatator"],
            "report": ["proces", "verbal"],
            "complaint": ["plangere"],
            "challenge": ["contest", "contesta"],
            "deadline": ["termen"],
            "fine_payment": ["plata", "amenda"],
            "payment_deadline": ["termen", "plata"],
            "reduced_payment": ["jumatate", "minim"],
            "contract_validity": ["contract", "valabil", "validitate"],
            "consent": ["consimtamant"],
            "capacity": ["capacitate"],
            "contract": ["contract"],
            "party": ["parte", "parti"],
            "creditor": ["creditor"],
            "debtor": ["debitor"],
            "validity": ["valabil", "validitate"],
            "nullity": ["nulitate", "anulare"],
            "civil_liability": ["raspundere", "civila"],
            "damage": ["prejudiciu", "daune"],
            "fault": ["vina", "culpa"],
            "damages": ["daune", "despagubiri"],
            "liability": ["raspundere"],
            "injured_party": ["persoana", "vatamata"],
            "liable_party": ["autor", "debitor"],
            "causation": ["cauzalitate"],
            "prescription": ["prescriptie", "prescrie"],
            "limitation_period": ["termen", "prescriptie"],
            "claim": ["actiune", "drept"],
            "debt": ["datorie", "creanta"],
            "tax_payment": ["plata", "impozit", "taxa"],
            "tax_obligation": ["obligatie", "fiscala"],
            "tax": ["impozit", "taxa", "tva"],
            "payment": ["plata"],
            "taxpayer": ["contribuabil"],
            "tax_authority": ["anaf", "organ", "fiscal"],
            "due_date": ["scadenta", "termen"],
            "tax_declaration": ["declaratie", "fiscala"],
            "filing_obligation": ["depunere", "declaratie"],
            "filing_deadline": ["termen", "depunere", "cand"],
            "tax_penalty": ["penalitati", "fiscale"],
            "interest": ["dobanzi", "majorari"],
            "penalty": ["penalitati"],
            "late_payment": ["intarziere", "neplata"],
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
    stripped = re.sub(r"\badi[?\ufffd]ional(a?)\b", r"aditional\1", stripped)
    return " ".join(stripped.replace(".", " ").replace("-", "_").split())


def _phrase_in_text(phrase: str, text: str) -> bool:
    normalized_phrase = normalize_legal_text(phrase)
    if " " in normalized_phrase:
        return normalized_phrase in text
    return normalized_phrase in set(re.split(r"[^a-z0-9_]+", text))
