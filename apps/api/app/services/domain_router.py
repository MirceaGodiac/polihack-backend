from dataclasses import dataclass
import unicodedata

DOMAIN_THRESHOLD = 0.35
AMBIGUITY_MARGIN = 0.15
SCORE_NORMALIZER = 2.0


@dataclass(frozen=True)
class DomainRoute:
    legal_domain: str | None
    domain_confidence: float
    domain_scores: dict[str, float]
    ambiguity_flags: list[str]


def strip_ro_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


class DomainRouter:
    lexicon: dict[str, tuple[str, ...]] = {
        "muncă": (
            "angajator",
            "salariat",
            "salariu",
            "contract individual de muncă",
            "contract de muncă",
            "codul muncii",
            "act adițional",
            "concediu",
            "demisie",
            "concediere",
            "ore suplimentare",
            "pontaj",
            "preaviz",
        ),
        "civil": (
            "contract civil",
            "codul civil",
            "codul de procedură civilă",
            "proprietate",
            "moștenire",
            "succesiune",
            "daune",
            "obligație civilă",
            "creditor",
            "debitor",
            "posesie",
        ),
        "fiscal": (
            "impozit",
            "taxă",
            "TVA",
            "ANAF",
            "declarație fiscală",
            "contribuții",
            "cod fiscal",
            "codul fiscal",
        ),
        "contravențional": (
            "amendă",
            "contravenție",
            "proces-verbal",
            "sancțiune contravențională",
            "plângere contravențională",
        ),
        "protecția datelor": (
            "date personale",
            "GDPR",
            "operator",
            "persoană vizată",
            "consimțământ",
            "DPO",
            "ANSPDCP",
        ),
        "societăți comerciale": (
            "SRL",
            "asociat",
            "administrator",
            "dividende",
            "firmă",
            "registrul comerțului",
            "capital social",
        ),
        "consumator": (
            "consumator",
            "comerciant",
            "garanție",
            "retur",
            "ANPC",
            "clauză abuzivă",
        ),
        "administrativ": (
            "autoritate publică",
            "primărie",
            "HCL",
            "act administrativ",
            "contencios administrativ",
            "plângere prealabilă",
        ),
        "penal": (
            "infracțiune",
            "pedeapsă",
            "plângere penală",
            "urmărire penală",
            "cod penal",
            "codul penal",
            "codul de procedură penală",
        ),
    }

    def route(self, normalized_question: str) -> DomainRoute:
        match_text = strip_ro_diacritics(normalized_question).casefold()
        scores = {
            domain: self._score_domain(match_text, terms)
            for domain, terms in self.lexicon.items()
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_domain, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        legal_domain = best_domain if best_score >= DOMAIN_THRESHOLD else None
        ambiguity_flags: list[str] = []
        if legal_domain is None:
            ambiguity_flags.append("low_domain_confidence")
        if best_score >= DOMAIN_THRESHOLD and second_score > 0:
            if best_score - second_score <= AMBIGUITY_MARGIN:
                ambiguity_flags.append("multiple_possible_domains")

        return DomainRoute(
            legal_domain=legal_domain,
            domain_confidence=best_score if legal_domain else 0.0,
            domain_scores=scores,
            ambiguity_flags=ambiguity_flags,
        )

    def _score_domain(self, match_text: str, terms: tuple[str, ...]) -> float:
        raw_score = 0.0
        for term in terms:
            normalized_term = strip_ro_diacritics(term).casefold()
            if normalized_term in match_text:
                raw_score += 1.0
        return round(min(1.0, raw_score / SCORE_NORMALIZER), 2)
