import re
import unicodedata

from ..schemas import ExactCitation

ARTICLE_RE = re.compile(r"\b(?:art\.?|articolul)\s*(?P<article>\d+(?:\^\d+)?)")
PARAGRAPH_RE = re.compile(r"\b(?:alin\.?|alineatul)\s*\(?\s*(?P<paragraph>\d+)\s*\)?")
LETTER_RE = re.compile(r"\b(?:lit\.?|litera)\s*(?P<letter>[a-z])\s*\)?")
POINT_RE = re.compile(r"\b(?:pct\.?|punctul)\s*(?P<point>\d+)")
THESIS_RE = re.compile(
    r"\bteza\s+(?P<thesis>(?:[a-z]\s+)?(?:i{1,3}|iv|v|vi{0,3}|ix|x)(?:-a)?)"
)
LAW_RE = re.compile(r"\blegea\s+nr\.?\s*(?P<number>\d+)\s*/\s*(?P<year>\d{4})")
OUG_RE = re.compile(
    r"\b(?:o\s*\.?\s*u\s*\.?\s*g\.?|oug)\s+nr\.?\s*(?P<number>\d+)\s*/\s*(?P<year>\d{4})"
)
OG_RE = re.compile(
    r"\b(?:o\s*\.?\s*g\.?|og)\s+nr\.?\s*(?P<number>\d+)\s*/\s*(?P<year>\d{4})"
)
HG_RE = re.compile(
    r"\b(?:h\s*\.?\s*g\.?|hg)\s+nr\.?\s*(?P<number>\d+)\s*/\s*(?P<year>\d{4})"
)
RELATIVE_RE = re.compile(r"\b(?:prezenta lege|prezentul cod|prezentul act normativ)\b")

COMPOUND_WINDOW = 180

NAMED_CODE_ALIASES: tuple[tuple[str, str, str], ...] = (
    ("codul de procedura penala", "Codul de procedură penală", "ro.codul_procedura_penala"),
    ("codul de procedura civila", "Codul de procedură civilă", "ro.codul_procedura_civila"),
    ("codul muncii", "Codul muncii", "ro.codul_muncii"),
    ("codul civil", "Codul civil", "ro.codul_civil"),
    ("codul fiscal", "Codul fiscal", "ro.codul_fiscal"),
    ("codul penal", "Codul penal", "ro.codul_penal"),
)

NUMBERED_ACT_ALIASES: dict[tuple[str, str, str], str] = {
    ("lege", "53", "2003"): "ro.codul_muncii",
    ("og", "2", "2001"): "ro.og_2_2001",
    ("oug", "57", "2019"): "ro.oug_57_2019",
}


class ExactCitationDetector:
    def detect(self, question: str) -> list[ExactCitation]:
        match_text = self._match_text(question)
        citations: list[ExactCitation] = []
        occupied: list[tuple[int, int]] = []

        for article_match in ARTICLE_RE.finditer(match_text):
            citation = self._build_article_citation(
                question=question,
                match_text=match_text,
                article_match=article_match,
            )
            citations.append(citation)
            occupied.append((citation.span_start or 0, citation.span_end or 0))

        for citation in self._detect_numbered_acts(question, match_text, occupied):
            citations.append(citation)
            occupied.append((citation.span_start or 0, citation.span_end or 0))

        for citation in self._detect_named_codes(question, match_text, occupied):
            citations.append(citation)
            occupied.append((citation.span_start or 0, citation.span_end or 0))

        for citation in self._detect_relative_references(question, match_text, occupied):
            citations.append(citation)
            occupied.append((citation.span_start or 0, citation.span_end or 0))

        for citation in self._detect_relative_components(question, match_text, occupied):
            citations.append(citation)
            occupied.append((citation.span_start or 0, citation.span_end or 0))

        return sorted(citations, key=lambda citation: citation.span_start or 0)

    def _build_article_citation(
        self,
        question: str,
        match_text: str,
        article_match: re.Match[str],
    ) -> ExactCitation:
        article = article_match.group("article")
        window_start = article_match.end()
        window_end = min(len(match_text), window_start + COMPOUND_WINDOW)
        window = match_text[window_start:window_end]

        paragraph_match = PARAGRAPH_RE.search(window)
        letter_match = LETTER_RE.search(window)
        point_match = POINT_RE.search(window)
        thesis_match = THESIS_RE.search(window)
        named_code = self._find_named_code(window)
        numbered_act = self._find_numbered_act(window)

        span_end = article_match.end()
        paragraph = paragraph_match.group("paragraph") if paragraph_match else None
        letter = letter_match.group("letter") if letter_match else None
        point = point_match.group("point") if point_match else None
        thesis = thesis_match.group("thesis") if thesis_match else None

        for component_match in (
            paragraph_match,
            letter_match,
            point_match,
            thesis_match,
        ):
            if component_match:
                span_end = max(span_end, window_start + component_match.end())

        act_type = None
        act_number = None
        act_year = None
        act_hint = None
        law_id_hint = None
        if named_code:
            act_hint = named_code["act_hint"]
            law_id_hint = named_code["law_id_hint"]
            span_end = max(span_end, window_start + int(named_code["end"]))
        elif numbered_act:
            act_type = numbered_act["act_type"]
            act_number = numbered_act["act_number"]
            act_year = numbered_act["act_year"]
            act_hint = numbered_act["act_hint"]
            law_id_hint = numbered_act["law_id_hint"]
            span_end = max(span_end, window_start + int(numbered_act["end"]))

        has_act = bool(act_hint or law_id_hint or act_number)
        has_detail = bool(paragraph or letter or point or thesis)
        citation_type = "compound" if has_act or has_detail else "article"
        confidence = self._article_confidence(has_act=has_act, has_detail=has_detail)
        needs_resolution = not bool(law_id_hint)
        lookup_filters = self._lookup_filters(
            law_id_hint=law_id_hint,
            act_type=act_type,
            act_number=act_number,
            act_year=act_year,
            article=article,
            paragraph=paragraph,
            letter=letter,
            point=point,
            thesis=thesis,
        )

        return ExactCitation(
            raw_text=question[article_match.start() : span_end].strip(),
            citation_type=citation_type,
            article=article,
            paragraph=paragraph,
            letter=letter,
            point=point,
            thesis=thesis,
            act_type=act_type,
            act_number=act_number,
            act_year=act_year,
            act_hint=act_hint,
            law_id_hint=law_id_hint,
            confidence=confidence,
            is_relative=False,
            needs_resolution=needs_resolution,
            lookup_filters=lookup_filters,
            span_start=article_match.start(),
            span_end=span_end,
        )

    def _detect_numbered_acts(
        self,
        question: str,
        match_text: str,
        occupied: list[tuple[int, int]],
    ) -> list[ExactCitation]:
        citations: list[ExactCitation] = []
        for pattern, act_type, citation_type in (
            (LAW_RE, "lege", "law"),
            (OUG_RE, "oug", "ordinance"),
            (OG_RE, "og", "ordinance"),
            (HG_RE, "hg", "government_decision"),
        ):
            for match in pattern.finditer(match_text):
                if self._overlaps(match.start(), match.end(), occupied):
                    continue
                citations.append(
                    self._numbered_act_citation(
                        question=question,
                        match=match,
                        act_type=act_type,
                        citation_type=citation_type,
                    )
                )
        return citations

    def _numbered_act_citation(
        self,
        question: str,
        match: re.Match[str],
        act_type: str,
        citation_type: str,
    ) -> ExactCitation:
        act_number = match.group("number")
        act_year = match.group("year")
        law_id_hint = NUMBERED_ACT_ALIASES.get((act_type, act_number, act_year))
        lookup_filters = self._lookup_filters(
            law_id_hint=law_id_hint,
            act_type=act_type,
            act_number=act_number,
            act_year=act_year,
        )
        return ExactCitation(
            raw_text=question[match.start() : match.end()],
            citation_type=citation_type,
            act_type=act_type,
            act_number=act_number,
            act_year=act_year,
            act_hint=question[match.start() : match.end()],
            law_id_hint=law_id_hint,
            confidence=0.85,
            is_relative=False,
            needs_resolution=law_id_hint is None,
            lookup_filters=lookup_filters,
            span_start=match.start(),
            span_end=match.end(),
        )

    def _detect_named_codes(
        self,
        question: str,
        match_text: str,
        occupied: list[tuple[int, int]],
    ) -> list[ExactCitation]:
        citations: list[ExactCitation] = []
        for alias, act_hint, law_id_hint in NAMED_CODE_ALIASES:
            for match in re.finditer(rf"\b{re.escape(alias)}\b", match_text):
                if self._overlaps(match.start(), match.end(), occupied):
                    continue
                citations.append(
                    ExactCitation(
                        raw_text=question[match.start() : match.end()],
                        citation_type="named_code",
                        act_hint=act_hint,
                        law_id_hint=law_id_hint,
                        confidence=0.80,
                        is_relative=False,
                        needs_resolution=False,
                        lookup_filters={"law_id": law_id_hint},
                        span_start=match.start(),
                        span_end=match.end(),
                    )
                )
        return citations

    def _detect_relative_references(
        self,
        question: str,
        match_text: str,
        occupied: list[tuple[int, int]],
    ) -> list[ExactCitation]:
        citations: list[ExactCitation] = []
        for match in RELATIVE_RE.finditer(match_text):
            if self._overlaps(match.start(), match.end(), occupied):
                continue
            citations.append(
                ExactCitation(
                    raw_text=question[match.start() : match.end()],
                    citation_type="law",
                    confidence=0.45,
                    is_relative=True,
                    needs_resolution=True,
                    lookup_filters={"relative_reference": match.group(0)},
                    span_start=match.start(),
                    span_end=match.end(),
                )
            )
        return citations

    def _detect_relative_components(
        self,
        question: str,
        match_text: str,
        occupied: list[tuple[int, int]],
    ) -> list[ExactCitation]:
        citations: list[ExactCitation] = []
        for pattern, field_name, citation_type in (
            (PARAGRAPH_RE, "paragraph", "paragraph"),
            (LETTER_RE, "letter", "letter"),
            (POINT_RE, "point", "point"),
            (THESIS_RE, "thesis", "thesis"),
        ):
            for match in pattern.finditer(match_text):
                if self._overlaps(match.start(), match.end(), occupied):
                    continue
                value = match.group(field_name)
                lookup_filters = {f"{field_name}_number": value}
                citations.append(
                    ExactCitation(
                        raw_text=question[match.start() : match.end()],
                        citation_type=citation_type,
                        paragraph=value if field_name == "paragraph" else None,
                        letter=value if field_name == "letter" else None,
                        point=value if field_name == "point" else None,
                        thesis=value if field_name == "thesis" else None,
                        confidence=0.55,
                        is_relative=True,
                        needs_resolution=True,
                        lookup_filters=lookup_filters,
                        span_start=match.start(),
                        span_end=match.end(),
                    )
                )
        return citations

    def _find_named_code(self, text: str) -> dict[str, str | int] | None:
        matches: list[dict[str, str | int]] = []
        for alias, act_hint, law_id_hint in NAMED_CODE_ALIASES:
            match = re.search(rf"\b{re.escape(alias)}\b", text)
            if match:
                matches.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "act_hint": act_hint,
                        "law_id_hint": law_id_hint,
                    }
                )
        if not matches:
            return None
        return min(matches, key=lambda item: int(item["start"]))

    def _find_numbered_act(self, text: str) -> dict[str, str | int | None] | None:
        matches: list[dict[str, str | int | None]] = []
        for pattern, act_type in (
            (LAW_RE, "lege"),
            (OUG_RE, "oug"),
            (OG_RE, "og"),
            (HG_RE, "hg"),
        ):
            match = pattern.search(text)
            if match:
                act_number = match.group("number")
                act_year = match.group("year")
                matches.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "act_type": act_type,
                        "act_number": act_number,
                        "act_year": act_year,
                        "act_hint": text[match.start() : match.end()],
                        "law_id_hint": NUMBERED_ACT_ALIASES.get(
                            (act_type, act_number, act_year)
                        ),
                    }
                )
        if not matches:
            return None
        return min(matches, key=lambda item: int(item["start"]))

    def _article_confidence(self, has_act: bool, has_detail: bool) -> float:
        if has_act and has_detail:
            return 0.98
        if has_act:
            return 0.95
        return 0.70

    def _lookup_filters(
        self,
        law_id_hint: str | None = None,
        act_type: str | None = None,
        act_number: str | None = None,
        act_year: str | None = None,
        article: str | None = None,
        paragraph: str | None = None,
        letter: str | None = None,
        point: str | None = None,
        thesis: str | None = None,
    ) -> dict[str, str]:
        filters: dict[str, str] = {}
        if law_id_hint:
            filters["law_id"] = law_id_hint
        if act_type:
            filters["act_type"] = act_type
        if act_number:
            filters["act_number"] = act_number
        if act_year:
            filters["act_year"] = act_year
        if article:
            filters["article_number"] = article
        if paragraph:
            filters["paragraph_number"] = paragraph
        if letter:
            filters["letter"] = letter
        if point:
            filters["point_number"] = point
        if thesis:
            filters["thesis"] = thesis
        return filters

    def _match_text(self, text: str) -> str:
        text = text.replace("ş", "ș").replace("Ş", "Ș")
        text = text.replace("ţ", "ț").replace("Ţ", "Ț")
        normalized = unicodedata.normalize("NFD", text)
        stripped = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        return stripped.casefold()

    def _overlaps(
        self,
        start: int,
        end: int,
        occupied: list[tuple[int, int]],
    ) -> bool:
        return any(start < occupied_end and end > occupied_start for occupied_start, occupied_end in occupied)
