from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import unicodedata

from bs4 import BeautifulSoup, Tag

from ingestion.normalizer import repair_romanian_mojibake


NON_TEXT_TAGS = {
    "head",
    "iframe",
    "link",
    "meta",
    "noscript",
    "object",
    "script",
    "style",
    "svg",
}
ALWAYS_DROP_TAGS = {
    "button",
    "form",
    "input",
    "nav",
    "option",
    "select",
    "textarea",
}
INLINE_TAGS = {
    "a",
    "abbr",
    "b",
    "cite",
    "em",
    "i",
    "label",
    "small",
    "span",
    "strong",
    "sub",
    "sup",
    "u",
}
CONTENT_SELECTOR_PATTERNS = (
    "textdocumentleg",
    "documentleg",
    "legal-content",
    "legal_content",
    "main-content",
    "main_content",
    "content",
    "document",
    "act-content",
)
CONTAINER_TAGS = {"article", "div", "main", "section", "td"}
NAVIGATION_ATTRIBUTE_RE = re.compile(
    r"(?:^|[-_\s])("
    r"breadcrumb|cautare|cookie|footer|header|menu|meniu|nav|"
    r"pagination|print|search|share|sidebar|social|toolbar"
    r")(?:$|[-_\s])",
    re.IGNORECASE,
)
LEGAL_MARKER_RE = re.compile(
    r"\b(?:Art\.|Articolul|TITLUL|CAPITOLUL|SEC(?:T|Ț)IUNEA)\b|"
    r"^\s*\(\d+\)|^\s*[a-z]\)",
    re.IGNORECASE | re.MULTILINE,
)
NAVIGATION_RESIDUE_TERMS = (
    "acasa",
    "cauta",
    "cautare",
    "cookie",
    "meniu",
    "tipareste",
)


@dataclass(frozen=True)
class HtmlCleanConfig:
    preserve_line_breaks: bool = True
    collapse_spaces: bool = True
    drop_empty_lines: bool = True


@dataclass(frozen=True)
class CleanTextResult:
    lines: list[str]
    warnings: list[str]
    removed_blocks_count: int
    selected_container: str | None
    text_hash: str | None


def clean_html_to_lines(
    html: str | None,
    *,
    config: HtmlCleanConfig | None = None,
) -> CleanTextResult:
    config = config or HtmlCleanConfig()
    if not html:
        return CleanTextResult(
            lines=[],
            warnings=["empty_html"],
            removed_blocks_count=0,
            selected_container=None,
            text_hash=None,
        )

    repaired_html = repair_romanian_mojibake(html) or html
    soup = BeautifulSoup(repaired_html, "html.parser")
    warnings: set[str] = set()
    removed_blocks_count = _remove_noise_blocks(soup)
    if removed_blocks_count:
        warnings.add("removed_navigation_blocks")

    selected, selected_container = _select_legal_container(soup)
    if selected is None:
        selected = soup.find("body") or soup
        selected_container = "body" if soup.find("body") else None
        warnings.update({"legal_container_not_found", "used_body_fallback"})

    _unwrap_inline_tags(selected)
    lines = _extract_lines(selected, config=config)
    if navigation_residue_count("\n".join(lines)):
        warnings.add("possible_navigation_residue")

    return CleanTextResult(
        lines=lines,
        warnings=sorted(warnings),
        removed_blocks_count=removed_blocks_count,
        selected_container=selected_container,
        text_hash=_text_hash(lines),
    )


def navigation_residue_count(text: str) -> int:
    normalized = _normalize_for_detection(text)
    return sum(
        len(re.findall(rf"\b{re.escape(term)}\b", normalized))
        for term in NAVIGATION_RESIDUE_TERMS
    )


def text_cleanliness_score(texts: list[str]) -> float:
    if not texts:
        return 1.0
    dirty_count = sum(1 for text in texts if navigation_residue_count(text))
    return round((len(texts) - dirty_count) / len(texts), 4)


def _is_live_tag(tag: Tag) -> bool:
    return isinstance(tag, Tag) and tag.name is not None and tag.attrs is not None


def _remove_noise_blocks(soup: BeautifulSoup) -> int:
    removed = 0
    for tag in list(soup.find_all(NON_TEXT_TAGS | ALWAYS_DROP_TAGS)):
        if not _is_live_tag(tag):
            continue
        tag.decompose()
        removed += 1

    for tag in list(soup.find_all(True)):
        if not _is_live_tag(tag):
            continue
        if tag.name in {"header", "footer"} and _is_navigation_like(tag):
            tag.decompose()
            removed += 1
            continue
        if _has_navigation_attributes(tag):
            tag.decompose()
            removed += 1
    return removed


def _select_legal_container(soup: BeautifulSoup) -> tuple[Tag | None, str | None]:
    candidates: list[tuple[int, Tag, str]] = []
    for tag in soup.find_all(True):
        if not _is_live_tag(tag):
            continue
        if tag.name not in CONTAINER_TAGS:
            continue
        label = _container_label(tag)
        score = _container_score(tag)
        if score > 0:
            candidates.append((score, tag, label))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (-item[0], _tag_depth(item[1]), item[2]))
    _, tag, label = candidates[0]
    return tag, label


def _container_score(tag: Tag) -> int:
    if not _is_live_tag(tag):
        return 0
    attrs = _attribute_text(tag)
    text = tag.get_text("\n", strip=True)
    normalized_attrs = attrs.casefold()
    has_content_selector = any(
        pattern in normalized_attrs for pattern in CONTENT_SELECTOR_PATTERNS
    )
    if tag.name in {"html", "body"} and not has_content_selector:
        return 0
    score = 0
    if has_content_selector:
        score += 10
    if tag.name == "main":
        score += 6
    if tag.name == "article":
        score += 6
    if LEGAL_MARKER_RE.search(text):
        score += 5
    if len(text) > 400:
        score += 2
    if _is_navigation_like(tag):
        score -= 20
    return score


def _container_label(tag: Tag) -> str:
    if not _is_live_tag(tag):
        return "unknown"
    if tag.get("id"):
        return f"#{tag.get('id')}"
    classes = tag.get("class") or []
    if classes:
        return f".{'.'.join(str(item) for item in classes)}"
    return tag.name or "unknown"


def _has_navigation_attributes(tag: Tag) -> bool:
    if not _is_live_tag(tag):
        return False
    if tag.name in {"body", "main", "article"}:
        return False
    return bool(NAVIGATION_ATTRIBUTE_RE.search(_attribute_text(tag)))


def _is_navigation_like(tag: Tag) -> bool:
    if not _is_live_tag(tag):
        return False
    attr_text = _attribute_text(tag)
    if NAVIGATION_ATTRIBUTE_RE.search(attr_text):
        return True
    text = _normalize_for_detection(tag.get_text(" ", strip=True))
    if not text:
        return False
    markers = sum(1 for term in NAVIGATION_RESIDUE_TERMS if term in text)
    return markers >= 2 and not LEGAL_MARKER_RE.search(tag.get_text("\n", strip=True))


def _attribute_text(tag: Tag) -> str:
    if not _is_live_tag(tag):
        return ""
    values: list[str] = [tag.name or ""]
    for key in ("id", "class", "role", "aria-label", "title"):
        value = tag.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value is not None:
            values.append(str(value))
    return " ".join(values)


def _unwrap_inline_tags(container: Tag) -> None:
    if not _is_live_tag(container):
        return
    for tag in list(container.find_all(INLINE_TAGS)):
        if not _is_live_tag(tag):
            continue
        tag.unwrap()


def _extract_lines(container: Tag, *, config: HtmlCleanConfig) -> list[str]:
    separator = "\n" if config.preserve_line_breaks else " "
    raw_text = container.get_text(separator=separator, strip=False)
    lines: list[str] = []
    for raw_line in raw_text.splitlines():
        line = repair_romanian_mojibake(raw_line.strip()) or ""
        if config.collapse_spaces:
            line = re.sub(r"[ \t\f\v]+", " ", line)
        if config.drop_empty_lines and not line:
            continue
        lines.append(line)
    return lines


def _text_hash(lines: list[str]) -> str | None:
    if not lines:
        return None
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _tag_depth(tag: Tag) -> int:
    depth = 0
    parent = tag.parent
    while isinstance(parent, Tag):
        depth += 1
        parent = parent.parent
    return depth


def _normalize_for_detection(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return " ".join(stripped.split())
