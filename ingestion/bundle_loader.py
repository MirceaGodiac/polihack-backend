from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CANONICAL_ARTIFACT_FILENAMES = {
    "legal_units": "legal_units.json",
    "legal_edges": "legal_edges.json",
    "reference_candidates": "reference_candidates.json",
    "legal_chunks": "legal_chunks.json",
    "embeddings_input": "embeddings_input.jsonl",
    "corpus_manifest": "corpus_manifest.json",
    "validation_report": "validation_report.json",
}


@dataclass(frozen=True)
class CanonicalBundle:
    root_path: Path
    artifact_paths: dict[str, Path]
    legal_units: list[dict[str, Any]]
    legal_edges: list[dict[str, Any]]
    reference_candidates: list[dict[str, Any]]
    legal_chunks: list[dict[str, Any]]
    embeddings_input: list[dict[str, Any]]
    corpus_manifest: dict[str, Any]
    validation_report: dict[str, Any]


def load_canonical_bundle(path: str | Path) -> CanonicalBundle:
    bundle_path = Path(path)
    missing = validate_bundle_files_present(bundle_path)
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Canonical bundle is missing required artifacts: {joined}")

    artifact_paths = {
        artifact: _resolve_artifact_path(bundle_path, artifact)
        for artifact in CANONICAL_ARTIFACT_FILENAMES
    }
    return CanonicalBundle(
        root_path=bundle_path,
        artifact_paths=artifact_paths,
        legal_units=load_legal_units(artifact_paths["legal_units"]),
        legal_edges=load_legal_edges(artifact_paths["legal_edges"]),
        reference_candidates=load_reference_candidates(
            artifact_paths["reference_candidates"]
        ),
        legal_chunks=load_legal_chunks(artifact_paths["legal_chunks"]),
        embeddings_input=load_embeddings_input_jsonl(
            artifact_paths["embeddings_input"]
        ),
        corpus_manifest=_load_json_object(artifact_paths["corpus_manifest"]),
        validation_report=_load_json_object(artifact_paths["validation_report"]),
    )


def load_legal_units(path: str | Path) -> list[dict[str, Any]]:
    return _load_json_list(_resolve_path(path, "legal_units"))


def load_legal_edges(path: str | Path) -> list[dict[str, Any]]:
    return _load_json_list(_resolve_path(path, "legal_edges"))


def load_reference_candidates(path: str | Path) -> list[dict[str, Any]]:
    return _load_json_list(_resolve_path(path, "reference_candidates"))


def load_legal_chunks(path: str | Path) -> list[dict[str, Any]]:
    return _load_json_list(_resolve_path(path, "legal_chunks"))


def load_embeddings_input_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved_path = _resolve_path(path, "embeddings_input")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        resolved_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"{resolved_path}:{line_number} must contain a JSON object"
            )
        records.append(value)
    return records


def validate_bundle_files_present(path: str | Path) -> list[str]:
    bundle_path = Path(path)
    missing: list[str] = []
    for artifact in CANONICAL_ARTIFACT_FILENAMES:
        if _resolve_artifact_path(bundle_path, artifact, required=False) is None:
            missing.append(artifact)
    return missing


def build_unit_index(units: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return _index_by_required_key(units, "id")


def build_chunk_index(chunks: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return _index_by_required_key(chunks, "chunk_id")


def build_contains_adjacency(
    edges: Iterable[dict[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    children: dict[str, list[str]] = {}
    parents: dict[str, list[str]] = {}
    for edge in edges:
        if edge.get("type") != "contains":
            continue
        source_id = str(edge.get("source_id") or "")
        target_id = str(edge.get("target_id") or "")
        if not source_id or not target_id:
            continue
        children.setdefault(source_id, []).append(target_id)
        parents.setdefault(target_id, []).append(source_id)
    return {
        "children": {
            unit_id: sorted(values) for unit_id, values in sorted(children.items())
        },
        "parents": {
            unit_id: sorted(values) for unit_id, values in sorted(parents.items())
        },
    }


def _resolve_path(path: str | Path, artifact: str) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        return _resolve_artifact_path(candidate, artifact)
    return candidate


def _resolve_artifact_path(
    bundle_path: Path,
    artifact: str,
    *,
    required: bool = True,
) -> Path | None:
    filename = CANONICAL_ARTIFACT_FILENAMES[artifact]
    exact_path = bundle_path / filename
    if exact_path.exists():
        return exact_path

    prefixed_matches = sorted(bundle_path.glob(f"*_{filename}"))
    if len(prefixed_matches) == 1:
        return prefixed_matches[0]
    if len(prefixed_matches) > 1:
        raise ValueError(
            f"Multiple candidate files found for artifact {artifact}: "
            + ", ".join(str(path) for path in prefixed_matches)
        )
    if required:
        raise FileNotFoundError(
            f"Canonical artifact {artifact} not found under {bundle_path}"
        )
    return None


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path} must contain a list of JSON objects")
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _index_by_required_key(
    records: Iterable[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Record missing required string key: {key}")
        if value in index:
            raise ValueError(f"Duplicate {key}: {value}")
        index[value] = record
    return index
