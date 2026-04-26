from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import asyncpg
from pydantic import BaseModel, ConfigDict, Field

from ingestion.embeddings import validate_embedding_vector
from ingestion.imports import (
    DEFAULT_EMBEDDING_DIM,
    ImportPlan,
    ImportRepositoryResult,
    ImportRunRepositoryResult,
)
from ingestion.normalizer import normalize_legal_text, repair_romanian_mojibake


MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[1]
    / "apps"
    / "api"
    / "app"
    / "db"
    / "migrations"
)


class ImportBundleApplyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    import_run_id: str
    status: str
    legal_units: ImportRepositoryResult = Field(default_factory=ImportRepositoryResult)
    legal_edges: ImportRepositoryResult = Field(default_factory=ImportRepositoryResult)
    embeddings: ImportRepositoryResult = Field(default_factory=ImportRepositoryResult)

    @property
    def counts(self) -> dict[str, Any]:
        return {
            "legal_units": self.legal_units.model_dump(),
            "legal_edges": self.legal_edges.model_dump(),
            "embeddings": self.embeddings.model_dump(),
        }


ConnectFunc = Callable[[str], Any | Awaitable[Any]]
ProgressCallback = Callable[[str], None]
UPSERT_BATCH_SIZE = 100


class PostgresImportRepository:
    """PostgreSQL repository for H08 D2 legal_units/legal_edges imports."""

    def __init__(
        self,
        database_url: str,
        *,
        connect_func: ConnectFunc | None = None,
    ) -> None:
        self.database_url = _to_asyncpg_url(database_url)
        self._connect_func = connect_func or asyncpg.connect

    def ensure_schema(self) -> None:
        _run_sync(self.aensure_schema())

    async def aensure_schema(self) -> None:
        statements = _migration_statements()
        async with self._managed_connection() as connection:
            for statement in statements:
                await connection.execute(statement)

    def upsert_legal_units(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> ImportRepositoryResult:
        return _run_sync(self.aupsert_legal_units(records))

    async def aupsert_legal_units(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        connection: Any | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ImportRepositoryResult:
        prepared_records = [_legal_unit_params(record) for record in records]
        if not prepared_records:
            return ImportRepositoryResult()
        if connection is not None:
            return await self._upsert_legal_units(
                connection,
                prepared_records,
                progress_callback=progress_callback,
            )
        async with self._managed_connection() as managed_connection:
            async with managed_connection.transaction():
                return await self._upsert_legal_units(
                    managed_connection,
                    prepared_records,
                    progress_callback=progress_callback,
                )

    def upsert_legal_edges(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> ImportRepositoryResult:
        return _run_sync(self.aupsert_legal_edges(records))

    async def aupsert_legal_edges(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        connection: Any | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ImportRepositoryResult:
        prepared_records = [_legal_edge_params(record) for record in records]
        if not prepared_records:
            return ImportRepositoryResult()
        if connection is not None:
            return await self._upsert_legal_edges(
                connection,
                prepared_records,
                progress_callback=progress_callback,
            )
        async with self._managed_connection() as managed_connection:
            async with managed_connection.transaction():
                return await self._upsert_legal_edges(
                    managed_connection,
                    prepared_records,
                    progress_callback=progress_callback,
                )

    def upsert_embeddings(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        expected_dim: int = DEFAULT_EMBEDDING_DIM,
    ) -> ImportRepositoryResult:
        return _run_sync(self.aupsert_embeddings(records, expected_dim=expected_dim))

    async def aupsert_embeddings(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        expected_dim: int = DEFAULT_EMBEDDING_DIM,
        connection: Any | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ImportRepositoryResult:
        prepared_records = [
            _embedding_params(record, expected_dim=expected_dim) for record in records
        ]
        if not prepared_records:
            return ImportRepositoryResult()
        if connection is not None:
            return await self._upsert_embeddings(
                connection,
                prepared_records,
                progress_callback=progress_callback,
            )
        async with self._managed_connection() as managed_connection:
            async with managed_connection.transaction():
                return await self._upsert_embeddings(
                    managed_connection,
                    prepared_records,
                    progress_callback=progress_callback,
                )

    def record_import_run(self, plan: ImportPlan) -> ImportRunRepositoryResult:
        return _run_sync(self.arecord_import_run(plan))

    async def arecord_import_run(
        self,
        plan: ImportPlan,
        *,
        connection: Any | None = None,
    ) -> ImportRunRepositoryResult:
        if connection is not None:
            return await self._record_import_run(connection, plan)
        async with self._managed_connection() as managed_connection:
            async with managed_connection.transaction():
                return await self._record_import_run(managed_connection, plan)

    def finalize_import_run(
        self,
        import_run_id: str,
        *,
        status: str,
        counts: Mapping[str, Any],
        errors: Iterable[Mapping[str, Any]] | None = None,
        warnings: Iterable[Mapping[str, Any]] | None = None,
    ) -> ImportRunRepositoryResult:
        return _run_sync(
            self.afinalize_import_run(
                import_run_id,
                status=status,
                counts=counts,
                errors=errors,
                warnings=warnings,
            )
        )

    async def afinalize_import_run(
        self,
        import_run_id: str,
        *,
        status: str,
        counts: Mapping[str, Any],
        errors: Iterable[Mapping[str, Any]] | None = None,
        warnings: Iterable[Mapping[str, Any]] | None = None,
        connection: Any | None = None,
    ) -> ImportRunRepositoryResult:
        if connection is not None:
            return await self._finalize_import_run(
                connection,
                import_run_id,
                status=status,
                counts=counts,
                errors=errors,
                warnings=warnings,
            )
        async with self._managed_connection() as managed_connection:
            async with managed_connection.transaction():
                return await self._finalize_import_run(
                    managed_connection,
                    import_run_id,
                    status=status,
                    counts=counts,
                    errors=errors,
                    warnings=warnings,
                )

    def apply_import_plan(
        self,
        plan: ImportPlan,
        *,
        legal_units: Iterable[Mapping[str, Any]],
        legal_edges: Iterable[Mapping[str, Any]],
        embeddings: Iterable[Mapping[str, Any]] | None = None,
        expected_embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        progress_callback: ProgressCallback | None = None,
        statement_timeout_seconds: int = 0,
    ) -> ImportBundleApplyResult:
        return _run_sync(
            self.aapply_import_plan(
                plan,
                legal_units=legal_units,
                legal_edges=legal_edges,
                embeddings=embeddings,
                expected_embedding_dim=expected_embedding_dim,
                progress_callback=progress_callback,
                statement_timeout_seconds=statement_timeout_seconds,
            )
        )

    async def aapply_import_plan(
        self,
        plan: ImportPlan,
        *,
        legal_units: Iterable[Mapping[str, Any]],
        legal_edges: Iterable[Mapping[str, Any]],
        embeddings: Iterable[Mapping[str, Any]] | None = None,
        expected_embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        progress_callback: ProgressCallback | None = None,
        statement_timeout_seconds: int = 0,
    ) -> ImportBundleApplyResult:
        if statement_timeout_seconds < 0:
            raise ValueError("statement_timeout_seconds must be greater than or equal to 0")
        unit_records = list(legal_units)
        edge_records = list(legal_edges)
        embedding_records = list(embeddings or [])
        async with self._managed_connection() as connection:
            async with connection.transaction():
                if statement_timeout_seconds:
                    await _set_statement_timeout(connection, statement_timeout_seconds)
                await self._record_import_run(connection, plan)
                _emit_progress(progress_callback, f"started import_run id={plan.import_run_id}")
                units_result = await self.aupsert_legal_units(
                    unit_records,
                    connection=connection,
                    progress_callback=progress_callback,
                )
                edges_result = await self.aupsert_legal_edges(
                    edge_records,
                    connection=connection,
                    progress_callback=progress_callback,
                )
                embeddings_result = await self.aupsert_embeddings(
                    embedding_records,
                    expected_dim=expected_embedding_dim,
                    connection=connection,
                    progress_callback=progress_callback,
                )
                apply_result = ImportBundleApplyResult(
                    import_run_id=plan.import_run_id,
                    status="succeeded",
                    legal_units=units_result,
                    legal_edges=edges_result,
                    embeddings=embeddings_result,
                )
                await self._finalize_import_run(
                    connection,
                    plan.import_run_id,
                    status="succeeded",
                    counts=apply_result.counts,
                    errors=[],
                    warnings=[warning.model_dump() for warning in plan.warnings],
                )
                _emit_progress(progress_callback, f"finalized import_run id={plan.import_run_id}")
                return apply_result

    @asynccontextmanager
    async def _managed_connection(self):
        connection = await _maybe_await(self._connect_func(self.database_url))
        try:
            yield connection
        finally:
            close = getattr(connection, "close", None)
            if close is not None:
                await _maybe_await(close())

    async def _upsert_legal_units(
        self,
        connection: Any,
        records: list[tuple[Any, ...]],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> ImportRepositoryResult:
        return await _run_upsert_batches(
            connection,
            records,
            sql_template=_UPSERT_LEGAL_UNIT_SQL,
            parameter_count=24,
            parameter_casts={13: "::jsonb", 21: "::jsonb", 24: "::jsonb"},
            progress_label="legal_units",
            progress_callback=progress_callback,
        )

    async def _upsert_legal_edges(
        self,
        connection: Any,
        records: list[tuple[Any, ...]],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> ImportRepositoryResult:
        return await _run_upsert_batches(
            connection,
            records,
            sql_template=_UPSERT_LEGAL_EDGE_SQL,
            parameter_count=7,
            parameter_casts={7: "::jsonb"},
            progress_label="legal_edges",
            progress_callback=progress_callback,
        )

    async def _upsert_embeddings(
        self,
        connection: Any,
        records: list[tuple[Any, ...]],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> ImportRepositoryResult:
        return await _run_upsert_batches(
            connection,
            records,
            sql_template=_UPSERT_EMBEDDING_SQL,
            parameter_count=9,
            parameter_casts={7: "::vector", 8: "::jsonb"},
            progress_label="embeddings",
            progress_callback=progress_callback,
        )

    async def _record_import_run(
        self,
        connection: Any,
        plan: ImportPlan,
    ) -> ImportRunRepositoryResult:
        row = await connection.fetchrow(
            _RECORD_IMPORT_RUN_SQL,
            plan.import_run_id,
            plan.source_dir,
            plan.mode,
            _json_param(plan.counts.model_dump()),
            _json_param([warning.model_dump() for warning in plan.warnings]),
            _json_param([error.model_dump() for error in plan.errors]),
        )
        return ImportRunRepositoryResult(import_run_id=row["id"], status=row["status"])

    async def _finalize_import_run(
        self,
        connection: Any,
        import_run_id: str,
        *,
        status: str,
        counts: Mapping[str, Any],
        errors: Iterable[Mapping[str, Any]] | None = None,
        warnings: Iterable[Mapping[str, Any]] | None = None,
    ) -> ImportRunRepositoryResult:
        row = await connection.fetchrow(
            _FINALIZE_IMPORT_RUN_SQL,
            import_run_id,
            status,
            _json_param(dict(counts)),
            _json_param(list(warnings or [])),
            _json_param(list(errors or [])),
        )
        return ImportRunRepositoryResult(import_run_id=row["id"], status=row["status"])


def _run_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError("sync repository methods cannot run inside an active event loop")


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _set_statement_timeout(connection: Any, seconds: int) -> None:
    milliseconds = int(seconds) * 1000
    await connection.execute(f"SET LOCAL statement_timeout = {milliseconds}")


async def _run_upsert_batches(
    connection: Any,
    records: list[tuple[Any, ...]],
    *,
    sql_template: str,
    parameter_count: int,
    parameter_casts: Mapping[int, str],
    progress_label: str,
    progress_callback: ProgressCallback | None,
) -> ImportRepositoryResult:
    inserted = 0
    updated = 0
    unchanged = 0
    processed = 0
    fetch_many = getattr(connection, "fetch", None)
    for batch in _chunks(records, UPSERT_BATCH_SIZE):
        if fetch_many is None:
            batch_rows = await _run_upsert_batch_row_by_row(
                connection,
                batch,
                sql_template=sql_template,
                parameter_count=parameter_count,
                parameter_casts=parameter_casts,
            )
        else:
            batch_rows = await fetch_many(
                _upsert_sql(
                    sql_template,
                    row_count=len(batch),
                    parameter_count=parameter_count,
                    parameter_casts=parameter_casts,
                ),
                *_flatten(batch),
            )
        batch_inserted, batch_updated, batch_unchanged = _upsert_result_counts(
            batch_size=len(batch),
            rows=batch_rows,
        )
        inserted += batch_inserted
        updated += batch_updated
        unchanged += batch_unchanged
        processed += len(batch)
        _emit_progress(
            progress_callback,
            f"{progress_label} progress processed={processed} total={len(records)}",
        )
    return ImportRepositoryResult(
        attempted=len(records),
        inserted=inserted,
        updated=updated,
        unchanged=unchanged,
    )


async def _run_upsert_batch_row_by_row(
    connection: Any,
    batch: list[tuple[Any, ...]],
    *,
    sql_template: str,
    parameter_count: int,
    parameter_casts: Mapping[int, str],
) -> list[Any]:
    sql = _upsert_sql(
        sql_template,
        row_count=1,
        parameter_count=parameter_count,
        parameter_casts=parameter_casts,
    )
    rows: list[Any] = []
    for params in batch:
        row = await connection.fetchrow(sql, *params)
        if row is not None:
            rows.append(row)
    return rows


def _upsert_result_counts(*, batch_size: int, rows: Iterable[Any]) -> tuple[int, int, int]:
    returned_rows = list(rows)
    inserted = sum(1 for row in returned_rows if bool(row["inserted"]))
    updated = len(returned_rows) - inserted
    unchanged = batch_size - len(returned_rows)
    return inserted, updated, unchanged


def _upsert_sql(
    template: str,
    *,
    row_count: int,
    parameter_count: int,
    parameter_casts: Mapping[int, str],
) -> str:
    return template.format(
        values=_values_clause(
            row_count=row_count,
            parameter_count=parameter_count,
            parameter_casts=parameter_casts,
        )
    )


def _values_clause(
    *,
    row_count: int,
    parameter_count: int,
    parameter_casts: Mapping[int, str],
) -> str:
    placeholders: list[str] = []
    parameter_index = 1
    for _ in range(row_count):
        row_placeholders = []
        for field_index in range(1, parameter_count + 1):
            row_placeholders.append(
                f"${parameter_index}{parameter_casts.get(field_index, '')}"
            )
            parameter_index += 1
        placeholders.append("(" + ", ".join(row_placeholders) + ", now(), now())")
    return ",\n".join(placeholders)


def _flatten(batch: Iterable[tuple[Any, ...]]) -> list[Any]:
    return [value for params in batch for value in params]


def _chunks(records: list[tuple[Any, ...]], size: int) -> Iterable[list[tuple[Any, ...]]]:
    for index in range(0, len(records), size):
        yield records[index : index + size]


def _emit_progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _to_asyncpg_url(database_url: str) -> str:
    if database_url.startswith("postgresql+asyncpg://"):
        return "postgresql://" + database_url[len("postgresql+asyncpg://") :]
    return database_url


def _migration_statements() -> list[str]:
    statements: list[str] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        sql = path.read_text(encoding="utf-8")
        statements.extend(
            statement.strip() for statement in sql.split(";") if statement.strip()
        )
    return statements


def _legal_unit_params(record: Mapping[str, Any]) -> tuple[Any, ...]:
    unit_id = _required_text(record, "id", artifact="legal_units")
    law_id = _text(record.get("law_id")) or "unknown"
    repaired_raw_text = repair_romanian_mojibake(_text(record.get("raw_text")) or "") or ""
    normalized_text = _text(record.get("normalized_text"))
    if normalized_text is None:
        normalized_text = normalize_legal_text(repaired_raw_text)
    else:
        normalized_text = normalize_legal_text(normalized_text)
    return (
        unit_id,
        _text(record.get("canonical_id")),
        _text(record.get("source_id")),
        law_id,
        _text(record.get("law_title")) or law_id,
        _text(record.get("act_type")),
        _text(record.get("act_number")),
        _date_param(record.get("publication_date"), "publication_date"),
        _date_param(record.get("effective_date"), "effective_date"),
        _date_param(record.get("version_start"), "version_start"),
        _date_param(record.get("version_end"), "version_end"),
        _text(record.get("status")) or "unknown",
        _json_param(record.get("hierarchy_path")),
        _text(record.get("article_number")),
        _text(record.get("paragraph_number")),
        _text(record.get("letter_number")),
        _text(record.get("point_number")),
        repaired_raw_text,
        normalized_text,
        _text(record.get("legal_domain")),
        _json_param(record.get("legal_concepts")),
        _text(record.get("source_url")),
        _text(record.get("parent_id")),
        _json_param(record.get("parser_warnings")),
    )


def _legal_edge_params(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _required_text(record, "id", artifact="legal_edges"),
        _required_text(record, "source_id", artifact="legal_edges"),
        _required_text(record, "target_id", artifact="legal_edges"),
        _required_text(record, "type", artifact="legal_edges"),
        _float_param(record.get("weight")),
        _float_param(record.get("confidence")),
        _json_param(record.get("metadata")),
    )


def _embedding_params(
    record: Mapping[str, Any] | BaseModel,
    *,
    expected_dim: int,
) -> tuple[Any, ...]:
    if expected_dim <= 0:
        raise ValueError("expected_dim must be greater than 0")
    data = _record_mapping(record)
    metadata = data.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("embeddings record metadata must be a JSON object")

    embedding_dim = data.get("embedding_dim")
    if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int):
        raise ValueError("embeddings record missing valid embedding_dim")
    if embedding_dim != expected_dim:
        raise ValueError(
            f"embeddings record dimension mismatch: expected {expected_dim}, got {embedding_dim}"
        )

    vector = validate_embedding_vector(data.get("embedding"), expected_dim=expected_dim)
    legal_unit_id = _text(data.get("legal_unit_id")) or _text(metadata.get("legal_unit_id"))
    if not legal_unit_id:
        raise ValueError("embeddings record missing legal_unit_id mapping")

    return (
        _required_text(data, "record_id", artifact="embeddings"),
        legal_unit_id,
        _text(data.get("chunk_id")) or _text(metadata.get("chunk_id")),
        _required_text(data, "model_name", artifact="embeddings"),
        embedding_dim,
        _required_text(data, "text_hash", artifact="embeddings"),
        _vector_param(vector),
        _json_param(metadata),
        _text(data.get("source_path")),
    )


def _required_text(record: Mapping[str, Any], field: str, *, artifact: str) -> str:
    value = _text(record.get(field))
    if not value:
        raise ValueError(f"{artifact} record missing required field: {field}")
    return value


def _record_mapping(record: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(record, BaseModel):
        return record.model_dump()
    return dict(record)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text != "" else None


def _date_param(value: Any, field: str) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError as exc:
            raise ValueError(f"legal_units record has invalid date field: {field}") from exc
    raise ValueError(f"legal_units record has invalid date field: {field}")


def _float_param(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _json_param(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _vector_param(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


_UPSERT_LEGAL_UNIT_SQL = """
INSERT INTO legal_units (
    id,
    canonical_id,
    source_id,
    law_id,
    law_title,
    act_type,
    act_number,
    publication_date,
    effective_date,
    version_start,
    version_end,
    status,
    hierarchy_path,
    article_number,
    paragraph_number,
    letter_number,
    point_number,
    raw_text,
    normalized_text,
    legal_domain,
    legal_concepts,
    source_url,
    parent_id,
    parser_warnings,
    created_at,
    updated_at
) VALUES {values}
ON CONFLICT (id) DO UPDATE SET
    canonical_id = EXCLUDED.canonical_id,
    source_id = EXCLUDED.source_id,
    law_id = EXCLUDED.law_id,
    law_title = EXCLUDED.law_title,
    act_type = EXCLUDED.act_type,
    act_number = EXCLUDED.act_number,
    publication_date = EXCLUDED.publication_date,
    effective_date = EXCLUDED.effective_date,
    version_start = EXCLUDED.version_start,
    version_end = EXCLUDED.version_end,
    status = EXCLUDED.status,
    hierarchy_path = EXCLUDED.hierarchy_path,
    article_number = EXCLUDED.article_number,
    paragraph_number = EXCLUDED.paragraph_number,
    letter_number = EXCLUDED.letter_number,
    point_number = EXCLUDED.point_number,
    raw_text = EXCLUDED.raw_text,
    normalized_text = EXCLUDED.normalized_text,
    legal_domain = EXCLUDED.legal_domain,
    legal_concepts = EXCLUDED.legal_concepts,
    source_url = EXCLUDED.source_url,
    parent_id = EXCLUDED.parent_id,
    parser_warnings = EXCLUDED.parser_warnings,
    updated_at = now()
WHERE (
    legal_units.canonical_id,
    legal_units.source_id,
    legal_units.law_id,
    legal_units.law_title,
    legal_units.act_type,
    legal_units.act_number,
    legal_units.publication_date,
    legal_units.effective_date,
    legal_units.version_start,
    legal_units.version_end,
    legal_units.status,
    legal_units.hierarchy_path,
    legal_units.article_number,
    legal_units.paragraph_number,
    legal_units.letter_number,
    legal_units.point_number,
    legal_units.raw_text,
    legal_units.normalized_text,
    legal_units.legal_domain,
    legal_units.legal_concepts,
    legal_units.source_url,
    legal_units.parent_id,
    legal_units.parser_warnings
) IS DISTINCT FROM (
    EXCLUDED.canonical_id,
    EXCLUDED.source_id,
    EXCLUDED.law_id,
    EXCLUDED.law_title,
    EXCLUDED.act_type,
    EXCLUDED.act_number,
    EXCLUDED.publication_date,
    EXCLUDED.effective_date,
    EXCLUDED.version_start,
    EXCLUDED.version_end,
    EXCLUDED.status,
    EXCLUDED.hierarchy_path,
    EXCLUDED.article_number,
    EXCLUDED.paragraph_number,
    EXCLUDED.letter_number,
    EXCLUDED.point_number,
    EXCLUDED.raw_text,
    EXCLUDED.normalized_text,
    EXCLUDED.legal_domain,
    EXCLUDED.legal_concepts,
    EXCLUDED.source_url,
    EXCLUDED.parent_id,
    EXCLUDED.parser_warnings
)
RETURNING (xmax = 0) AS inserted
"""


_UPSERT_LEGAL_EDGE_SQL = """
INSERT INTO legal_edges (
    id,
    source_id,
    target_id,
    type,
    weight,
    confidence,
    metadata,
    created_at,
    updated_at
) VALUES {values}
ON CONFLICT (id) DO UPDATE SET
    source_id = EXCLUDED.source_id,
    target_id = EXCLUDED.target_id,
    type = EXCLUDED.type,
    weight = EXCLUDED.weight,
    confidence = EXCLUDED.confidence,
    metadata = EXCLUDED.metadata,
    updated_at = now()
WHERE (
    legal_edges.source_id,
    legal_edges.target_id,
    legal_edges.type,
    legal_edges.weight,
    legal_edges.confidence,
    legal_edges.metadata
) IS DISTINCT FROM (
    EXCLUDED.source_id,
    EXCLUDED.target_id,
    EXCLUDED.type,
    EXCLUDED.weight,
    EXCLUDED.confidence,
    EXCLUDED.metadata
)
RETURNING (xmax = 0) AS inserted
"""


_UPSERT_EMBEDDING_SQL = """
INSERT INTO legal_embeddings (
    record_id,
    legal_unit_id,
    chunk_id,
    model_name,
    embedding_dim,
    text_hash,
    embedding,
    metadata,
    source_path,
    created_at,
    updated_at
) VALUES {values}
ON CONFLICT (record_id, model_name, text_hash) DO UPDATE SET
    legal_unit_id = EXCLUDED.legal_unit_id,
    chunk_id = EXCLUDED.chunk_id,
    embedding_dim = EXCLUDED.embedding_dim,
    embedding = EXCLUDED.embedding,
    metadata = EXCLUDED.metadata,
    source_path = EXCLUDED.source_path,
    updated_at = now()
WHERE (
    legal_embeddings.legal_unit_id,
    legal_embeddings.chunk_id,
    legal_embeddings.embedding_dim,
    legal_embeddings.embedding::text,
    legal_embeddings.metadata,
    legal_embeddings.source_path
) IS DISTINCT FROM (
    EXCLUDED.legal_unit_id,
    EXCLUDED.chunk_id,
    EXCLUDED.embedding_dim,
    EXCLUDED.embedding::text,
    EXCLUDED.metadata,
    EXCLUDED.source_path
)
RETURNING (xmax = 0) AS inserted
"""


_RECORD_IMPORT_RUN_SQL = """
INSERT INTO import_runs (
    id,
    source_dir,
    mode,
    status,
    counts,
    warnings,
    errors,
    created_at,
    finished_at
) VALUES (
    $1, $2, $3, 'running', $4::jsonb, $5::jsonb, $6::jsonb, now(), NULL
)
ON CONFLICT (id) DO UPDATE SET
    source_dir = EXCLUDED.source_dir,
    mode = EXCLUDED.mode,
    status = EXCLUDED.status,
    counts = EXCLUDED.counts,
    warnings = EXCLUDED.warnings,
    errors = EXCLUDED.errors,
    finished_at = NULL
RETURNING id, status
"""


_FINALIZE_IMPORT_RUN_SQL = """
INSERT INTO import_runs (
    id,
    status,
    counts,
    warnings,
    errors,
    created_at,
    finished_at
) VALUES (
    $1, $2, $3::jsonb, $4::jsonb, $5::jsonb, now(), now()
)
ON CONFLICT (id) DO UPDATE SET
    status = EXCLUDED.status,
    counts = EXCLUDED.counts,
    warnings = EXCLUDED.warnings,
    errors = EXCLUDED.errors,
    finished_at = EXCLUDED.finished_at
RETURNING id, status
"""
