# LexAI Backend Documentation

This directory is the backend documentation entry point. It explains the current repository state, not just the original parser prototype.

Recommended reading order:

1. [Backend Architecture](backend-architecture.md)  
   System purpose, package layout, architectural layers, source-of-truth boundaries, and current implementation status.

2. [RAG Pipeline](rag-pipeline.md)  
   Detailed `/api/query` flow from query understanding through retrieval, ranking, EvidencePack, generation, citation verification, repair, and graph enrichment.

3. [Query API](query-api.md)  
   Runtime endpoints, request/response contracts, debug shape, health checks, raw retrieval, admin ingestion, and current API limitations.

4. [Data Ingestion and Storage](data-ingestion-storage.md)  
   Parser pipeline, canonical bundle artifacts, PostgreSQL/pgvector schema, DB import workflow, embeddings, and fixture strategy.

5. [Development and Operations](development-operations.md)  
   Local setup, environment variables, test commands, smoke checks, deployment files, and operational constraints.

Older notes:

- [Ingestion Canonical Bundle](ingestion-canonical-bundle.md)
- [Parser Architecture](parser-architecture.md)

Those older notes are still useful for parser context, but the files above are the authoritative high-level map for the current backend.
