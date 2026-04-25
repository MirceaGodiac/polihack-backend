from fastapi import APIRouter

from ..schemas import QueryRequest, QueryResponse
from ..services.query_orchestrator import QueryOrchestrator

router = APIRouter(tags=["query"])
orchestrator = QueryOrchestrator()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    return await orchestrator.run(request)
