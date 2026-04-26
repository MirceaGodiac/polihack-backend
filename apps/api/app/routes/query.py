from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..schemas import QueryGraphResponse, QueryRequest, QueryResponse
from ..services.query_orchestrator import QueryOrchestrator
from ..services.query_response_store import QueryResponseStore

router = APIRouter(tags=["query"])
orchestrator = QueryOrchestrator()
query_response_store = QueryResponseStore()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    response = await orchestrator.run(request)
    query_response_store.save(response)
    return response


@router.get("/query/{query_id}", response_model=QueryResponse)
async def get_query(query_id: str) -> QueryResponse | JSONResponse:
    response = query_response_store.get(query_id)
    if response is None:
        return _query_not_found(query_id)
    return response


@router.get("/query/{query_id}/graph", response_model=QueryGraphResponse)
async def get_query_graph(query_id: str) -> QueryGraphResponse | JSONResponse:
    response = query_response_store.get_graph(query_id)
    if response is None:
        return _query_not_found(query_id)
    return response


def _query_not_found(query_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error_code": "query_not_found",
            "message": "Query response not found for query_id",
            "query_id": query_id,
        },
    )
