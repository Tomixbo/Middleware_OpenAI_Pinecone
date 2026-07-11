import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel

load_dotenv()

# uvicorn app:app --host 0.0.0.0 --port 10000


class QueryModel(BaseModel):
    query: str


class KnowledgeResult(BaseModel):
    text: str
    document: str | None = None
    paragraph: str | None = None
    score: float | None = None


class KnowledgeSearchOutput(BaseModel):
    results: list[KnowledgeResult]
    result_count: int


app = FastAPI(title="CAP CONSEILS RAG")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

index_hugo_name = os.getenv("PINECONE_INDEX_HUGO")
index_marie_name = os.getenv("PINECONE_INDEX_MARIE")
index_kate_name = os.getenv("PINECONE_INDEX_KATE")
index_cleon_name = os.getenv("PINECONE_INDEX_CLEON")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pc_v2 = Pinecone(api_key=os.getenv("PINECONE_API_KEY_v2"))

index_hugo = pc.Index(index_hugo_name)
index_marie = pc.Index(index_marie_name)
index_kate = pc.Index(index_kate_name)
index_cleon = pc.Index(index_cleon_name)
index_hugo_v2 = pc_v2.Index(index_hugo_name)
index_marie_v2 = pc_v2.Index(index_marie_name)
index_kate_v2 = pc_v2.Index(index_kate_name)
index_cleon_v2 = pc_v2.Index(index_cleon_name)

security = HTTPBearer()


def validate_token(
    http_auth_credentials: HTTPAuthorizationCredentials = Security(security),
):
    if http_auth_credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=403, detail="Invalid authentication scheme")

    token = http_auth_credentials.credentials
    if token != os.getenv("RENDER_API_TOKEN"):
        raise HTTPException(status_code=403, detail="Invalid token")


def search_index(index: Any, query: str, top_k: int) -> list[KnowledgeResult]:
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    embedding_response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    embedding = embedding_response.data[0].embedding

    response = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
    )

    results: list[KnowledgeResult] = []
    matches = getattr(response, "matches", None)
    if matches is None:
        matches = response.to_dict().get("matches", [])

    for match in matches:
        metadata = getattr(match, "metadata", None) or match.get("metadata", {})
        text = metadata.get("text")
        if isinstance(text, str) and text.strip():
            score = getattr(match, "score", None)
            if score is None and isinstance(match, dict):
                score = match.get("score")

            paragraph = metadata.get("paragraph")
            if paragraph is not None:
                paragraph = str(paragraph)

            results.append(
                KnowledgeResult(
                    text=text,
                    document=metadata.get("document"),
                    paragraph=paragraph,
                    score=score,
                )
            )

    return results


def build_search_output(index: Any, query: str, top_k: int) -> KnowledgeSearchOutput:
    results = search_index(index, query, top_k=top_k)
    return KnowledgeSearchOutput(
        results=results,
        result_count=len(results),
    )


@app.post("/hugo/")
async def get_context_hugo(
    query_data: QueryModel,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    return search_index(index_hugo, query_data.query, top_k=3)


@app.post("/hugo_v2/")
async def get_context_hugo_v2(
    query_data: QueryModel,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    return search_index(index_hugo_v2, query_data.query, top_k=10)


@app.post("/marie/")
async def get_context_marie(
    query_data: QueryModel,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    return search_index(index_marie, query_data.query, top_k=3)


@app.post("/marie_v2/")
async def get_context_marie_v2(
    query_data: QueryModel,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    return search_index(index_marie_v2, query_data.query, top_k=10)


@app.post("/kate/")
async def get_context_kate(
    query_data: QueryModel,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    return search_index(index_kate, query_data.query, top_k=3)


@app.post("/kate_v2/")
async def get_context_kate_v2(
    query_data: QueryModel,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    return search_index(index_kate_v2, query_data.query, top_k=10)


@app.post("/cleon/")
async def get_context_cleon(
    query_data: QueryModel,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    return search_index(index_cleon, query_data.query, top_k=3)


@app.post("/cleon_v2/")
async def get_context_cleon_v2(
    query_data: QueryModel,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    return search_index(index_cleon_v2, query_data.query, top_k=10)


hugo_mcp = FastMCP(
    name="HUGO Knowledge",
    instructions=(
        "Use this server before answering questions about French accounting "
        "or taxation for CAP CONSEILS."
    ),
    json_response=True,
    streamable_http_path="/",
)


@hugo_mcp.tool()
def search_hugo_knowledge(query: str) -> KnowledgeSearchOutput:
    """
    Search CAP CONSEILS internal knowledge on French accounting and tax.
    Use before answering on bookkeeping, tax, VAT, corporate income tax,
    depreciation, provisions, expenses, revenue, tax filings, or tax
    obligations. Do not use for labor law, corporate law, or audit.
    """
    return build_search_output(index_hugo_v2, query, top_k=10)


kate_mcp = FastMCP(
    name="KATE Knowledge",
    instructions=(
        "Use this server before answering questions about French labor "
        "and employment law for CAP CONSEILS."
    ),
    json_response=True,
    streamable_http_path="/",
)


@kate_mcp.tool()
def search_kate_knowledge(query: str) -> KnowledgeSearchOutput:
    """
    Search CAP CONSEILS internal knowledge on French labor and employment law.
    Use before answering on employment contracts, payroll, leave, working time,
    absences, disciplinary actions, termination, dismissals, employer-employee
    relations, or social obligations. Do not use for accounting, tax,
    corporate law, or audit.
    """
    return build_search_output(index_kate_v2, query, top_k=10)


marie_mcp = FastMCP(
    name="MARIE Knowledge",
    instructions=(
        "Use this server before answering questions about French corporate "
        "law for CAP CONSEILS."
    ),
    json_response=True,
    streamable_http_path="/",
)


@marie_mcp.tool()
def search_marie_knowledge(query: str) -> KnowledgeSearchOutput:
    """
    Search CAP CONSEILS internal knowledge on French corporate law.
    Use before answering on company formation, governance, bylaws,
    shareholders, directors, meetings, annual accounts approval,
    capital transactions, share transfers, transformations, mergers,
    dissolutions, or corporate filings. Do not use for accounting,
    tax, labor law, or audit.
    """
    return build_search_output(index_marie_v2, query, top_k=10)


cleon_mcp = FastMCP(
    name="CLEON Knowledge",
    instructions=(
        "Use this server before answering questions about audit and statutory "
        "audit in France for CAP CONSEILS."
    ),
    json_response=True,
    streamable_http_path="/",
)


@cleon_mcp.tool()
def search_cleon_knowledge(query: str) -> KnowledgeSearchOutput:
    """
    Search CAP CONSEILS internal knowledge on audit and statutory audit in
    France. Use before answering on audit standards, engagement planning,
    risk assessment, internal control, audit procedures, audit evidence,
    materiality, closing work, auditor reports, or legal audit obligations.
    Do not use for routine accounting, tax, labor law, or corporate law.
    """
    return build_search_output(index_cleon_v2, query, top_k=10)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(hugo_mcp.session_manager.run())
        await stack.enter_async_context(kate_mcp.session_manager.run())
        await stack.enter_async_context(marie_mcp.session_manager.run())
        await stack.enter_async_context(cleon_mcp.session_manager.run())
        yield


app.router.lifespan_context = lifespan
app.mount("/mcp/hugo", hugo_mcp.streamable_http_app())
app.mount("/mcp/kate", kate_mcp.streamable_http_app())
app.mount("/mcp/marie", marie_mcp.streamable_http_app())
app.mount("/mcp/cleon", cleon_mcp.streamable_http_app())
