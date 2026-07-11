import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

import pinecone

load_dotenv()

# uvicorn app:app --host 0.0.0.0 --port 10000
app = FastAPI()

# Setup environment variables
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
environment = os.getenv("PINECONE_ENV")
index_hugo_name = os.getenv("PINECONE_INDEX_HUGO")
index_marie_name = os.getenv("PINECONE_INDEX_MARIE")
index_kate_name = os.getenv("PINECONE_INDEX_KATE")
index_cleon_name = os.getenv("PINECONE_INDEX_CLEON")

# Initialize pinecone client
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


# Middleware to secure HTTP endpoint
security = HTTPBearer()


def validate_token(
    http_auth_credentials: HTTPAuthorizationCredentials = Security(security),
):
    if http_auth_credentials.scheme.lower() == "bearer":
        token = http_auth_credentials.credentials
        if token != os.getenv("RENDER_API_TOKEN"):
            raise HTTPException(status_code=403, detail="Invalid token")
    else:
        raise HTTPException(status_code=403, detail="Invalid authentication scheme")


class QueryModel(BaseModel):
    query: str


class MCPToolCallArguments(BaseModel):
    query: str


def search_index(index, query: str, top_k: int):
    res = openai_client.embeddings.create(
        input=[query], model="text-embedding-3-small"
    )
    embedding = res.data[0].embedding
    results = index.query(
        vector=embedding, top_k=top_k, include_metadata=True
    ).to_dict()
    return [match["metadata"]["text"] for match in results["matches"]]


def build_initialize_response(request_id):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "protocolVersion": "2025-03-26",
            "serverInfo": {
                "name": "middleware-openai-pinecone",
                "version": "1.0.0",
            },
            "capabilities": {
                "tools": {},
            },
        },
    }


def build_tools_list_response(request_id, tool_name: str, description: str):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "tools": [
                {
                    "name": tool_name,
                    "description": description,
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Knowledge search query",
                            }
                        },
                        "required": ["query"],
                    },
                }
            ]
        },
    }


def build_tools_call_response(request_id, tool_name: str, context):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": "\n\n".join(context),
                }
            ],
            "structuredContent": {
                "tool": tool_name,
                "results": context,
            },
        },
    }


def build_method_not_found_response(request_id, method: str):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32601,
            "message": f"Unsupported MCP method: {method}",
        },
    }


async def handle_mcp_request(
    payload: dict,
    tool_name: str,
    description: str,
    query_handler,
):
    request_id = payload.get("id")
    method = payload.get("method")

    if method == "initialize":
        return build_initialize_response(request_id)

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return build_tools_list_response(request_id, tool_name, description)

    if method == "tools/call":
        params = payload.get("params", {})
        if params.get("name") != tool_name:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported tool: {params.get('name')}",
            )

        arguments = MCPToolCallArguments(**params.get("arguments", {}))
        context = await query_handler(QueryModel(query=arguments.query))
        return build_tools_call_response(request_id, tool_name, context)

    return build_method_not_found_response(request_id, method)


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


@app.post("/mcp/hugo")
async def mcp_hugo(
    payload: dict,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    return await handle_mcp_request(
        payload=payload,
        tool_name="search_hugo_knowledge",
        description="Search CAP CONSEILS internal knowledge on French accounting and tax. Use before answering on bookkeeping, tax, VAT, corporate income tax, depreciation, provisions, expenses, revenue, tax filings, or tax obligations. Returns relevant passages from sources such as Memento Fiscal 2026 and Memento Comptabilite 2026. Do not use for labor law, corporate law, or audit.",
        query_handler=get_context_hugo_v2,
    )


@app.post("/mcp/marie")
async def mcp_marie(
    payload: dict,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    return await handle_mcp_request(
        payload=payload,
        tool_name="search_marie_knowledge",
        description="Search CAP CONSEILS internal knowledge on French corporate law. Use before answering on company formation, governance, bylaws, shareholders, directors, meetings, annual accounts approval, capital transactions, share transfers, transformations, mergers, dissolutions, or corporate filings. Do not use for accounting, tax, labor law, or audit.",
        query_handler=get_context_marie_v2,
    )


@app.post("/mcp/kate")
async def mcp_kate(
    payload: dict,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    return await handle_mcp_request(
        payload=payload,
        tool_name="search_kate_knowledge",
        description="Search CAP CONSEILS internal knowledge on French labor and employment law. Use before answering on employment contracts, payroll, leave, working time, absences, disciplinary actions, termination, dismissals, employer-employee relations, or social obligations. Do not use for accounting, tax, corporate law, or audit.",
        query_handler=get_context_kate_v2,
    )


@app.post("/mcp/cleon")
async def mcp_cleon(
    payload: dict,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    return await handle_mcp_request(
        payload=payload,
        tool_name="search_cleon_knowledge",
        description="Search CAP CONSEILS internal knowledge on audit and statutory audit in France. Use before answering on audit standards, engagement planning, risk assessment, internal control, audit procedures, audit evidence, materiality, closing work, auditor reports, or legal audit obligations. Do not use for routine accounting, tax, labor law, or corporate law.",
        query_handler=get_context_cleon_v2,
    )
