import os
from dotenv import load_dotenv
from fastapi import FastAPI
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
pinecone_api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENV")
index_hugo_name = os.getenv("PINECONE_INDEX_HUGO")
index_marie_name = os.getenv("PINECONE_INDEX_MARIE")

# Initialize pinecone client
pc = Pinecone(api_key=os.getenv(pinecone_api_key))
index_hugo = pc.Index(index_hugo_name)
index_marie = pc.Index(index_marie_name)


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

@app.post("/hugo/")
async def get_context_hugo(
    query_data: QueryModel,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    # convert query to embeddings
    res = openai_client.embeddings.create(
        input=[query_data.query], model="text-embedding-3-small"
    )
    embedding = res.data[0].embedding
    # Search for matching Vectors
    results = index_hugo.query(vector=embedding, top_k=3, include_metadata=True).to_dict()
    # Filter out metadata fron search result
    context = [match["metadata"]["text"] for match in results["matches"]]
    # Retrun context
    return context

@app.post("/marie/")
async def get_context_marie(
    query_data: QueryModel,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    # convert query to embeddings
    res = openai_client.embeddings.create(
        input=[query_data.query], model="text-embedding-3-small"
    )
    embedding = res.data[0].embedding
    # Search for matching Vectors
    results = index_marie.query(vector=embedding, top_k=3, include_metadata=True).to_dict()
    # Filter out metadata fron search result
    context = [match["metadata"]["text"] for match in results["matches"]]
    # Retrun context
    return context

