import os
import shutil
import re
import requests
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank

# ========== Configuration ==========
TEI_MODEL = os.getenv("TEI_MODEL","intfloat/multilingual-e5-small")
RERANK_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
TGI_MODEL = os.getenv("TGI_MODEL", "./Qwen3-0.6B-Q4_K_M.gguf:/models/model.gguf")
STORAGE_DIR = "./index_storage"

# ========== Init ==========
app = FastAPI()

generator_endpoint = "http://llamacpp:8080/v1/chat/completions"

Settings.embed_model = TextEmbeddingsInference(
    model_name=TEI_MODEL,
    base_url="http://tei:80",
    embed_batch_size=32,
    timeout=60
)
Settings.llm = None

reranker = SentenceTransformerRerank(model=RERANK_MODEL)
reranker.top_n = 10

index = None

# ========== Models ==========
class UploadRequest(BaseModel):
    texts: list[str]

# ========== Helper Functions ==========

def chunk_texts(texts: list[str]) -> list[str]:
    """
    Splits a list of input texts into smaller chunks using sentence-level segmentation.

    Args:
        texts (list[str]): A list of strings to be chunked.

    Returns:
        list[str]: A list of smaller text chunks.
    """
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64, separator=" ", paragraph_separator="\n\n")
    return [chunk for doc in texts for chunk in splitter.split_text(doc)]

def persist_index(nodes):
    """
    Persists the given nodes into a vector store index. If an index already exists, it will be loaded and updated.

    Args:
        nodes (list[TextNode]): List of TextNode objects to be inserted into the index.

    Side Effects:
        Updates or creates a persisted vector index on disk.
    """
    global index
    if os.path.exists(STORAGE_DIR) and os.listdir(STORAGE_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
        index.insert_nodes(nodes)
    else:
        index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=STORAGE_DIR)

def rerank_nodes(query: str, nodes):
    """
    Reranks a list of retrieved nodes based on relevance to the given query using a sentence transformer reranker.

    Args:
        query (str): The input query to compare relevance.
        nodes (list[Node]): List of retrieved nodes to rerank.

    Returns:
        list[Node]: A reranked list of nodes with the most relevant ones at the top.
    """
    return reranker.postprocess_nodes(nodes=nodes, query_bundle=QueryBundle(query))

def build_message(query: str, context: str):
    """
    Builds a structured prompt message for the LLM with system instructions, context, and user query.

    Args:
        query (str): The user’s question.
        context (str): The relevant context retrieved from the vector index.

    Returns:
        list[dict]: A list of message objects formatted for the chat completion API.
    """

    system_message = """
    Eres un asistente, tu objetivo es proporcionar respuestas exhaustivas y detalladas, utilizando toda la información disponible en el contexto. Debes asegurarte de abordar completamente cada pregunta, desglosando todos los aspectos relevantes y considerando todas las posibles opciones y variaciones mencionadas en el contexto.
    Es esencial que no limites tus respuestas a generalizaciones breves; en lugar de eso, desglosa las posibles respuestas, soluciones, pasos o detalles importantes que el contexto proporcione. Si la pregunta se refiere a un proceso o procedimiento, asegúrate de explicar cada etapa y los posibles matices involucrados. Si hay diferentes alternativas o elementos a considerar, proporciona detalles sobre cada uno de ellos.
    Si no cuentas con suficiente información para ofrecer una respuesta completa, o si la pregunta no se relaciona con el contexto, responde: 'No tengo suficiente información para responder a eso.'
    Recuerda que tu propósito es ser lo más detallado y completo posible, sin omitir ninguna información relevante, con el fin de que el usuario reciba una respuesta clara, precisa y útil.
    """
    
    return [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": (
                f"Contexto relevante:\n{context}\n\n"
                f"Pregunta: {query}\n\n /no_think"
            )
        }
    ]

def clean_response(raw_response: str) -> str:
    """
    Cleans the raw LLM output by removing any intermediate <think> tags and whitespace.

    Args:
        raw_response (str): Raw response text returned by the language model.

    Returns:
        str: Cleaned and stripped response.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)

    return cleaned.strip()



# ========== Endpoints ==========

@app.post("/upload")
async def upload_documents(req: UploadRequest):
    """
    Upload and index documents by splitting them into chunks and storing them in a vector index.

    Args:
        req (UploadRequest): A request object containing a list of text strings to be processed and indexed.

    Returns:
        dict: A confirmation message and the number of nodes created.

    Raises:
        HTTPException: If no texts are provided or an internal error occurs during processing.
    """
    try:
        print("📤 Received upload request.")
        if not req.texts:
            print("⚠️ No texts provided in request.")
            raise HTTPException(status_code=400, detail="No texts provided.")

        print(f"📝 Number of input documents: {len(req.texts)}")
        chunked_texts = chunk_texts(req.texts)
        print(f"🔪 Texts chunked into {len(chunked_texts)} segments.")

        nodes = [TextNode(text=text) for text in chunked_texts]
        print("📦 Creating index and persisting nodes...")
        persist_index(nodes)

        print("✅ Index creation complete.")
        return {"message": "Vector index successfully created", "nodes_count": len(nodes)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading documents: {e}")


@app.post("/generate")
async def generate_text(request: Request):
    """
    Generate a detailed response to a query using a retrieval-augmented generation (RAG) workflow.

    The function:
    - Loads or creates a vector index.
    - Retrieves top-k similar documents.
    - Reranks them for relevance.
    - Builds a prompt including the query and context.
    - Sends it to a language model for generation.
    - Cleans and returns the final response.

    Args:
        request (Request): A FastAPI request containing the query under `new_message.content`.

    Returns:
        dict: Generated text and the list of contexts (source chunks) used.

    Raises:
        HTTPException: If the query is missing or any part of the RAG process fails.
    """
    global index
    try:
        print("🔍 Received generate request.")
        data = await request.json()
        query = data.get("new_message", {}).get("content")
        print(f"🧠 Query: {query}")

        if not query:
            raise HTTPException(status_code=400, detail="'content' missing in 'new_message'.")

        if index is None:
            print("📂 Loading index from storage...")
            storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
            index = load_index_from_storage(storage_context)

        print("🔎 Running similarity search...")
        query_engine = index.as_query_engine(similarity_top_k=20)
        print("🔎 Retrieve Nodes...")
        nodes = query_engine.retrieve(query)
        print(f"📄 Retrieved {len(nodes)} nodes")

        print("📊 Running reranker...")
        nodes = rerank_nodes(query, nodes)
        print(f"✅ Reranked to {len(nodes)} nodes")

        context = "".join([f"<doc>\n{node.text}</doc>" for node in nodes])
        print(f"📚 Context size: {len(context)} characters")

        payload = {
            "model": TGI_MODEL, 
            "messages": build_message(query, context),
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 1024,
            "repetition_penalty": 1.05,
            "min_p": 0.1
        }

        print("🚀 Sending payload to generation endpoint...")
        request_response = requests.post(
            generator_endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120
        )
        print("📥 Response received.")
        request_response.raise_for_status()

        response_data = request_response.json()
        response = clean_response(response_data["choices"][0]["message"]["content"])

        return {"generated_text": response, "contexts": [node.text for node in nodes]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")


@app.get("/")
def read_root():
    """
    Root endpoint to check API status.
    Returns:
        JSON response indicating API health.
    """
    return {"message": "RAG API is running successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

