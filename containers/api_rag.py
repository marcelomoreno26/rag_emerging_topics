import os
import shutil
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
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

# ========== Configuration ==========
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
TGI_MODEL = os.getenv("TGI_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
STORAGE_DIR = "./index_storage"

# ========== Init ==========
app = FastAPI()
generator = InferenceClient("http://tgi:80")
tokenizer = AutoTokenizer.from_pretrained(TGI_MODEL)

Settings.embed_model = TextEmbeddingsInference(
    model_name=os.getenv("TEI_MODEL", EMBEDDING_MODEL),
    base_url="http://tei:80",
    embed_batch_size=32
)
Settings.llm = None

reranker = SentenceTransformerRerank(model=RERANK_MODEL)
reranker.top_n = 5

index = None

# ========== Models ==========
class UploadRequest(BaseModel):
    texts: list[str]

# ========== Helper Functions ==========

def chunk_texts(texts: list[str]) -> list[str]:
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128, separator=" ", paragraph_separator="\n\n")
    return [chunk for doc in texts for chunk in splitter.split_text(doc)]

def persist_index(nodes):
    global index
    if os.path.exists(STORAGE_DIR) and os.listdir(STORAGE_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
        index.insert_nodes(nodes)
    else:
        index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=STORAGE_DIR)

def rerank_nodes(query: str, nodes):
    return reranker.postprocess_nodes(nodes=nodes, query_bundle=QueryBundle(query))

def build_prompt(query: str, context: str):
    system_message = """
    Eres un asistente, tu objetivo es proporcionar respuestas exhaustivas y detalladas, utilizando toda la información disponible en el contexto. Debes asegurarte de abordar completamente cada pregunta, desglosando todos los aspectos relevantes y considerando todas las posibles opciones y variaciones mencionadas en el contexto.
    Es esencial que no limites tus respuestas a generalizaciones breves; en lugar de eso, desglosa las posibles respuestas, soluciones, pasos o detalles importantes que el contexto proporcione. Si la pregunta se refiere a un proceso o procedimiento, asegúrate de explicar cada etapa y los posibles matices involucrados. Si hay diferentes alternativas o elementos a considerar, proporciona detalles sobre cada uno de ellos.
    Si no cuentas con suficiente información para ofrecer una respuesta completa, o si la pregunta no se relaciona con el contexto, responde: 'No tengo suficiente información para responder a eso.'
    Recuerda que tu propósito es ser lo más detallado y completo posible, sin omitir ninguna información relevante, con el fin de que el usuario reciba una respuesta clara, precisa y útil.
    """
    return tokenizer.apply_chat_template([
                                        {"role": "system", "content": system_message},
                                        {
                                            "role": "user",
                                            "content": (
                                                f"Contexto relevante:\n{context}\n\n"
                                                f"Pregunta: {query}\n\n"
                                                "Recuerda contestar a la pregunta de manera exhaustiva, completa y clara, "
                                                "incluyendo todos los detalles y ejemplos que encuentres en el contexto que sean relevantes."
                                            )
                                        }
                                    ], add_generation_prompt=True, tokenize=False)


# ========== Endpoints ==========

@app.post("/upload")
async def upload_documents(req: UploadRequest):
    try:
        if not req.texts:
            raise HTTPException(status_code=400, detail="No texts provided.")

        chunked_texts = chunk_texts(req.texts)
        nodes = [TextNode(text=text) for text in chunked_texts]
        persist_index(nodes)

        return {"message": "Index created", "nodes_count": len(nodes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading documents: {e}")

@app.post("/generate")
async def generate_text(request: Request):
    global index
    try:
        data = await request.json()
        query = data.get("new_message", {}).get("content")
        if not query:
            raise HTTPException(status_code=400, detail="'content' missing in 'new_message'.")

        if index is None:
            storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
            index = load_index_from_storage(storage_context)

        query_engine = index.as_query_engine(similarity_top_k=15)
        nodes = query_engine.retrieve(query)
        nodes = rerank_nodes(query, nodes)

        context = "".join([f"<doc>\n{node.text}</doc>" for node in nodes])
        prompt = build_prompt(query, context)

        response = generator.text_generation(
            prompt,
            max_new_tokens=2048,
            top_p=0.9,
            temperature=0.2,
            return_full_text=False
        )

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

