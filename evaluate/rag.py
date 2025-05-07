import requests
import os
import shutil
import atexit
from datasets import load_dataset
from llama_index.core import (
    Settings,
    StorageContext, 
    load_index_from_storage,
    VectorStoreIndex,
    Document
)
from llama_index.core.schema import TextNode, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser



class RAG:
    def __init__(self, chunking_technique: str,embeddings_model: str= "BAAI/bge-m3", reranker_model: str="jinaai/jina-reranker-v2-base-multilingual", storage_dir: str="./index_storage", generation_port: int = 8000, retrieval_techniques_port: int=8001):
        """
        Initialize the VectorDB object.
        
        Parameters
        ----------
        embeddings_model : str
            The model name for HuggingFace embeddings.
        storage_dir : str
            Directory where the index data will be stored.
        """
        self.chunking_technique = chunking_technique
        self.generation_port = generation_port
        self.retrieval_techniques_port = retrieval_techniques_port
        self.storage_dir = storage_dir
        Settings.embed_model = HuggingFaceEmbedding(model_name=embeddings_model)
        Settings.llm = None
        self.reranker = SentenceTransformerRerank(
                        model=reranker_model)
        os.makedirs(self.storage_dir, exist_ok=True)
        atexit.register(self._cleanup)


    def _cleanup(self):
        """
        Clean up by deleting the storage directory when the program exits.
        """
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)

    
    def _chunking(self, documents: list[str]) -> list[str]:
        """
        Chunk a list of documents into smaller segments based on the specified chunking technique.

        Parameters
        ----------
        documents : list of str
            List of documents to be chunked.

        Returns
        -------
        list of list of str
            A list where each element is a list of strings representing the chunks for a document.
        """
        if self.chunking_technique == "semantic":
            documents = [Document(text=doc) for doc in documents]

            splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                embed_model=Settings.embed_model,
                breakpoint_percentile_threshold=80
            )

            nodes = splitter.build_semantic_nodes_from_documents(documents)

            return [node.text for node in nodes]
        
        elif self.chunking_technique == "sentence_splitter":
            splitter = SentenceSplitter(
                chunk_size=512, #1024
                chunk_overlap=64,  #128 
                separator=" ",
                paragraph_separator="\n\n"
            )


            return [text for doc in documents for text in splitter.split_text(doc)]

    
    def _upload_texts(self, texts: list[str]):
        """
        Upload a list of texts into the vector database.

        Parameters
        ----------
        texts : list of str
            List of texts to be uploaded.
        """
        nodes = [TextNode(text=text) for text in texts]

        if os.listdir(self.storage_dir):
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            self.index = load_index_from_storage(storage_context)
            self.index.insert_nodes(nodes)
        else:
            self.index = VectorStoreIndex(nodes)

        self.index.storage_context.persist(persist_dir=self.storage_dir)
        
    
    def _inference(self, messages: list, port: int, sampling_params: dict=None):
        """
        Sends a request to the API to generate content based on the provided messages.

        Parameters
        ----------
        messages : list of dict
            A list of message dictionaries with "role" (str) and "content" (str).
        
        port : int
            The port number where the API is hosted.

        Returns
        -------
        list or None
            A list of predictions from the API, or None if the request fails.
        """
        if sampling_params:
            response = requests.post(f"http://localhost:{port}/generate/", json={"messages": messages})
        else:
            response = requests.post(f"http://localhost:{port}/generate/", json={"messages": messages, "sampling_params":sampling_params})

        data = response.json()
        if not data.get("success"):
            print(f"API inference failed.")
            return None
        return data.get("predictions")
    

    def _get_hyde(self, queries: list[str]) -> list[str]:
        """
        Generate synthetic answers for queries to be used for retrieval (HyDE method).

        Parameters
        ----------
        queries : list of str
            The original user queries.

        Returns
        -------
        list of str
            List of generated answers corresponding to each query.
        """
        system_message = "Eres un experto en la redacción de respuestas detalladas sobre diversos temas, capaz de abordar cualquier consulta con precisión y profundidad."
        prompt = "Contesta la siguiente pregunta de forma detallada, proporcionando toda la información relevante. Devuelve únicamente la respuesta a la pregunta: {}"

        
        messages = [[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": prompt.format(query)
            }
            ] 
            for query in queries]
        

        return self._inference(messages, self.retrieval_techniques_port)


    def _get_query_rewriting(self, queries: list[str]) -> list[str]:
        """
        Rewrite queries to be more precise, complete, and clear.

        Parameters
        ----------
        queries : list of str
            Original user queries.

        Returns
        -------
        list of str
            Rewritten queries.
        """
        system_message = """
                        Eres un experto en procesamiento de lenguaje natural. Tu tarea es mejorar consultas de usuario para que sean más claras, precisas y relevantes, sin cambiar su significado original. Aplica las siguientes estrategias al reformular:
                        1. Expande abreviaturas y siglas (por ejemplo, "NLP" → "procesamiento de lenguaje natural").
                        2. Corrige errores ortográficos y gramaticales.
                        3. Enriquece semánticamente usando sinónimos o términos relacionados para mejorar la recuperación.
                        4. Aclara entidades si son ambiguas (por ejemplo, "Apple" → "Apple, la empresa tecnológica").
                        5. Haz la consulta más útil añadiendo ejemplos, pasos, instrucciones o detalles específicos, si esto puede mejorar la recuperación sin cambiar el objetivo original.
                        Si la consulta ya es clara y no se puede mejorar, devuélvela sin cambios.
                        """
        prompt = "Reformula la siguiente consulta para que sea más clara, precisa y relevante, manteniendo su significado. Si consideras que no se puede mejorar, devuelve la consulta original: {}"


        messages = [
            [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": prompt.format(query)
                }
            ]
            for query in queries
        ]

        return self._inference(messages, self.retrieval_techniques_port)

    
    def _rerank(self, queries: list[str], list_nodes: list[list]):
        """
        Rerank retrieved nodes using a reranker model.

        Parameters
        ----------
        queries : list of str
            List of queries.

        list_nodes : list of list
            Retrieved nodes for each query.

        Returns
        -------
        list of list
            Reranked list of nodes per query.
        """
        reranked_list_nodes = []
        for query, nodes in zip(queries, list_nodes):
            query_bundle = QueryBundle(query)
            reranked_nodes = self.reranker.postprocess_nodes(nodes=nodes, query_bundle=query_bundle)
            reranked_list_nodes.append(reranked_nodes)


        return reranked_list_nodes
    

    def set_config(self, retrieval_technique: str, rerank: bool, top_k: int):
        """
        Set the retrieval configuration for document querying.

        Parameters
        ----------
        retrieval_technique : str
            The method to use for query transformation. Options: "none", "HyDE", "query_rewriting".

        rerank : bool
            Whether to apply reranking to the retrieved documents.

        top_k : int
            The number of top documents to return. If reranking is enabled, this determines
            the number of reranked results (with a higher initial k=15 for retrieval).
        """
        self.retrieval_technique = retrieval_technique
        self.rerank = rerank
        if self.rerank:
            self.top_k = 20
            self.reranker.top_n = top_k
        else:
            self.top_k = top_k


    def upload_hf(self, dataset_name: str, split:str, column: str):
        """
        Upload a specified column of a HuggingFace dataset to the vector database.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load from HuggingFace.
        split : str
            The dataset split to load (e.g., 'train', 'validation').
        column : str
            The column in the dataset to extract for indexing.
        """
        dataset = load_dataset(dataset_name, split=split)
        texts = dataset[column]
        texts = list(set(text.strip() for text in texts if text is not None and text.strip())) # remove duplicate, empty and None.
        chunked_texts = self._chunking(texts)
        self._upload_texts(texts=chunked_texts)
        print(f"Column '{column}' from dataset '{dataset_name}' uploaded correctly.")


    def retrieve_documents(self, queries: list[str]) -> list[str]:
        """
        Retrieve documents for multiple queries in a batch.

        Parameters
        ----------
        queries : list of str
            List of query messages.

        Returns
        -------
        list of str
            List of concatenated strings of retrieved documents for each query.
        """
        query_engine = self.index.as_query_engine(streaming=False, similarity_top_k=self.top_k)

        if self.retrieval_technique == "HyDE":
            messages = self._get_hyde(queries=queries)
        elif self.retrieval_technique == "query_rewriting":
            messages = self._get_query_rewriting(queries=queries)
        elif self.retrieval_technique == "none":
            messages = queries
        
        
        list_nodes = [query_engine.retrieve(message) for message in messages]

        if self.rerank:
            list_nodes = self._rerank(queries=queries, list_nodes=list_nodes)

        return ["".join([f"<doc>\n{node.text}</doc>" for node in retrieved]) for retrieved in list_nodes]



    def generate_answers(self, queries: list[str], documents: list[str], sampling_params: dict=None) -> list[str]:
        """
        Generate detailed and comprehensive answers based on the given queries and their retrieved documents.

        Parameters
        ----------
        queries : list of str
            List of user queries to be answered.

        documents : list of str
            List of corresponding document contexts for each query, used to generate grounded responses.

        sampling_params : dict, optional
            Sampling parameters for generation (e.g., temperature, top_p). If not provided, defaults are used.

        Returns
        -------
        list of str
            A list of generated answers, each grounded in the corresponding document context.
        """
        system_message = """
        Eres un asistente, tu objetivo es proporcionar respuestas exhaustivas y detalladas, utilizando toda la información disponible en el contexto. Debes asegurarte de abordar completamente cada pregunta, desglosando todos los aspectos relevantes y considerando todas las posibles opciones y variaciones mencionadas en el contexto.
        Es esencial que no limites tus respuestas a generalizaciones breves; en lugar de eso, desglosa las posibles respuestas, soluciones, pasos o detalles importantes que el contexto proporcione. Si la pregunta se refiere a un proceso o procedimiento, asegúrate de explicar cada etapa y los posibles matices involucrados. Si hay diferentes alternativas o elementos a considerar, proporciona detalles sobre cada uno de ellos.
        Si no cuentas con suficiente información para ofrecer una respuesta completa, o si la pregunta no se relaciona con el contexto, responde: 'No tengo suficiente información para responder a eso.'
        Recuerda que tu propósito es ser lo más detallado y completo posible, sin omitir ninguna información relevante, con el fin de que el usuario reciba una respuesta clara, precisa y útil.
        """

        messages = []

        for query, query_documents in zip(queries, documents):
            messages.append([
                {
                    "role":"system",
                    "content": system_message
                },
                {
                    "role":"input",
                    "content": query_documents
                },
                {
                    "role":"user",
                    "content": query + "\nRecuerda contestar a la pregunta de manera exhaustiva, completa y clara, incluyendo todos los detalles y ejemplos que encuentres en el contexto que sean relevantes para poder responder a la consulta presentada en su totalidad. /no_think"
                }
            ])
        

        return self._inference(messages=messages, port=self.generation_port, sampling_params=sampling_params)




