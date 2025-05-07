import itertools
import json
import os
from tqdm import tqdm
from rag import RAG
from datasets import load_dataset
from argparse import ArgumentParser



if __name__=="__main__":
    top_k = [10]
    retrieval_technique = ["none"]
    chunking_technique = "sentence_splitter"
    rerank = [True]

    combinations = list(itertools.product(top_k, retrieval_technique, rerank))
    grid = [{"top_k": k, "retrieval_technique": r, "rerank": re} for k, r, re in combinations]


    evaluation_dataset = load_dataset("IIC/RagQuAS", split="test")
    queries = evaluation_dataset["question"]
    references = evaluation_dataset["answer"]
    rag = RAG(chunking_technique=chunking_technique, generation_port=8000, retrieval_techniques_port=8000, embeddings_model="intfloat/multilingual-e5-small")

    folder_path = "rag_predictions"
    os.makedirs(folder_path, exist_ok=True) 
    
    for i in range(1,6):
        rag.upload_hf("IIC/RagQuAS", split="test", column=f"text_{i}")

    for config in tqdm(grid):
        rag.set_config(**config)

        filename = f"qwen3_{config['retrieval_technique']}_{chunking_technique}_rerank{config['rerank']}_top_k{config['top_k']}.json"
        filepath = os.path.join(folder_path, filename)

        if os.path.exists(filepath):
            print(f"Skipping {filename}, already exists.")
            continue


        documents = rag.retrieve_documents(queries=queries)
        responses = rag.generate_answers(queries=queries, documents=documents)
        
        dataset = [{
                    "user_input":query,
                    "retrieved_contexts":relevant_docs,
                    "response":response,
                    "reference":reference
                } 
                for query, relevant_docs, response, reference in zip(queries, documents, responses, references)
                ]
        
        config["chunking_technique"] = chunking_technique

        output_data = {
            "config": config,
            "dataset": dataset
        }



        with open(filepath, "w") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(f"Results saved to {filepath}")


    