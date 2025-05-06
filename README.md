

## 🔍 RAG System Evaluation

To determine the best-performing configuration for our Retrieval-Augmented Generation (RAG) system, we evaluated 24 different combinations of retrieval strategies, reranking, top-k document settings, and chunking techniques using the [**LlamaIndex** framework](https://docs.llamaindex.ai/en/stable/). The exact code implementation for evaluation and corresponding files can be found inside the **`evaluate`** folder. This folder also contains a separate `README.md` file to understand how the elements inside work together.

### 📚 Evaluation Dataset

For evaluating the performance of the RAG system, we used the [**RagQuAS** dataset](https://huggingface.co/datasets/IIC/RagQuAS), a benchmark specifically designed for **Retrieval-Augmented Generation in Spanish**.

#### 📝 Dataset Overview

RagQuAS is a **Spanish-language dataset** created to evaluate RAG systems. It contains **201 question-answer pairs** covering a **wide variety of domains** such as hobbies, linguistics, pets and health among others.

Each entry includes:

* A **question** (used as the input query),
* An **answer** (used as the ground truth for evaluation),
* Multiple **context passages** (`text_1` to `text_5`), which contain the complete source documents relevant to the question.

#### ⚙️ Data Usage for Evaluation

In our evaluation setup:

* The `question` field was used as the query for the RAG system.
* The `answer` field was used as the ground truth for measuring performance.
* The fields `text_1` to `text_5` were added to the vector database as **retrieval corpus**.

Though the dataset also provides cleaned versions of the context passages (`context_i`), we opted to use the **full original texts** (`text_i`) for retrieval. This choice was made to better reflect real-world scenarios, where documents are retrieved in their complete form, providing a more adaptable evaluation environment.

#### 🔎 Context Diversity & Challenge
One of the unique challenges of this dataset is that each domain or topic has its own distinct set of texts, meaning that the texts stored in the vector database include irrelevant documents for each specific query. For example, when answering a question about "health," the system will also search through documents from unrelated domains, such as "astronomy" or "cars."

This setup makes the retrieval task more difficult because the RAG system must effectively distinguish relevant from irrelevant information, which challenges its retrieval and generation capabilities. This helps to better evaluate how well the RAG system can handle real-world scenarios, where it may have to sift through a large variety of unrelated documents to find the right context for a specific query.



### 🤖 Language Model: LeNIA-1.5B

For the initial test we will use [**LeNIA-1.5B**](https://huggingface.co/LenguajeNaturalAI/leniachat-qwen2-1.5B-v0), a small (1.5B parameters) Spanish language model built on the **Qwen2 architecture**, for generation in the RAG system. LeNIA was trained on Spanish instruction and QA data to follow prompts naturally and effectively.

To keep things lightweight and efficient, the model was [**quantized to 4-bit (GGUF 4\_K\_M)**](https://huggingface.co/QuantFactory/leniachat-qwen2-1.5B-v0-GGUF), making it faster to run with minimal performance loss.



### 🧠 Embedding Model

We used the [**BGE-M3**](https://huggingface.co/BAAI/bge-m3) model to embed both documents and queries. This multilingual model performs well in Spanish and supports high-quality semantic retrieval using cosine similarity.

### 🔁 Retrieval Techniques

Three retrieval strategies were tested:

* **None** – Use the original query directly.
* **HyDE** – Generate a hypothetical answer using an LLM and retrieve based on that.
* **Query Rewriting** – Improve the query using an LLM before retrieval.

### 🧱 Chunking Techniques

Two chunking methods were used to prepare the documents:

* **Sentence Splitter** – Splits the content based on a specified number of tokens (**1024**) with an overlap (**128 tokens**), prioritizing sentence boundaries. Overlap helps preserve context across chunks, while sentence boundaries improve chunk coherence for better retrieval and generation.
* **Semantic Chunking** – Uses embedding similarity to create coherent semantic chunks, with a **95th percentile threshold** to define breakpoints.

### 🔢 Top-k Retrieval

For each configuration, we retrieved either **5 or 10 documents**. When **reranking was enabled**, we first retrieved the **top 15 documents**, reranked them using a dedicated model, and then selected the **top-k (5 or 10)** for use in generation.

### 🧹 Reranking

Reranking was tested in both on and off settings:

* **With Rerank** – Used the [**Jina Base v2 multilingual reranker**](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) to reorder the top 15 retrieved documents.
* **Without Rerank** – Used the top-k documents directly, based on similarity score.

### 🧪 Evaluation Setup

In total, **24 configurations** were evaluated across:

* 3 retrieval techniques
* 2 top-k values (5, 10)
* 2 rerank options (True, False)
* 2 chunking techniques (sentence splitter and semantic with 95% threshold)


### 📏 Evaluation Metrics (via RAGAS)
We used [**RAGAS**](https://github.com/explodinggradients/ragas) (Retrieval-Augmented Generation Assessment) to automatically evaluate each configuration by comparing with the ground truth and contexts retrieved.

In RAGAS an LLM is used to calculate various aspects of the RAG. The [**Llama 3.1 8B model**](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) was used for this evaluation. Each system was evaluated using the following metrics:

* **Answer Relevancy** – Measures how well the answer matches the ground truth in terms of meaning and accuracy.
* **Context Precision** – The proportion of the retrieved context that is actually relevant to the question.
* **Context Recall** – How much of the relevant context was successfully retrieved.
* **Faithfulness** – Assesses whether claims made in the generated answer are actually supported by the retrieved context.

These metrics are combined into an **overall average score** to compare configurations.

## 📊 RAG Configuration results

| retrieval\_technique | chunking\_technique | rerank | top\_k | answer\_relevancy | context\_precision | faithfulness | context\_recall | average\_score |
| -------------------- | ------------------- | ------ | ------ | ----------------- | ------------------ | ------------ | --------------- | -------------- |
| none                 | sentence\_splitter  | True   | 5      | 0.80           | **0.97**           | 0.75         | 0.93            | **0.8625**     |
| HyDE                 | sentence\_splitter  | True   | 10     | 0.79              | 0.96               | **0.76**     | 0.94            | **0.8625**     |
| none                 | sentence\_splitter  | False  | 10     | 0.79              | **0.97**           | 0.75         | 0.94            | **0.8625**     |
| none                 | semantic            | True   | 5      | 0.80           | 0.96               | 0.74         | 0.94        | 0.8550          |
| HyDE                 | semantic            | True   | 5      | 0.81              | 0.96               | 0.72         | 0.94            | 0.8575         |
| HyDE                 | sentence\_splitter  | True   | 5      | 0.79              | **0.97**           | 0.74         | 0.93            | 0.8575         |
| none                 | semantic            | True   | 10     | 0.81              | 0.94               | 0.73         | **0.95**        | 0.8575         |
| query\_rewriting     | sentence\_splitter  | True   | 5      | 0.79              | 0.96               | **0.76**     | 0.92            | 0.8575         |
| HyDE                 | semantic            | True   | 10     | 0.80               | 0.94               | 0.75         | 0.93            | 0.8550          |
| none                 | semantic            | False  | 5      | 0.80          | 0.96               | 0.74         | 0.92            | 0.8550          |
| none                 | semantic            | True   | 5      | 0.78              | 0.96               | 0.74         | 0.94            | 0.8550          |
| query\_rewriting     | sentence\_splitter  | True   | 10     | 0.80               | 0.94               | 0.74         | 0.94            | 0.8550          |
| none                 | semantic            | False  | 10     | 0.79              | 0.95               | **0.76**     | 0.92            | 0.8550          |
| query\_rewriting     | semantic            | True   | 5      | 0.80               | 0.95               | 0.71         | **0.95**        | 0.8525         |
| none                 | sentence\_splitter  | True   | 10     | **0.81**          | 0.96               | 0.71         | 0.93            | 0.8525         |
| HyDE                 | semantic            | False  | 10     | 0.78              | 0.93               | 0.74         | 0.94            | 0.8475         |
| HyDE                 | sentence\_splitter  | False  | 5      | 0.79              | 0.95               | 0.73         | 0.92            | 0.8475         |
| HyDE                 | sentence\_splitter  | False  | 10     | 0.78              | 0.95               | 0.72         | 0.93            | 0.8450          |
| HyDE                 | semantic            | False  | 5      | 0.80          | **0.97**           | 0.70         | 0.91            | 0.8450          |
| query\_rewriting     | semantic            | True   | 10     | 0.80               | 0.93               | 0.71         | 0.93            | 0.8425         |
| query\_rewriting     | semantic            | False  | 5      | 0.78              | 0.92               | 0.71         | 0.92            | 0.8325         |
| query\_rewriting     | sentence\_splitter  | False  | 5      | 0.79              | 0.95               | 0.70         | 0.89            | 0.8325         |
| query\_rewriting     | semantic            | False  | 10     | 0.79              | 0.91               | 0.68         | 0.94            | 0.8300           |
| query\_rewriting     | sentence\_splitter  | False  | 10     | 0.78              | 0.93               | 0.67         | 0.90            | 0.8200           |

### 🔍 Key Findings

The **best-performing configuration**, with an average score of **0.8625**, used the **original query**, **sentence-based chunking**, **reranking enabled**, and **top\_k = 5**. While three configurations achieved the same top score, we prefer this one because it is more efficient: unlike HyDE, it avoids additional generation steps that add computational overhead without improving performance, and compared to the no-rerank + top\_k = 10 setup, it uses fewer context chunks and benefits from reranking—both of which, as shown below, generally lead to better results.

However, it's important to note that these results are not 100% reliable. Both RAGAS and Llama 3.1 8B encountered difficulties parsing the metrics correctly for various data points across each configuration. As a result, the evaluation for each metric is not always based on the full set of 201 data points. Despite this limitation, we interpret the results as they are, given that a similar amount of null values were detected across each metric and configuration.

###  **Average Score: With vs Without Rerank**

| Rerank | Avg. Score |
| ------ | ---------- |
| True   | **0.8556** |
| False  | 0.8430     |
---

**Reranking** improved results across the board by helping filter the most relevant passages from a larger pool (15 → top\_k). Even though gains were small, they were consistent.

###  **Average Score by Top\_k**

| top\_k | Avg. Score |
| ------ | ---------- |
| 5      | **0.8508** |
| 10     | 0.8488     |
---
**Top\_k = 5** led to slightly better performance on average than **Top\_k = 10**, suggesting that fewer, higher-quality chunks are often more useful than retrieving a broader, noisier context window. This effect was most noticeable when reranking was applied.

### **Average Score by Retrieval Technique**

| Retrieval Technique | Avg. Score |
| ------------------- | ---------- |
| none                | **0.8569** |
| HyDE                | 0.8522     |
| query\_rewriting    | 0.8403     |
---

* The **original query** outperformed both HyDE and query rewriting, likely because the dataset’s questions are closely paraphrased from the texts — making them naturally well-aligned with their source documents.
* **HyDE**, while theoretically helpful, may have confused the retrieval system by generating generic or less grounded hypothetical texts. These often missed the subtle examples or details present in the original `text_i` columns, weakening retrieval precision.
* **Query Rewriting** underperformed and didn’t improve even with attempts to optimize the query for rewriting. While the exact reason is unclear, it may be that the original queries were already well-formed. Attempts to rewrite them often either changed their intent or added unnecessary information, degrading performance instead of helping.

### **Average Score by Chunking Technique**

| Chunking Technique | Avg. Score |
| ------------------ | ---------- |
| sentence\_splitter | **0.8505** |
| semantic           | 0.8492     |
---
**Chunking** based on sentence boundaries slightly outperformed semantic chunking. It likely maintained better context continuity, which helped the LLM remain grounded during generation.


## 🧪 Additional Tests

| model | retrieval_technique | chunking_technique | breakpoint_percentile_threshold | rerank | top_k | answer_relevancy | context_precision | faithfulness | context_recall | average_score |
|--------|----------------------|---------------------|-------------------------------|--------|--------|-------------------|--------------------|--------------|----------------|----------------|
| lenia | none                | semantic            | 85.0                         | True   | 5      | 0.80              | **0.96**           | 0.73         | **0.95**       | **0.860**      |
| lenia | none                | semantic            | 85.0                         | True   | 10     | 0.81              | 0.93               | **0.74**     | 0.94           | 0.855          |
| lenia | query_rewriting     | semantic            | 85.0                         | True   | 5      | 0.78              | 0.95               | 0.72         | 0.93           | 0.845          |
| lenia | query_rewriting     | semantic            | 85.0                         | True   | 10     | 0.82              | 0.92               | 0.69         | 0.93           | 0.840          |
| lenia | none                | semantic            | 80.0                         | True   | 5      | **0.83**          | 0.95               | 0.69         | 0.94           | 0.8525         |
| lenia | none                | semantic            | 80.0                         | True   | 10     | 0.80              | 0.94               | 0.71         | **0.95**       | 0.850          |
| qwen  | none                | sentence_splitter   | N/A                          | True   | 5      | 0.80              | **0.96**           | 0.61         | 0.94           | 0.8275         |
| qwen  | query_rewriting     | sentence_splitter   | N/A                          | True   | 5      | 0.78              | **0.96**           | 0.62         | **0.95**       | 0.8275         |


To further explore potential gains, we evaluated a range of additional configurations using both semantic chunking and a different model. For semantic chunking, we experimented with two different breakpoint thresholds (80 and 85), to identify the ideal breakpoint. While some runs showed **slightly improved metrics**, none surpassed the **average score of 0.8625** from our best-performing configuration using the **sentence splitter** and no retrieval enhancement.

We also attempted **query rewriting**, refining the prompts that generate the rewritten queries. However, these configurations consistently fell short of the performance achieved with the no retrieval enhancement.

Lastly, we tested a smaller model, [**Qwen-2.5-0.5B**](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), using the [**Q4\_K\_M GGUF quantized**](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) version. Since the speed and efficiency of RAG systems depend heavily on model size, we wanted to understand how much performance would be affected by switching to a smaller, less capable model. Even under our best RAG setup from earlier experiments, **Qwen’s performance aligned with the weakest results among the 24 previous combinations using LeNIA**. We see the metric that decreased significantly when using Qwen was faithfulness, which suggests that while Qwen is faster, it sacrifices some consistency in how well the generated response aligns with and supports the retrieved context compared to LeNIA. However, Qwen took only **3/4 of the time** to generate responses for the 201 queries as compared to LeNIA. Thus, it comes down to how much quality we are willing to lose for the increased time performance. 


## RAG API with Docker

This project provides a Retrieval-Augmented Generation (RAG) API using FastAPI, LlamaIndex, and Hugging Face models. The system consists of three containers:

1. **Text Embeddings Inference (TEI)**: Responsible for computing text embeddings.
2. **Text Generation Inference (TGI)**: Generates responses based on retrieved context.
3. **RAG API**: Manages indexing and query processing.

### Prerequisites

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

### Setup and Running the Containers

#### 1. Build the Containers

```sh
docker compose build
```

#### 2. Start the Containers

```sh
docker compose up -d
```

This will start the following services:

- `tei` on port `8080`
- `tgi` on port `8081`
- `rag` (the API) on port `8000`

#### 3. Stop the Containers

```sh
docker compose down
```

### API Endpoints

Once the RAG API is running, you can interact with it via HTTP requests.

#### **1. Upload Text Documents**

To create the vector database, send a list of texts to be indexed.

```sh
curl -X POST "http://localhost:8000/upload" -H "Content-Type: application/json" -d '{"texts": ["The capital of France is Paris.", "Python was created by Guido van Rossum."]}'
```

#### **2. Generate Responses**

Once documents are indexed, ask questions based on the stored knowledge.

```sh
curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{"new_message": {"role": "user", "content": "What is the capital of France?"}}'
```

### Testing the API with Python

Save the following script as `test_rag.py` and run it after starting the containers:

```python
import requests

def main():
    base_url = "http://localhost:8000"  # Adjust if needed

    # Upload sample texts
    texts = [
        "The capital of France is Paris. France is in Europe.",
        "Don Quixote was written by Miguel de Cervantes in the early 17th century.",
        "Python is a popular programming language created by Guido van Rossum."
    ]
    upload_payload = {"texts": texts}
    print("Uploading documents...\n")
    resp_upload = requests.post(f"{base_url}/upload", json=upload_payload)
    print("Status code /upload:", resp_upload.status_code)
    print("Response /upload:", resp_upload.json())

    # Example questions
    questions = [
        "What is the capital of France?",
        "Who created the Python language?",
        "Who wrote Don Quixote?"
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        generate_payload = {"new_message": {"role": "user", "content": q}}
        resp_generate = requests.post(f"{base_url}/generate", json=generate_payload)
        print("Status code /generate:", resp_generate.status_code)
        if resp_generate.ok:
            data = resp_generate.json()
            print("Generated response:", data.get("generated_text"))
        else:
            print("Error:", resp_generate.text)

if __name__ == "__main__":
    main()
```

### Summary

- **Build the images**: `docker compose build`
- **Run the containers**: `docker compose up -d`
- **Stop the containers**: `docker compose down`
- **Test the API**: Use `curl` commands or `test_rag.py`
