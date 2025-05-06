## Evaluate Folder Structure

- **`rag_predictions/`**  
  Contains all generated predictions for each RAG configuration, evaluated on the [**RagQuAS** dataset](https://huggingface.co/datasets/IIC/RagQuAS).

- **`rag_evaluations/`**  
  Stores evaluation outputs:
  - Individual RAGAS metric results for each configuration.
  - `rag_evaluation_summary.csv`: Summarizes average RAGAS metrics across all configurations.

- **`requirements.txt`**
    Contains all necessary dependencies to run scripts.

- **Python Scripts**
  - **`llm_api.py`**  
    Hosts an LLM API used to make predictions. An example of how to run the [**LeNIA-1.5B**](https://huggingface.co/LenguajeNaturalAI/leniachat-qwen2-1.5B-v0) with [**4-bit quantization (GGUF 4\_K\_M)**](https://huggingface.co/QuantFactory/leniachat-qwen2-1.5B-v0-GGUF) can be seen below:
    ```bash
    python3 llm_api.py --model_name=QuantFactory/leniachat-qwen2-1.5B-v0-GGUF --tokenizer=LenguajeNaturalAI/leniachat-qwen2-1.5B-v0 --filename=leniachat-qwen2-1.5B-v0.Q4_K_M.gguf
    ```
  
  - **`rag.py`**  
    Implements the `RAG` class supporting different configurations:
    - Reranking
    - HyDE (Hypothetical Document Embeddings)
    - Query Rewriting
    - Chunking (Semantic or Sentence Splitter)
    - Uploads data to vector database from HuggingFace datasets.

  - **`generate_responses.py`**  
    Automates the generation of predictions:
    - Creates a grid of RAG configurations.
    - Calls API from `llm_api.py` for predictions and uses RAG class in **`rag.py`**.
    - Saves results as JSON files in `rag_predictions/`.

  - **`evaluate_ragas.py`**  
    Evaluates each prediction set using the [RAGAS](https://docs.ragas.io/en/latest/) framework, storing detailed metrics in `rag_evaluations/`. This script uses NovitaAI which provides cheap and fast inference for the [**Llama 3.1 8B model**](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

  - **`format_evaluations.py`**  
    Processes individual evaluations to calculate average RAGAS scores per configuration for easier comparison and returns results as `rag_evaluation_summary.csv`.


