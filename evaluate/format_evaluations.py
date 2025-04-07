import os
import re
import pandas as pd
import numpy as np

folder_path = "rag_evaluations"

# Updated regex to include chunking_technique
pattern = re.compile(
    r"(?P<retrieval_technique>none|HyDE|query_rewriting)_(?P<chunking_technique>semantic|sentence_splitter)_rerank(?P<rerank>True|False)_top_k(?P<top_k>\d+)\.csv"
)

evaluation = []

for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        retrieval_technique = match.group("retrieval_technique")
        chunking_technique = match.group("chunking_technique")
        rerank = match.group("rerank") == "True"
        top_k = int(match.group("top_k"))

        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        # Updated metrics
        answer_relevancy = round(df["answer_relevancy"].mean(skipna=True), 2)
        context_precision = round(df["context_precision"].mean(skipna=True), 2)
        faithfulness = round(df["faithfulness"].mean(skipna=True), 2)
        context_recall = round(df["context_recall"].mean(skipna=True), 2)

        # Null counts
        answer_relevancy_nulls = df["answer_relevancy"].isnull().sum()
        context_precision_nulls = df["context_precision"].isnull().sum()
        faithfulness_nulls = df["faithfulness"].isnull().sum()
        context_recall_nulls = df["context_recall"].isnull().sum()

        average_score = np.mean([answer_relevancy, context_precision, faithfulness, context_recall])

        evaluation.append({
            "retrieval_technique": retrieval_technique,
            "chunking_technique": chunking_technique,
            "rerank": rerank,
            "top_k": top_k,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "faithfulness": faithfulness,
            "context_recall": context_recall,
            "average_score": average_score,
            "answer_relevancy_nulls": answer_relevancy_nulls,
            "context_precision_nulls": context_precision_nulls,
            "faithfulness_nulls": faithfulness_nulls,
            "context_recall_nulls": context_recall_nulls,
        })

evaluation_df = pd.DataFrame(evaluation).sort_values(by="average_score", ascending=False)
evaluation_df.to_csv(os.path.join(folder_path, "rag_evaluation_summary.csv"), index=False)
