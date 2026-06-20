# Self Configuring RAG Optimizer

## Overview

A Retrieval-Augmented Generation (RAG) system that automatically optimizes:

- Chunk Size
- Chunk Overlap
- Top-K Retrieval

using Optuna Hyperparameter Optimization.

## Features

- PDF Upload
- FAISS Vector Database
- HuggingFace Embeddings
- Ollama (Llama 3.2)
- Streamlit Interface
- Automatic RAG Optimization

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Optuna
- Ollama
- HuggingFace

## Run

```bash
streamlit run app.py
```

## Model

```bash
ollama run llama3.2:3b
```