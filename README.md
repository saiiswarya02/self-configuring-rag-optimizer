# Self-Configuring RAG Hyperparameter Optimizer

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that automatically optimizes retrieval parameters using Optuna.

The system:

* Accepts PDF documents through a Streamlit interface
* Creates embeddings using Hugging Face models
* Stores vectors in FAISS
* Retrieves relevant context
* Generates answers using Ollama (Llama 3.2)
* Automatically finds the best RAG configuration

## Features

* PDF Upload
* Automatic Chunk Size Optimization
* Chunk Overlap Optimization
* Top-K Retrieval Optimization
* FAISS Vector Database
* Hugging Face Embeddings
* Ollama Llama 3.2 Integration
* Streamlit User Interface

## Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* Hugging Face
* Optuna
* Ollama
* Llama 3.2

## Project Architecture

PDF Upload
→ Text Extraction
→ Chunking
→ Embeddings
→ FAISS Vector Store
→ Retrieval
→ Ollama LLM
→ Answer Generation

## Installation

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

## Ollama Model

```bash
ollama run llama3.2:3b
```

## Future Improvements

* RAGAS Evaluation
* Semantic Similarity Scoring
* Multiple Embedding Models
* Performance Dashboard
* Advanced Hyperparameter Optimization
