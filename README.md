# Multi-Document AI Chatbot 📚🤖

A Retrieval-Augmented Generation (RAG) chatbot that allows users to chat with multiple PDF documents using Generative AI.

## Features

* Multi-PDF Question Answering
* Semantic Search using Vector Embeddings
* ChromaDB Vector Database
* Conversational Memory
* Groq + Llama Integration
* Streamlit Interface

## Tech Stack

* Python
* Streamlit
* LangChain
* ChromaDB
* HuggingFace Embeddings
* Groq API
* Llama 3.3

## Architecture

PDFs → Text Chunking → Embeddings → ChromaDB → Retriever → LLM → Answer

## Setup

1. Clone the repository
2. Create a virtual environment
3. Install dependencies:
   pip install -r requirements.txt
4. Add your Groq API key in config.json
5. Run:
   python vectorize_documents.py
   streamlit run main.py

## Author

Sweedel Rodrigues
