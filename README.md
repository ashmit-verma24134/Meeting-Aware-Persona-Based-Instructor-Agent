---

## Meeting-Aware-Persona-Based-Instructor-Agent

## Overview

This project implements a **meeting-aware, persona-based instructor agent** that answers **specific questions** from meeting transcripts using a **Retrieval-Augmented Generation (RAG)** pipeline.

The system does **not summarize meetings**. Instead, it retrieves **relevant transcript chunks** using vector similarity search and generates **grounded answers** strictly based on retrieved content.

The agent is designed for **academic use**, focusing on clarity, correctness, and controlled information access.

---

## Problem Statement

Students often miss details discussed in meetings or lectures and need a way to ask **precise follow-up questions** without rewatching or rereading entire transcripts.

This project solves that problem by:

* Embedding meeting transcripts
* Matching user queries against transcript chunks
* Generating answers grounded only in relevant transcript context

---

## System Architecture

### High-Level Flow

1. Meeting transcripts are chunked into smaller segments
2. Each chunk is converted into a vector embedding
3. Embeddings are stored in a vector index
4. User query is embedded using the same model
5. Vector similarity search retrieves relevant chunks
6. Retrieved chunks are passed to the language model
7. A final answer is generated using retrieved context

---

## Core Components

### Transcript Chunking

* Transcripts are pre-generated externally
* Text is split into semantically meaningful chunks
* Chunking improves retrieval accuracy

### Embedding Generation

* Each transcript chunk is converted into a dense vector
* User queries are embedded in the same vector space
* Ensures consistent similarity comparison

### Vector Store

* Embeddings are stored in a FAISS vector index
* Enables fast nearest-neighbor search
* Returns top-k relevant transcript chunks

### Retrieval-Augmented Generation

* Retrieved chunks are injected as context
* Language model generates an answer
* Model is constrained to transcript content only

---

## Project Structure

```
.
├── README.md
├── index.html
├── chunk_embeddings.json
├── vector.index
└── test/
```

### File Descriptions

* `README.md`
  Project documentation

* `index.html`
  Simple frontend interface for interacting with the agent

* `chunk_embeddings.json`
  Stores transcript chunks and their embeddings

* `vector.index`
  FAISS vector index for similarity search

* `test/`
  Test files and experiments

---

## Key Design Principles

### Transcript-Grounded Answers

The agent cannot answer questions outside the transcript content.
If information is missing, it explicitly states so.

### Isolation and Safety

* Only provided transcripts are queried
* No external knowledge is injected
* Prevents hallucination

### Modular Pipeline

Each stage (chunking, embedding, retrieval, generation) is independent and replaceable.

---

## Technology Stack

* Python
* FAISS for vector similarity search
* Sentence Transformers / OpenAI Embeddings
* HTML for basic UI
* LLM for answer generation

---

## Current Limitations

* English-only transcripts
* No live meeting ingestion
* No speech-to-text pipeline inside the system
* Single-user local setup

---

## Future Scope

### Agent-Based Extensions

* Query understanding agent
* Persona-aware response generation
* Multi-agent orchestration using LangGraph

### Feature Enhancements

* User authentication
* Multi-meeting support
* Multilingual transcripts
* Web and Slack integration

---

## Academic Relevance

This project demonstrates:

* Information Retrieval concepts
* Vector databases and embeddings
* Retrieval-Augmented Generation
* Safe and grounded LLM usage
* Agent-ready system design

Suitable for coursework in **AI, NLP, ML, and Agent-Based Systems**.

---

## Author

Ashmit Verma
B.Tech Student
IIIT Delhi

---


