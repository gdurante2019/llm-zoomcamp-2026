# LLM Zoomcamp 2026

This repo contains work for DataTalksClub's 2026 LLM Zoomcamp.  The course homepage is at https://github.com/DataTalksClub/llm-zoomcamp/tree/main.  

This is the third iteration of this program.  The course has undergone revisions since the first offering, but the foundational concepts of building LLM apps with RAG (Retrieval Augmented Generation), agents, and vector search remain the same.

The course takes us through initial setup of Agentic RAG all the way through to an end-to-end application, including orchestration, data ingestion, evaluation, and monitoring.  Students who complete capstone projects that meet course requirements receive a certificate at the end.

## Module 1:  Agentic RAG

Part 1 of this module introduces us to RAG and text search using minsearch (in-memory) and sqlitesearch (persistent RAG).  Part 2 of this module covers agentic RAG and agentic RAG with tool calling.  

Work is in 01-agentic-rag folder of this repo:  
https://github.com/gdurante2019/llm-zoomcamp-2026/tree/main/01-agentic-rag

## Module 2:  Vector Search

In this module, we advance from text search (which matches exact words in the query to the documents in the corpus) to vector search (which allows us to search by the meaning in the words, not just matching key words).  We explore a variety of libraries and approaches to vector search, including Alexey's minsearch library (which has both text search and vector search modules), sqlitesearch, pgvector, sentence-transformers, ONNX reader, and more.  

Module lessons are in the 02-vector-search folder at https://github.com/gdurante2019/llm-zoomcamp-2026/tree/main/02-vector-search.  The ONNX Runtime lesson and homework file (which uses ONNX Runtime and related utilities) is located at https://github.com/gdurante2019/llm-zoomcamp-2026/tree/main/llm-zoomcamp-onnx-homework-02. 