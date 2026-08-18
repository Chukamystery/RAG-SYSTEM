# Phase 4 — Retrieval-Augmented Generation (RAG)

## Project Overview

This project implements a basic Retrieval-Augmented Generation (RAG)
system that answers questions using a small company knowledge base.

The system combines semantic search with a local Large Language Model
to retrieve relevant information and generate grounded answers.

---

## Project Structure

```text
phase4_RAG/
│
├── data/
│   ├── customer_support.txt
│   ├── expense_policy.txt
│   ├── leave_policy.txt
│   ├── onboarding.txt
│   ├── pricing_policy.txt
│   ├── refund_policy.txt
│   ├── remote_work.txt
│   └── security_policy.txt
│
├── rag.py
├── ragbackup.py
├── pca_embeddings.png
└── README.md