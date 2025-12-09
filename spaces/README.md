---
title: SQuery-Lens
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# SQuery-Lens

SQL Query Analyzer & Schema Ranker for text-to-SQL systems.

## Features

- **Query Analyzer**: Classify query complexity, category, keywords
- **Schema Ranker**: Rank tables by relevance to natural language queries

## Models

- Query Analyzer: DistilBERT (83.2% accuracy)
- Schema Ranker: Sentence-BERT (80% recall @ top-15)

## Links

- [GitHub](https://github.com/amannirala13/SQuery-Lens)
- [Models](https://huggingface.co/amannirala13/squery-lens-models)
- [Datasets](https://huggingface.co/datasets/amannirala13/squery-lens-data)
