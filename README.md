# SQL Query Analyzer

A multi-model system for analyzing natural language queries for SQL generation. Provides query classification, complexity analysis, and **schema relevance ranking** for text-to-SQL systems.

## ✨ Features

| Feature | Model | Accuracy | Use Case |
|---------|-------|----------|----------|
| **Query Analyzer** | DistilBERT | 83.2% | Classify query complexity, type, keywords |
| **Schema Ranker** | Sentence-BERT | 80% recall | Pre-filter tables for RAG systems |

---

## 🚀 Quick Start

```bash
# 1. Setup
./setup.sh

# 2. Download pre-trained models (recommended)
huggingface-cli download amannirala13/squery-lens-models --local-dir ./models

# Or train from scratch (~1 hour total)
# ./train.sh           # Query Analyzer (~16 min)
# ./train_ranker.sh    # Schema Ranker (~46 min)

# 3. Test
./analyze.sh "Find customers who spent more than 1000"
./rank.sh "Show all orders" "customers,orders,products,logs"

# 4. Demo
python demo_ranker.py
```

### 📦 Models & Data (Hugging Face Hub)

| Resource | Link | Size |
|----------|------|------|
| **Pre-trained Models** | [amannirala13/squery-lens-models](https://huggingface.co/amannirala13/squery-lens-models) | 1.6 GB |
| **Training Datasets** | [amannirala13/squery-lens-data](https://huggingface.co/datasets/amannirala13/squery-lens-data) | 262 MB |

```bash
# Download models only
huggingface-cli download amannirala13/squery-lens-models --local-dir ./models

# Download datasets (for training from scratch)
huggingface-cli download amannirala13/squery-lens-data --local-dir ./data --repo-type dataset
```

---

## 📁 Project Structure

```
sql_classifier/
├── src/
│   ├── model/                    # Query Analyzer (DistilBERT)
│   ├── schema_ranker/            # Schema Ranker (Sentence-BERT)
│   │   ├── ranker.py             # Python API for RAG integration
│   │   └── train.py
│   ├── training/
│   └── api/
│
├── models/
│   ├── best_model_enhanced.pt    # Query Analyzer
│   └── schema_ranker/            # Schema Ranker
│
├── # Scripts
├── setup.sh          # Install dependencies
├── train.sh          # Train Query Analyzer
├── train_ranker.sh   # Train Schema Ranker
├── serve.sh          # Start API server
├── analyze.sh        # Analyze a query
├── rank.sh           # Rank tables
└── demo_ranker.py    # Interactive demo
```

---

## 📊 Query Analyzer

Classifies natural language queries to help route and optimize SQL generation.

### Outputs
- **Complexity**: simple, medium, complex
- **Category**: Data Manipulation, Schema Definition, etc.
- **Subcategories**: Read, Write, Aggregation, Filtering, Joining
- **Keywords**: SELECT, JOIN, WHERE, GROUP BY, etc.
- **Estimated Tables**: 1, 2, 3+

### Usage
```bash
$ ./analyze.sh "Calculate total revenue by product category"

Complexity:     medium (92.8%)
Category:       Data Manipulation (99.7%)
Subcategories:  ['Read', 'Aggregation', 'Grouping']
Keywords:       ['SELECT', 'FROM', 'GROUP BY', 'AS']
Est. Tables:    1
```

---

## 🎯 Schema Ranker

Ranks tables/columns by relevance to a natural language query.

### When to Use

| Scenario | Schema Ranker Needed? |
|----------|----------------------|
| RAG uses general embeddings (OpenAI, Cohere) | ✅ Yes - domain-specific ranking |
| RAG already has SQL-tuned embeddings | ❌ Redundant |
| 100+ tables, need fast pre-filtering | ✅ Yes - 7ms for 500 tables |
| Schema already in RAG vector DB | ❌ Query RAG directly |

### CLI Usage
```bash
# Simple table names
./rank.sh "Find customer orders" "customers,orders,products,logs"

# Full schemas (use | separator)
./rank.sh "Find customer orders" \
  "customers (id, name, email)|orders (id, customer_id, total)|products (id, name, price)"
```

### Python API for RAG Integration

```python
from src.schema_ranker import SchemaRanker

# Initialize (load once)
ranker = SchemaRanker()

# Load your schema
ranker.load_schema_with_columns({
    "customers": ["id", "name", "email"],
    "orders": ["id", "customer_id", "total"],
    "products": ["id", "name", "price"],
    ...
})

# Get hints for RAG
hints = ranker.get_hints("Find customers who ordered products", top_k=20)

# Use with RAG
rag.search(query, filter_tables=hints.table_names)      # Filter
rag.search(query, boost=hints.to_boost_weights())       # Boost
```

### Performance

| Metric | Value |
|--------|-------|
| Table encoding | 0.6s for 500 tables (one-time) |
| Query ranking | **7ms per query** |
| Recall @ top-15 | 80% |
| Recall @ top-30 | 80% |

### Limitations

The Schema Ranker does **text similarity matching** only. It does NOT understand:
- Database relationships (foreign keys)
- Business logic (payment = usage × rate)
- JOIN patterns

For complex queries requiring relationship reasoning, use the ranker as a **first-pass filter**, then let LLM reason over the candidates.

---

## 📡 API Server

```bash
./serve.sh
# Server runs at http://localhost:8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze` | Analyze query + rank tables |
| POST | `/analyze/batch` | Batch analysis |
| GET | `/info` | Model info |

### Example
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Find customers", "tables": ["customers", "orders"]}'
```

---

## 💻 TypeScript Client

```typescript
import { SQLQueryAnalyzer } from './client/classifier';

const analyzer = new SQLQueryAnalyzer('http://localhost:8000');
const result = await analyzer.analyze("Find customers", ["customers", "orders"]);

console.log(result.complexity);  // "simple"
console.log(result.tables);      // [{table: "customers", confidence: 0.92}, ...]
```

---

## 🏗️ Architecture

### Query Analyzer
```
Query → DistilBERT → Shared Layer (768→512)
                          ↓
        ┌────────┬────────┼────────┬────────┐
        ↓        ↓        ↓        ↓        ↓
   Complexity Keywords Category Subcat  TableCount
```

### Schema Ranker (Bi-Encoder)
```
Query → Encoder → Query Embedding
                        ↓
                  Cosine Similarity → Scores
                        ↑
Tables → Encoder → Table Embeddings (pre-computed)
```

### Recommended RAG Pipeline
```
User Query
     ↓
┌─────────────────────────────────────┐
│ Query Analyzer                      │
│ → complexity, keywords, category    │
│ → Route: simple→template, complex→LLM
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│ Schema Ranker (optional)            │
│ → Pre-filter 500 tables → 20        │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│ RAG Vector Search                   │
│ → Get detailed schema for top tables│
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│ LLM                                 │
│ → Generate SQL with focused context │
└─────────────────────────────────────┘
```

---

## 📈 Training Results

### Query Analyzer
| Output | Accuracy |
|--------|----------|
| Complexity | 74.9% |
| Keywords | 73.8% F1 |
| Category | 98.7% |
| Subcategory | 88.0% F1 |
| Table Count | 80.8% |
| **Combined** | **83.2%** |

### Schema Ranker
| Metric | Value |
|--------|-------|
| Pearson Correlation | 0.872 |
| Spearman Correlation | 0.805 |
| Training Samples | 350K pairs |
| Training Time | 46 min |

---

## 🔧 Scripts

| Script | Description |
|--------|-------------|
| `./setup.sh` | Install dependencies |
| `./train.sh` | Train Query Analyzer |
| `./train_ranker.sh` | Train Schema Ranker |
| `./serve.sh` | Start API server |
| `./analyze.sh "query"` | Analyze a query |
| `./rank.sh "query" "tables"` | Rank tables |
| `python demo_ranker.py` | Interactive demo |

---

## 📚 Datasets

| Dataset | Samples | Use |
|---------|---------|-----|
| text-to-sql-100K | 100K | Query Analyzer training |
| data_table_2 | 2.8M | Schema Ranker training |

---

## ⚙️ Hardware

Automatically uses:
- **MPS** (Metal) on Apple Silicon
- **CUDA** on NVIDIA GPUs
- **CPU** as fallback
