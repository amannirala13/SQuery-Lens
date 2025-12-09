#!/bin/bash
# Test Schema Ranker with full schema support
# Usage: 
#   ./test_schema_ranker.sh "query" "table1,table2,table3"
#   ./test_schema_ranker.sh "query" "table1|table2|table3"  (use | for full schemas)

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -z "$1" ]; then
    echo "Usage: ./test_schema_ranker.sh \"query\" \"table1,table2,table3\""
    echo ""
    echo "Examples:"
    echo "  # Simple table names:"
    echo "  ./test_schema_ranker.sh \"Find customer orders\" \"customers,orders,products,logs\""
    echo ""
    echo "  # Full schemas (use | separator):"
    echo "  ./test_schema_ranker.sh \"Find customer orders\" \"customers (id, name, email)|orders (id, customer_id, total)|products (id, name, price)\""
    exit 1
fi

source venv/bin/activate

QUERY="$1"
TABLES="$2"

# Detect separator
if [[ "$TABLES" == *"|"* ]]; then
    SEPARATOR="|"
else
    SEPARATOR=","
fi

python3 << EOF
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load model
model = SentenceTransformer('models/schema_ranker/schema_ranker_model')

query = """$QUERY"""
tables_str = """$TABLES"""
separator = """$SEPARATOR"""

tables = [t.strip() for t in tables_str.split(separator) if t.strip()]

# Rank
query_emb = model.encode(query)
table_embs = model.encode(tables)
scores = cos_sim(query_emb, table_embs)[0].tolist()

ranked = sorted(zip(tables, scores), key=lambda x: -x[1])

print(f'Query: "{query}"')
print()
print('Ranked schema elements:')
for name, score in ranked:
    # Truncate long names for display
    display_name = name[:40] + "..." if len(name) > 40 else name
    bar = '█' * max(0, int(score * 20))
    print(f'  {display_name:45s} {score:.3f} {bar}')
EOF
