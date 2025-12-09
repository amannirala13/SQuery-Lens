"""
SQuery-Lens: SQL Query Analyzer & Schema Ranker
Interactive demo on Hugging Face Spaces
"""

import gradio as gr
import torch
from huggingface_hub import snapshot_download
import os
import sys
import pickle

# Download models from HF Hub on startup
MODEL_DIR = "./models"
if not os.path.exists(MODEL_DIR):
    print("Downloading models from Hugging Face Hub...")
    snapshot_download(
        repo_id="amannirala13/squery-lens-models",
        local_dir=MODEL_DIR,
        repo_type="model"
    )

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after downloading models
from transformers import DistilBertTokenizer
from src.model.model import EnhancedSQLAnalyzer
from src.schema_ranker.ranker import SchemaRanker

# Device configuration
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# Load Query Analyzer
print("Loading Query Analyzer...")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model_enhanced.pt")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders_enhanced.pkl")

# Load encoders
with open(ENCODERS_PATH, 'rb') as f:
    encoders = pickle.load(f)

# Initialize model
num_complexity = len(encoders['complexity'].classes_)
num_keywords = len(encoders['keywords'].classes_)
num_category = len(encoders['category'].classes_)
num_subcategory = len(encoders['subcategory'].classes_)
num_table_count = len(encoders['table_count'].classes_)

model = EnhancedSQLAnalyzer(
    num_complexity_classes=num_complexity,
    num_keywords=num_keywords,
    num_categories=num_category,
    num_subcategories=num_subcategory,
    num_table_counts=num_table_count
)

# Load weights
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

print("Loading Schema Ranker...")
schema_ranker = SchemaRanker(
    model_path=os.path.join(MODEL_DIR, "schema_ranker/schema_ranker_model")
)

def analyze_query(query: str) -> dict:
    """Analyze a natural language query for SQL generation."""
    if not query.strip():
        return {"error": "Please enter a query"}
    
    # Tokenize
    encoding = tokenizer(
        query,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        
    complexity_logits, keywords_logits, category_logits, subcategory_logits, table_count_logits = outputs
    
    # Process complexity
    complexity_probs = torch.softmax(complexity_logits, dim=1)
    complexity_idx = torch.argmax(complexity_probs, dim=1).item()
    complexity_label = encoders['complexity'].inverse_transform([complexity_idx])[0]
    
    # Process category
    category_probs = torch.softmax(category_logits, dim=1)
    category_idx = torch.argmax(category_probs, dim=1).item()
    category_label = encoders['category'].inverse_transform([category_idx])[0]
    
    # Process keywords (multi-label, threshold=0.5)
    keywords_probs = torch.sigmoid(keywords_logits)
    keywords_indices = (keywords_probs[0] > 0.5).nonzero(as_tuple=True)[0].cpu().numpy()
    keywords_labels = encoders['keywords'].inverse_transform([[1 if i in keywords_indices else 0 for i in range(len(encoders['keywords'].classes_))]])[0]
    keywords_labels = [k for k, v in zip(encoders['keywords'].classes_, keywords_labels) if v == 1]
    
    # Process subcategories (multi-label, threshold=0.5)
    subcategory_probs = torch.sigmoid(subcategory_logits)
    subcategory_indices = (subcategory_probs[0] > 0.5).nonzero(as_tuple=True)[0].cpu().numpy()
    subcategory_labels = encoders['subcategory'].inverse_transform([[1 if i in subcategory_indices else 0 for i in range(len(encoders['subcategory'].classes_))]])[0]
    subcategory_labels = [k for k, v in zip(encoders['subcategory'].classes_, subcategory_labels) if v == 1]
    
    # Process table count
    table_count_probs = torch.softmax(table_count_logits, dim=1)
    table_count_idx = torch.argmax(table_count_probs, dim=1).item()
    table_count_label = encoders['table_count'].inverse_transform([table_count_idx])[0]
    
    return {
        "🎯 Complexity": f"{complexity_label} ({complexity_probs[0, complexity_idx]:.1%})",
        "📂 Category": f"{category_label} ({category_probs[0, category_idx]:.1%})",
        "🏷️ Subcategories": ", ".join(subcategory_labels) if subcategory_labels else "None",
        "🔑 Keywords": ", ".join(keywords_labels) if keywords_labels else "None",
        "📊 Estimated Tables": table_count_label
    }

def rank_tables(query: str, tables: str) -> str:
    """Rank tables by relevance to a query."""
    if not query.strip():
        return "Please enter a query"
    if not tables.strip():
        return "Please enter table names (comma-separated)"
    
    # Parse tables
    table_list = [t.strip() for t in tables.split(",") if t.strip()]
    
    if not table_list:
        return "No valid tables provided"
    
    # Load schema (simple table names)
    schema_ranker.load_schema(table_list)
    
    # Get rankings
    hints = schema_ranker.get_hints(query, top_k=min(10, len(table_list)))
    
    # Format output
    lines = ["### 📊 Table Rankings\n"]
    lines.append("| Rank | Table | Relevance Score |")
    lines.append("|------|-------|-----------------|")
    
    for i, (table, score) in enumerate(zip(hints.table_names, hints.scores), 1):
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        lines.append(f"| {i} | `{table}` | {bar} {score:.2f} |")
    
    return "\n".join(lines)

def combined_analysis(query: str, tables: str) -> tuple:
    """Run both analysis and ranking."""
    analysis = analyze_query(query)
    ranking = rank_tables(query, tables) if tables.strip() else "No tables provided for ranking"
    return analysis, ranking

# Build Gradio Interface
with gr.Blocks(
    title="SQuery-Lens",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
    css="""
    .gradio-container { max-width: 1200px !important; }
    .header { text-align: center; margin-bottom: 2rem; }
    """
) as demo:
    
    gr.Markdown("""
    # 🔍 SQuery-Lens
    ### SQL Query Analyzer & Schema Ranker
    
    Analyze natural language queries for text-to-SQL systems. Get complexity classification, 
    keyword extraction, and table relevance ranking.
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="📝 Natural Language Query",
                placeholder="e.g., Find all customers who spent more than $1000 last month",
                lines=3
            )
            
            tables_input = gr.Textbox(
                label="🗃️ Available Tables (comma-separated)",
                placeholder="e.g., customers, orders, products, payments, logs",
                lines=2
            )
            
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze Query", variant="primary")
                clear_btn = gr.Button("🗑️ Clear")
        
        with gr.Column(scale=1):
            analysis_output = gr.JSON(label="📊 Query Analysis")
            ranking_output = gr.Markdown(label="📈 Table Rankings")
    
    gr.Markdown("""
    ---
    ### 💡 Example Queries
    """)
    
    gr.Examples(
        examples=[
            ["Find customers who ordered products worth more than $500", "customers, orders, products, payments, shipping"],
            ["Show monthly revenue by product category", "products, categories, sales, inventory, suppliers"],
            ["List all employees in the engineering department", "employees, departments, salaries, projects, teams"],
            ["Calculate average order value per customer segment", "customers, orders, segments, transactions, returns"],
        ],
        inputs=[query_input, tables_input],
    )
    
    gr.Markdown("""
    ---
    ### 📚 About
    
    - **Query Analyzer**: DistilBERT-based classifier (83.2% accuracy)
    - **Schema Ranker**: Sentence-BERT bi-encoder (80% recall @ top-15)
    
    [GitHub](https://github.com/amannirala13/SQuery-Lens) | 
    [Models](https://huggingface.co/amannirala13/squery-lens-models) | 
    [Datasets](https://huggingface.co/datasets/amannirala13/squery-lens-data)
    """)
    
    # Event handlers
    analyze_btn.click(
        fn=combined_analysis,
        inputs=[query_input, tables_input],
        outputs=[analysis_output, ranking_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", {}, ""),
        inputs=[],
        outputs=[query_input, tables_input, analysis_output, ranking_output]
    )

if __name__ == "__main__":
    demo.launch()

