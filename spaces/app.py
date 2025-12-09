"""
SQuery-Lens: SQL Query Analyzer & Schema Ranker
Interactive demo on Hugging Face Spaces
"""

import gradio as gr
import torch
from huggingface_hub import hf_hub_download, snapshot_download
import os
import sys

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
from src.api.inference import QueryAnalyzer
from src.schema_ranker import SchemaRanker

# Initialize models
print("Loading Query Analyzer...")
query_analyzer = QueryAnalyzer()

print("Loading Schema Ranker...")
schema_ranker = SchemaRanker()

def analyze_query(query: str) -> dict:
    """Analyze a natural language query for SQL generation."""
    if not query.strip():
        return {"error": "Please enter a query"}
    
    result = query_analyzer.analyze(query)
    
    return {
        "🎯 Complexity": f"{result['complexity']['label']} ({result['complexity']['confidence']:.1%})",
        "📂 Category": f"{result['category']['label']} ({result['category']['confidence']:.1%})",
        "🏷️ Subcategories": ", ".join(result.get('subcategories', {}).get('labels', [])),
        "🔑 Keywords": ", ".join(result.get('keywords', {}).get('labels', [])),
        "📊 Estimated Tables": str(result.get('table_count', {}).get('label', 'N/A'))
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
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
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
