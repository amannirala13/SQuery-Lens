"""
Training Script for Schema Ranker
Uses Sentence-BERT with contrastive learning for table/column relevance
"""
import sys
import json
import time
from datetime import datetime
from pathlib import Path

from sentence_transformers import (
    SentenceTransformer, 
    losses,
    evaluation,
    InputExample
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from schema_ranker import config
from schema_ranker.data_processor import ColumnRankingDataset


def create_evaluator(examples, name="val"):
    """Create evaluator for validation"""
    sentences1 = [ex.texts[0] for ex in examples]
    sentences2 = [ex.texts[1] for ex in examples]
    scores = [ex.label for ex in examples]
    
    return evaluation.EmbeddingSimilarityEvaluator(
        sentences1, sentences2, scores,
        name=name,
        show_progress_bar=True
    )


def main():
    print(f"\n{'='*60}")
    print("Schema Ranker Training (Sentence-BERT)")
    print(f"{'='*60}")
    print(f"Device: {config.DEVICE}")
    print(f"Base model: {config.BASE_MODEL}")
    print(f"Time: {datetime.now()}")
    
    # Load data
    print("\n[1/4] Loading data...")
    dataset = ColumnRankingDataset()
    dataset.load()
    
    print(f"\nTrain examples: {len(dataset.train_examples)}")
    print(f"Val examples: {len(dataset.val_examples)}")
    
    # Load model
    print("\n[2/4] Loading base model...")
    model = SentenceTransformer(config.BASE_MODEL)
    print(f"Loaded {config.BASE_MODEL}")
    
    # Create dataloader
    print("\n[3/4] Preparing training...")
    train_dataloader = DataLoader(
        dataset.train_examples,
        shuffle=True,
        batch_size=config.BATCH_SIZE
    )
    
    # Loss function - Cosine Similarity Loss for regression on similarity
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Evaluator
    evaluator = create_evaluator(dataset.val_examples[:5000])  # Subset for speed
    
    # Calculate steps
    total_steps = len(train_dataloader) * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    # Train
    print("\n[4/4] Training...")
    start_time = time.time()
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=config.NUM_EPOCHS,
        evaluation_steps=config.EVAL_STEPS,
        warmup_steps=warmup_steps,
        output_path=str(config.BEST_MODEL_PATH),
        save_best_model=True,
        show_progress_bar=True
    )
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Time: {training_time/60:.1f} minutes")
    print(f"Model saved to: {config.BEST_MODEL_PATH}")
    
    # Test the model
    print("\n[Test] Running quick inference test...")
    model = SentenceTransformer(str(config.BEST_MODEL_PATH))
    
    test_queries = [
        "What is the total sales by region?",
        "Find customers who ordered products",
        "Delete expired user sessions"
    ]
    
    test_schemas = ["customers", "orders", "sales", "products", "sessions", "logs", "config"]
    
    print("\n=== Schema Ranking Results ===")
    for query in test_queries:
        query_emb = model.encode(query)
        schema_embs = model.encode(test_schemas)
        
        # Cosine similarity
        from sentence_transformers.util import cos_sim
        scores = cos_sim(query_emb, schema_embs)[0].tolist()
        
        ranked = sorted(zip(test_schemas, scores), key=lambda x: -x[1])
        
        print(f"\nQuery: \"{query}\"")
        print("Top 3 tables/columns:")
        for name, score in ranked[:3]:
            print(f"  {name}: {score:.3f}")


if __name__ == "__main__":
    main()
