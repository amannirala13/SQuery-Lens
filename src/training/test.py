"""
Test Script for SQL Category Classifier
Evaluates the trained model on the test set and provides inference examples
"""
import sys
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
from transformers import DistilBertTokenizer

from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import config
from model.data_processor import DataProcessor
from model.model import load_model


class Tester:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = config.DEVICE
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def predict_batch(self, dataloader):
        all_cat_preds, all_cat_labels = [], []
        all_subcat_preds, all_subcat_labels = [], []
        
        for batch in tqdm(dataloader, desc="Testing"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            cat_labels = batch['category_label'].to(self.device)
            subcat_labels = batch['subcategory_label'].to(self.device)
            
            cat_logits, subcat_logits = self.model(input_ids, attention_mask)
            cat_preds = torch.argmax(cat_logits, dim=1)
            subcat_preds = torch.argmax(subcat_logits, dim=1)
            
            all_cat_preds.extend(cat_preds.cpu().numpy())
            all_cat_labels.extend(cat_labels.cpu().numpy())
            all_subcat_preds.extend(subcat_preds.cpu().numpy())
            all_subcat_labels.extend(subcat_labels.cpu().numpy())
        
        return np.array(all_cat_preds), np.array(all_cat_labels), np.array(all_subcat_preds), np.array(all_subcat_labels)
    
    @torch.no_grad()
    def predict_single(self, text: str):
        """Predict category and subcategory for a single text input"""
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=config.MAX_LENGTH,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        cat_logits, subcat_logits = self.model(input_ids, attention_mask)
        cat_probs = torch.softmax(cat_logits, dim=1)
        subcat_probs = torch.softmax(subcat_logits, dim=1)
        
        cat_pred = torch.argmax(cat_logits, dim=1).item()
        subcat_pred = torch.argmax(subcat_logits, dim=1).item()
        
        category = self.processor.category_encoder.inverse_transform([cat_pred])[0]
        subcategory = self.processor.subcategory_encoder.inverse_transform([subcat_pred])[0]
        
        return {
            'category': category,
            'subcategory': subcategory,
            'category_confidence': cat_probs[0, cat_pred].item(),
            'subcategory_confidence': subcat_probs[0, subcat_pred].item()
        }
    
    def test(self, test_loader):
        print("\nRunning test evaluation...")
        cat_preds, cat_labels, subcat_preds, subcat_labels = self.predict_batch(test_loader)
        
        cat_names = list(self.processor.category_encoder.classes_)
        subcat_names = list(self.processor.subcategory_encoder.classes_)
        
        # Category metrics
        cat_acc = accuracy_score(cat_labels, cat_preds)
        cat_prec, cat_rec, cat_f1, _ = precision_recall_fscore_support(cat_labels, cat_preds, average='weighted', zero_division=0)
        
        # Subcategory metrics
        subcat_acc = accuracy_score(subcat_labels, subcat_preds)
        subcat_prec, subcat_rec, subcat_f1, _ = precision_recall_fscore_support(subcat_labels, subcat_preds, average='weighted', zero_division=0)
        
        # Exact match
        exact_match = np.mean((cat_preds == cat_labels) & (subcat_preds == subcat_labels))
        
        print(f"\n{'='*60}")
        print("TEST SET RESULTS")
        print(f"{'='*60}")
        print(f"\nCategory Classification:")
        print(f"  Accuracy:  {cat_acc:.4f} ({cat_acc*100:.2f}%)")
        print(f"  Precision: {cat_prec:.4f}")
        print(f"  Recall:    {cat_rec:.4f}")
        print(f"  F1 Score:  {cat_f1:.4f}")
        
        print(f"\nSubcategory Classification:")
        print(f"  Accuracy:  {subcat_acc:.4f} ({subcat_acc*100:.2f}%)")
        print(f"  Precision: {subcat_prec:.4f}")
        print(f"  Recall:    {subcat_rec:.4f}")
        print(f"  F1 Score:  {subcat_f1:.4f}")
        
        print(f"\nCombined Metrics:")
        print(f"  Exact Match Rate: {exact_match:.4f} ({exact_match*100:.2f}%)")
        print(f"  Combined Accuracy: {(cat_acc + subcat_acc)/2:.4f}")
        
        # Detailed classification reports
        print(f"\n{'='*60}")
        print("DETAILED CATEGORY REPORT")
        unique_cat = np.unique(np.concatenate([cat_labels, cat_preds]))
        cat_names_filtered = [cat_names[i] for i in unique_cat if i < len(cat_names)]
        print(classification_report(cat_labels, cat_preds, labels=unique_cat, target_names=cat_names_filtered, zero_division=0))
        
        print(f"\n{'='*60}")
        print("DETAILED SUBCATEGORY REPORT")
        unique_subcat = np.unique(np.concatenate([subcat_labels, subcat_preds]))
        subcat_names_filtered = [subcat_names[i] for i in unique_subcat if i < len(subcat_names)]
        print(classification_report(subcat_labels, subcat_preds, labels=unique_subcat, target_names=subcat_names_filtered, zero_division=0))
        
        return {
            'category': {'accuracy': cat_acc, 'precision': cat_prec, 'recall': cat_rec, 'f1': cat_f1},
            'subcategory': {'accuracy': subcat_acc, 'precision': subcat_prec, 'recall': subcat_rec, 'f1': subcat_f1},
            'exact_match': exact_match
        }


def demo_inference(tester):
    """Demonstrate inference on sample inputs"""
    print(f"\n{'='*60}")
    print("DEMO INFERENCE")
    print(f"{'='*60}")
    
    test_queries = [
        "Find all customers who placed orders last month",
        "Create a new table for storing user preferences",
        "Calculate the average salary grouped by department",
        "Delete all expired sessions from the database",
        "Grant read access to the analytics team",
        "Show me the top 10 products by revenue using window functions",
        "Update the inventory count for low stock items",
        "Find employees whose salary is above the department average"
    ]
    
    for query in test_queries:
        result = tester.predict_single(query)
        print(f"\nQuery: {query}")
        print(f"  → Category: {result['category']} ({result['category_confidence']*100:.1f}%)")
        print(f"  → Subcategory: {result['subcategory']} ({result['subcategory_confidence']*100:.1f}%)")


def main():
    print(f"\n{'='*60}")
    print("SQL Category Classifier - Test Evaluation")
    print(f"{'='*60}")
    print(f"Device: {config.DEVICE}")
    
    if not config.BEST_MODEL_PATH.exists():
        print(f"\nError: No trained model found at {config.BEST_MODEL_PATH}")
        print("Please run train.py first to train the model.")
        sys.exit(1)
    
    # Load data
    print("\n[1/3] Loading data...")
    processor = DataProcessor()
    df = processor.load_and_process_data()
    cat_labels, subcat_labels = processor.prepare_labels(df)
    split_data = processor.split_data(df, cat_labels, subcat_labels)
    dataloaders = processor.create_dataloaders(split_data)
    
    # Load model
    print("\n[2/3] Loading model...")
    num_cat, num_subcat = processor.get_num_classes()
    model = load_model(num_cat, num_subcat)
    
    # Test
    print("\n[3/3] Running test evaluation...")
    tester = Tester(model, processor)
    results = tester.test(dataloaders['test'])
    
    # Demo inference
    demo_inference(tester)
    
    # Save results
    results_path = config.LOGS_DIR / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n✓ Test results saved to {results_path}")


if __name__ == "__main__":
    main()
