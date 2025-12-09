"""
Validation Script for SQL Category Classifier
"""
import sys
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm

from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import config
from model.data_processor import DataProcessor
from model.model import load_model


class Validator:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = config.DEVICE
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def predict_batch(self, dataloader):
        all_cat_preds, all_cat_labels = [], []
        all_subcat_preds, all_subcat_labels = [], []
        
        for batch in tqdm(dataloader, desc="Validating"):
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
    
    def compute_metrics(self, predictions, labels, class_names, task_name):
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
        
        print(f"\n{task_name} Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}
    
    def validate(self, val_loader):
        cat_preds, cat_labels, subcat_preds, subcat_labels = self.predict_batch(val_loader)
        
        category_names = list(self.processor.category_encoder.classes_)
        subcategory_names = list(self.processor.subcategory_encoder.classes_)
        
        cat_metrics = self.compute_metrics(cat_preds, cat_labels, category_names, "Category")
        subcat_metrics = self.compute_metrics(subcat_preds, subcat_labels, subcategory_names, "Subcategory")
        
        exact_match = np.mean((cat_preds == cat_labels) & (subcat_preds == subcat_labels))
        print(f"\nExact Match Rate: {exact_match:.4f}")
        
        return {'category': cat_metrics, 'subcategory': subcat_metrics, 'exact_match': exact_match}


def main():
    print(f"\nSQL Category Classifier - Validation")
    print(f"Device: {config.DEVICE}")
    
    if not config.BEST_MODEL_PATH.exists():
        print(f"Error: No trained model found. Run train.py first.")
        sys.exit(1)
    
    processor = DataProcessor()
    df = processor.load_and_process_data()
    cat_labels, subcat_labels = processor.prepare_labels(df)
    split_data = processor.split_data(df, cat_labels, subcat_labels)
    dataloaders = processor.create_dataloaders(split_data)
    
    num_cat, num_subcat = processor.get_num_classes()
    model = load_model(num_cat, num_subcat)
    
    validator = Validator(model, processor)
    results = validator.validate(dataloaders['val'])
    
    results_path = config.LOGS_DIR / "validation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
