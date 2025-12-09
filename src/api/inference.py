"""
Interactive Inference Script for SQL Category Classifier
Use this to classify natural language queries after training
"""
import sys
import torch
from transformers import DistilBertTokenizer

import config
from data_processor import DataProcessor
from model import load_model


class SQLClassifier:
    """Easy-to-use interface for SQL category classification"""
    
    def __init__(self):
        self.device = config.DEVICE
        print(f"Loading model on {self.device}...")
        
        # Load processor and encoders
        self.processor = DataProcessor()
        self.processor.load_sql_categories()
        self.processor.load_encoders()
        
        # Load model
        num_cat = len(self.processor.category_encoder.classes_)
        num_subcat = len(self.processor.subcategory_encoder.classes_)
        self.model = load_model(num_cat, num_subcat)
        
        # Tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
        
        print("Model loaded successfully!\n")
    
    @torch.no_grad()
    def predict(self, text: str) -> dict:
        """Predict category and subcategory for input text"""
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
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
            'category_confidence': float(cat_probs[0, cat_pred]),
            'subcategory_confidence': float(subcat_probs[0, subcat_pred])
        }
    
    def classify(self, text: str):
        """Pretty print classification result"""
        result = self.predict(text)
        print(f"Input: {text}")
        print(f"  Category:    {result['category']} ({result['category_confidence']*100:.1f}%)")
        print(f"  Subcategory: {result['subcategory']} ({result['subcategory_confidence']*100:.1f}%)")
        return result


def main():
    """Interactive mode for testing the classifier"""
    if not config.BEST_MODEL_PATH.exists():
        print("Error: No trained model found. Run train.py first.")
        sys.exit(1)
    
    classifier = SQLClassifier()
    
    print("SQL Category Classifier - Interactive Mode")
    print("Enter natural language queries to classify.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            query = input("\nEnter query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not query:
                continue
            
            classifier.classify(query)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
