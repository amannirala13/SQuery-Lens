#!/usr/bin/env python3
"""
CLI for SQL Category Classifier
Usage: python classify_cli.py "your query here"

NOTE: This loads the model on every call (~5-10 sec).
For production, use the API server (api_server.py) instead.
"""
import sys
import json
import argparse
import torch
from transformers import DistilBertTokenizer

import config
from data_processor import DataProcessor
from model import load_model


def classify(query: str) -> dict:
    """Classify a single query and return result as dict"""
    # Load processor and encoders
    processor = DataProcessor()
    processor.load_sql_categories()
    processor.load_encoders()
    
    # Load model
    num_cat = len(processor.category_encoder.classes_)
    num_subcat = len(processor.subcategory_encoder.classes_)
    model = load_model(num_cat, num_subcat)
    
    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Predict
    with torch.no_grad():
        encoding = tokenizer(
            query,
            add_special_tokens=True,
            max_length=config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(config.DEVICE)
        attention_mask = encoding['attention_mask'].to(config.DEVICE)
        
        cat_logits, subcat_logits = model(input_ids, attention_mask)
        
        cat_probs = torch.softmax(cat_logits, dim=1)
        subcat_probs = torch.softmax(subcat_logits, dim=1)
        
        cat_pred = torch.argmax(cat_logits, dim=1).item()
        subcat_pred = torch.argmax(subcat_logits, dim=1).item()
        
        category = processor.category_encoder.inverse_transform([cat_pred])[0]
        subcategory = processor.subcategory_encoder.inverse_transform([subcat_pred])[0]
    
    return {
        'query': query,
        'category': category,
        'subcategory': subcategory,
        'category_confidence': float(cat_probs[0, cat_pred]),
        'subcategory_confidence': float(subcat_probs[0, subcat_pred])
    }


def main():
    parser = argparse.ArgumentParser(description='Classify SQL queries')
    parser.add_argument('query', type=str, help='Natural language query to classify')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    # Suppress transformers warnings
    import warnings
    warnings.filterwarnings('ignore')
    import logging
    logging.getLogger('transformers').setLevel(logging.ERROR)
    
    result = classify(args.query)
    
    if args.json:
        print(json.dumps(result))
    else:
        print(f"Query: {result['query']}")
        print(f"Category: {result['category']} ({result['category_confidence']*100:.1f}%)")
        print(f"Subcategory: {result['subcategory']} ({result['subcategory_confidence']*100:.1f}%)")


if __name__ == '__main__':
    main()
