#!/usr/bin/env python3
"""
Test script for table ranking
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import DistilBertTokenizer
from model import config
from model.model import EnhancedSQLAnalyzer

def main():
    print('Loading fine-tuned table ranking model...')
    model = EnhancedSQLAnalyzer()
    checkpoint = torch.load(
        config.MODEL_DIR / 'best_model_table_ranking.pt', 
        map_location=config.DEVICE, 
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()
    
    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
    print('Model loaded!\n')
    
    # Test cases with realistic table lists
    test_cases = [
        # E-commerce scenario
        ('Find customers who bought products last week', 
         ['customers', 'orders', 'products', 'order_items', 'categories', 
          'shipping', 'returns', 'reviews', 'logs', 'config']),
        
        # HR scenario
        ('Calculate average salary by department',
         ['employees', 'departments', 'salaries', 'positions', 
          'attendance', 'benefits', 'config', 'audit_logs']),
        
        # Analytics scenario
        ('Show top 10 selling products by revenue',
         ['products', 'sales', 'transactions', 'inventory', 
          'suppliers', 'customers', 'sessions', 'logs']),
        
        # User management
        ('Delete inactive user sessions',
         ['users', 'sessions', 'tokens', 'permissions', 
          'roles', 'activity_logs', 'notifications', 'config']),
        
        # Reporting
        ('Get monthly revenue report grouped by region',
         ['revenue', 'sales', 'regions', 'countries', 
          'stores', 'products', 'customers', 'calendar']),
        
        # Inventory
        ('Find products with low stock in warehouse',
         ['products', 'inventory', 'warehouses', 'stock_levels', 
          'suppliers', 'orders', 'shipments', 'alerts']),
    ]
    
    print('='*70)
    print('TABLE RANKING TEST RESULTS')
    print('='*70)
    
    for query, tables in test_cases:
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
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            table_embeddings = model.encode_tables(tables)
            scores = model.rank_tables(outputs['query_embedding'], table_embeddings)
            scores = scores[0].cpu().numpy()
        
        # Sort by relevance
        ranked = sorted(zip(tables, scores), key=lambda x: x[1], reverse=True)
        
        print(f'\nQuery: "{query}"')
        print('Ranked tables (top 5):')
        for i, (table, score) in enumerate(ranked[:5]):
            marker = '✓' if i < 3 else ' '
            print(f'  {marker} {i+1}. {table:20s} {score*100:.1f}%')
        print(f'  ... and {len(ranked)-5} more tables')


if __name__ == "__main__":
    main()
