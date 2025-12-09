"""
Table Ranking Fine-tuning
Trains the model to better rank tables by relevance using contrastive learning
"""
import sys
import json
import re
import time
from pathlib import Path
from typing import List, Tuple, Set
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizer
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import config
from model.model import load_model, EnhancedSQLAnalyzer


class TableRankingDataset(Dataset):
    """Dataset for table ranking with contrastive pairs"""
    
    def __init__(
        self,
        queries: List[str],
        positive_tables: List[List[str]],  # Tables mentioned in each query's SQL
        all_tables: List[str],  # All unique tables in dataset
        tokenizer: DistilBertTokenizer,
        num_negatives: int = 5,
        max_length: int = config.MAX_LENGTH
    ):
        self.queries = queries
        self.positive_tables = positive_tables
        self.all_tables = all_tables
        self.tokenizer = tokenizer
        self.num_negatives = num_negatives
        self.max_length = max_length
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[idx]
        pos_tables = self.positive_tables[idx]
        
        # Sample negatives (tables not in this query's SQL)
        neg_candidates = [t for t in self.all_tables if t not in pos_tables]
        neg_tables = random.sample(neg_candidates, min(self.num_negatives, len(neg_candidates)))
        
        # Tokenize query
        query_encoding = self.tokenizer(
            query,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(0),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(0),
            'positive_tables': pos_tables,
            'negative_tables': neg_tables
        }


def extract_tables_from_sql(sql: str) -> List[str]:
    """Extract table names from SQL query"""
    if pd.isna(sql):
        return []
    
    sql_upper = sql.upper()
    tables = set()
    
    # FROM clause
    from_pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    tables.update(re.findall(from_pattern, sql, re.IGNORECASE))
    
    # JOIN clause
    join_pattern = r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    tables.update(re.findall(join_pattern, sql, re.IGNORECASE))
    
    # INTO clause (for INSERT)
    into_pattern = r'\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    tables.update(re.findall(into_pattern, sql, re.IGNORECASE))
    
    # UPDATE clause
    update_pattern = r'\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    tables.update(re.findall(update_pattern, sql, re.IGNORECASE))
    
    # Filter out SQL keywords that might be caught
    sql_keywords = {'SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'AND', 'OR', 'GROUP', 'ORDER', 'BY', 'LIMIT'}
    tables = {t.lower() for t in tables if t.upper() not in sql_keywords}
    
    return list(tables)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for table ranking
    Uses InfoNCE-style loss to bring positive pairs closer
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        query_embeddings: torch.Tensor,  # [batch, dim]
        positive_embeddings: torch.Tensor,  # [batch, dim]
        negative_embeddings: torch.Tensor  # [batch, num_neg, dim]
    ) -> torch.Tensor:
        """
        Compute contrastive loss
        """
        batch_size = query_embeddings.size(0)
        
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=-1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=-1)
        
        # Positive similarity: [batch]
        pos_sim = torch.sum(query_embeddings * positive_embeddings, dim=-1) / self.temperature
        
        # Negative similarity: [batch, num_neg]
        neg_sim = torch.bmm(
            negative_embeddings,  # [batch, num_neg, dim]
            query_embeddings.unsqueeze(-1)  # [batch, dim, 1]
        ).squeeze(-1) / self.temperature  # [batch, num_neg]
        
        # Contrastive loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [batch, 1 + num_neg]
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_embeddings.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


class TableRankingTrainer:
    """Trainer for table ranking fine-tuning"""
    
    def __init__(self, model: EnhancedSQLAnalyzer, tokenizer: DistilBertTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.DEVICE
        
        # Only train the table projection layer (freeze everything else)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.table_projection.parameters():
            param.requires_grad = True
        
        self.criterion = ContrastiveLoss(temperature=0.1)
        
        self.optimizer = AdamW(
            self.model.table_projection.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        print(f"Training table projection layer only")
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")
    
    def encode_table(self, table_name: str) -> torch.Tensor:
        """Encode a single table name"""
        encoding = self.tokenizer(
            table_name,
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            bert_out = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = bert_out.last_hidden_state[:, 0, :]
            shared_repr = self.model.shared_layer(cls_output)
        
        # Only table_projection is trainable
        table_emb = self.model.table_projection(shared_repr)
        return table_emb.squeeze(0)
    
    def train_step(self, batch) -> float:
        """Single training step"""
        self.model.train()
        
        query_ids = batch['query_input_ids'].to(self.device)
        query_mask = batch['query_attention_mask'].to(self.device)
        pos_tables = batch['positive_tables']  # List of lists
        neg_tables = batch['negative_tables']  # List of lists
        
        batch_size = query_ids.size(0)
        
        # Get query embeddings
        with torch.no_grad():
            bert_out = self.model.bert(input_ids=query_ids, attention_mask=query_mask)
            cls_output = bert_out.last_hidden_state[:, 0, :]
            shared_repr = self.model.shared_layer(cls_output)
        
        query_embeddings = self.model.table_projection(shared_repr)  # [batch, dim]
        
        # Encode positive tables (use first positive for simplicity)
        pos_embeddings = []
        for tables in pos_tables:
            if tables:
                emb = self.encode_table(tables[0])
            else:
                emb = torch.zeros(query_embeddings.size(1), device=self.device)
            pos_embeddings.append(emb)
        pos_embeddings = torch.stack(pos_embeddings)  # [batch, dim]
        
        # Encode negative tables
        neg_embeddings = []
        for tables in neg_tables:
            table_embs = []
            for t in tables:
                table_embs.append(self.encode_table(t))
            if table_embs:
                neg_embeddings.append(torch.stack(table_embs))
            else:
                neg_embeddings.append(torch.zeros(1, query_embeddings.size(1), device=self.device))
        
        # Pad to same number of negatives
        max_neg = max(e.size(0) for e in neg_embeddings)
        padded_neg = []
        for e in neg_embeddings:
            if e.size(0) < max_neg:
                padding = torch.zeros(max_neg - e.size(0), e.size(1), device=self.device)
                e = torch.cat([e, padding], dim=0)
            padded_neg.append(e)
        neg_embeddings = torch.stack(padded_neg)  # [batch, max_neg, dim]
        
        # Compute loss
        self.optimizer.zero_grad()
        loss = self.criterion(query_embeddings, pos_embeddings, neg_embeddings)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, train_loader: DataLoader, num_epochs: int = 3):
        """Full training loop"""
        print(f"\n{'='*60}")
        print("Table Ranking Fine-tuning")
        print(f"{'='*60}")
        
        for epoch in range(num_epochs):
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                loss = self.train_step(batch)
                total_loss += loss
                pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Save updated model
        checkpoint_path = config.MODEL_DIR / "best_model_table_ranking.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, checkpoint_path)
        print(f"\nModel saved to {checkpoint_path}")
        
        return self.model


def prepare_table_ranking_data():
    """Prepare training data for table ranking"""
    print("Loading data...")
    df = pd.read_csv(config.DATA_PATH)
    
    if config.MAX_SAMPLES and len(df) > config.MAX_SAMPLES:
        df = df.sample(n=config.MAX_SAMPLES, random_state=config.RANDOM_SEED)
    
    print(f"Processing {len(df)} samples...")
    
    # Extract queries and tables
    queries = []
    positive_tables = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting tables"):
        # Get query
        if 'structured_input' in df.columns:
            text = row['structured_input']
            if pd.notna(text) and 'prompt:' in text:
                text = text.split('prompt:')[1]
                if 'schema:' in text:
                    text = text.split('schema:')[0]
                text = text.strip()
            else:
                text = ""
        else:
            text = row.get('instruction', '')
        
        if not text:
            continue
        
        # Get tables from SQL
        sql = row.get('sql', row.get('query', ''))
        tables = extract_tables_from_sql(sql)
        
        if tables:  # Only include samples with identifiable tables
            queries.append(text)
            positive_tables.append(tables)
    
    # Get all unique tables
    all_tables = list(set(t for tables in positive_tables for t in tables))
    
    print(f"\nDataset stats:")
    print(f"  Queries: {len(queries)}")
    print(f"  Unique tables: {len(all_tables)}")
    print(f"  Sample tables: {all_tables[:10]}")
    
    return queries, positive_tables, all_tables


def collate_fn(batch):
    """Custom collate for variable-length table lists"""
    return {
        'query_input_ids': torch.stack([b['query_input_ids'] for b in batch]),
        'query_attention_mask': torch.stack([b['query_attention_mask'] for b in batch]),
        'positive_tables': [b['positive_tables'] for b in batch],
        'negative_tables': [b['negative_tables'] for b in batch]
    }


def main():
    print(f"\n{'='*60}")
    print("Table Ranking Fine-tuning")
    print(f"{'='*60}")
    print(f"Device: {config.DEVICE}")
    
    # Load base model
    print("\nLoading pre-trained model...")
    model = load_model()
    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Prepare data
    queries, positive_tables, all_tables = prepare_table_ranking_data()
    
    # Create dataset
    dataset = TableRankingDataset(
        queries=queries,
        positive_tables=positive_tables,
        all_tables=all_tables,
        tokenizer=tokenizer,
        num_negatives=5
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Train
    trainer = TableRankingTrainer(model, tokenizer)
    trainer.train(dataloader, num_epochs=3)
    
    # Test
    print(f"\n{'='*60}")
    print("Testing improved table ranking")
    print(f"{'='*60}")
    
    test_cases = [
        ("Find customers who placed orders", ["customers", "orders", "products", "logs"]),
        ("Calculate total sales by category", ["sales", "categories", "users", "config"]),
    ]
    
    model.eval()
    for query, tables in test_cases:
        encoding = tokenizer(
            query, add_special_tokens=True, max_length=config.MAX_LENGTH,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(config.DEVICE)
            attention_mask = encoding['attention_mask'].to(config.DEVICE)
            
            outputs = model(input_ids, attention_mask)
            table_embeddings = model.encode_tables(tables)
            scores = model.rank_tables(outputs['query_embedding'], table_embeddings)
            scores = scores[0].cpu().numpy()
        
        print(f"\nQuery: \"{query}\"")
        for table, score in sorted(zip(tables, scores), key=lambda x: x[1], reverse=True):
            print(f"  {table}: {score*100:.1f}%")


if __name__ == "__main__":
    main()
