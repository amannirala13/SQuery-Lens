"""
Data Processor for Column Ranker
Extracts query-column pairs with relevance labels from data_table_2
"""
import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict
from pathlib import Path
from tqdm import tqdm
import random

from sentence_transformers import InputExample

from . import config


def extract_headers(table_str: str) -> List[str]:
    """Extract column headers from table string"""
    match = re.search(r"header':\s*array\(\[(.*?)\], dtype", str(table_str), re.DOTALL)
    if match:
        headers = re.findall(r"'([^']+)'", match.group(1))
        # Filter out garbage
        headers = [h for h in headers if len(h) > 1 and len(h) < 50 and not h.startswith(':')]
        return headers
    return []


def extract_sql_columns(sql_str: str) -> Tuple[int, List[int]]:
    """Extract selected column and condition columns from SQL"""
    sel_match = re.search(r"'sel':\s*(\d+)", str(sql_str))
    conds_match = re.search(r"'column_index':\s*array\(\[([^\]]*)\]", str(sql_str))
    
    sel = int(sel_match.group(1)) if sel_match else None
    conds = []
    if conds_match and conds_match.group(1).strip():
        try:
            conds = [int(x.strip()) for x in conds_match.group(1).split(',') if x.strip()]
        except:
            pass
    
    return sel, conds


def create_training_pairs(df: pd.DataFrame, max_samples: int = None) -> List[InputExample]:
    """
    Create training pairs for Sentence-BERT
    
    For each query:
    - Positive pairs: query + relevant column (label=1.0)
    - Negative pairs: query + irrelevant column (label=0.0)
    """
    examples = []
    
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=config.RANDOM_SEED)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating pairs"):
        question = str(row['question']).strip()
        if not question or len(question) < 5:
            continue
            
        headers = extract_headers(row['table'])
        if len(headers) < 2:
            continue
            
        sel, conds = extract_sql_columns(row['sql'])
        if sel is None:
            continue
        
        # Get relevant column indices
        relevant_indices = set()
        if sel < len(headers):
            relevant_indices.add(sel)
        for c in conds:
            if c < len(headers):
                relevant_indices.add(c)
        
        if not relevant_indices:
            continue
        
        # Create positive pairs (relevant columns)
        for idx in relevant_indices:
            examples.append(InputExample(
                texts=[question, headers[idx]],
                label=1.0
            ))
        
        # Create negative pairs (sample from irrelevant columns)
        irrelevant_indices = [i for i in range(len(headers)) if i not in relevant_indices]
        n_negatives = min(len(irrelevant_indices), len(relevant_indices) * 2)  # 2 negatives per positive
        
        for idx in random.sample(irrelevant_indices, n_negatives) if n_negatives > 0 else []:
            examples.append(InputExample(
                texts=[question, headers[idx]],
                label=0.0
            ))
    
    return examples


def load_and_prepare_data(split: str = "train") -> List[InputExample]:
    """Load data and create training examples"""
    
    if split == "train":
        path = config.TRAIN_PATH
    elif split == "val":
        path = config.VAL_PATH
    else:
        path = config.TEST_PATH
    
    print(f"Loading {split} data from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows")
    
    max_samples = config.MAX_SAMPLES if split == "train" else min(10000, len(df))
    examples = create_training_pairs(df, max_samples)
    
    print(f"Created {len(examples)} training pairs")
    
    # Count pos/neg
    pos = sum(1 for e in examples if e.label > 0.5)
    neg = len(examples) - pos
    print(f"  Positive: {pos}, Negative: {neg}")
    
    return examples


class ColumnRankingDataset:
    """Helper class to hold dataset info"""
    
    def __init__(self):
        self.train_examples = None
        self.val_examples = None
    
    def load(self):
        """Load train and validation data"""
        self.train_examples = load_and_prepare_data("train")
        self.val_examples = load_and_prepare_data("val")
        return self


if __name__ == "__main__":
    # Test data loading
    dataset = ColumnRankingDataset()
    dataset.load()
    
    print("\n=== Sample Training Pairs ===")
    for ex in dataset.train_examples[:5]:
        label = "RELEVANT" if ex.label > 0.5 else "NOT RELEVANT"
        print(f"\n{label}:")
        print(f"  Query: {ex.texts[0]}")
        print(f"  Column: {ex.texts[1]}")
