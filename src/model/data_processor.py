"""
Enhanced Data Processor for SQL Query Analyzer
Extracts complexity, keywords, table counts, and prepares multi-label data
"""
import pandas as pd
import numpy as np
import re
import pickle
from typing import Tuple, Dict, List, Set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from tqdm import tqdm

from . import config


class EnhancedSQLDataset(Dataset):
    """PyTorch Dataset for Enhanced SQL Query Analysis"""
    
    def __init__(
        self, 
        texts: List[str],
        complexity_labels: np.ndarray,
        keyword_labels: np.ndarray,  # Multi-label binary
        category_labels: np.ndarray,
        subcategory_labels: np.ndarray,  # Multi-label binary
        table_count_labels: np.ndarray,
        tokenizer: DistilBertTokenizer,
        max_length: int = config.MAX_LENGTH
    ):
        self.texts = texts
        self.complexity_labels = complexity_labels
        self.keyword_labels = keyword_labels
        self.category_labels = category_labels
        self.subcategory_labels = subcategory_labels
        self.table_count_labels = table_count_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'complexity_label': torch.tensor(self.complexity_labels[idx], dtype=torch.long),
            'keyword_labels': torch.tensor(self.keyword_labels[idx], dtype=torch.float),
            'category_label': torch.tensor(self.category_labels[idx], dtype=torch.long),
            'subcategory_labels': torch.tensor(self.subcategory_labels[idx], dtype=torch.float),
            'table_count_label': torch.tensor(self.table_count_labels[idx], dtype=torch.long),
        }


class EnhancedDataProcessor:
    """Handles all data processing for enhanced model"""
    
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
        
        # Single-label encoders
        self.complexity_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.table_count_encoder = LabelEncoder()
        
        # Multi-label binarizers
        self.keyword_binarizer = MultiLabelBinarizer(classes=config.SQL_KEYWORDS)
        self.subcategory_binarizer = MultiLabelBinarizer(classes=config.SUBCATEGORIES)
        
        # Fit binarizers with all classes
        self.keyword_binarizer.fit([config.SQL_KEYWORDS])
        self.subcategory_binarizer.fit([config.SUBCATEGORIES])
    
    def extract_sql_keywords(self, sql: str) -> List[str]:
        """Extract SQL keywords from query"""
        if pd.isna(sql):
            return []
        
        sql_upper = sql.upper()
        found = []
        
        # Sort by length to match longer keywords first
        sorted_keywords = sorted(config.SQL_KEYWORDS, key=len, reverse=True)
        
        for kw in sorted_keywords:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, sql_upper):
                found.append(kw)
        
        # Check for subquery
        if '(' in sql and 'SELECT' in sql_upper:
            if sql_upper.count('SELECT') > 1:
                found.append('SUBQUERY')
        
        return list(set(found))
    
    def determine_complexity(self, sql: str, keywords: List[str]) -> str:
        """Determine query complexity based on SQL structure"""
        if pd.isna(sql):
            return "simple"
        
        sql_upper = sql.upper()
        
        # Complex indicators
        complex_indicators = [
            'SUBQUERY' in keywords,
            any(kw in keywords for kw in ['UNION', 'INTERSECT', 'EXCEPT']),
            sql_upper.count('JOIN') >= 2,
            sql_upper.count('SELECT') > 1,
            'CASE' in keywords,
            'HAVING' in keywords,
            any(kw in sql_upper for kw in ['OVER(', 'OVER (']),
        ]
        
        # Medium indicators
        medium_indicators = [
            any(kw in keywords for kw in ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN']),
            'GROUP BY' in keywords,
            any(kw in keywords for kw in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']),
            'ORDER BY' in keywords,
            len(keywords) > 5,
        ]
        
        if sum(complex_indicators) >= 2:
            return "complex"
        elif sum(medium_indicators) >= 2:
            return "medium"
        else:
            return "simple"
    
    def count_tables(self, sql: str) -> str:
        """Estimate number of tables in query"""
        if pd.isna(sql):
            return "1"
        
        sql_upper = sql.upper()
        
        # Count JOINs
        join_count = len(re.findall(r'\bJOIN\b', sql_upper))
        
        # Count table references after FROM
        from_match = re.search(r'\bFROM\b(.+?)(?:WHERE|GROUP|ORDER|LIMIT|$)', sql_upper, re.DOTALL)
        
        table_count = 1  # At least one table
        if join_count > 0:
            table_count = join_count + 1
        
        if table_count == 1:
            return "1"
        elif table_count == 2:
            return "2"
        else:
            return "3+"
    
    def determine_category(self, sql: str, sql_type: str = None) -> str:
        """Determine main category"""
        if pd.isna(sql):
            return "Data Manipulation"
        
        sql_upper = sql.upper().strip()
        
        if sql_upper.startswith(('CREATE', 'ALTER', 'DROP', 'TRUNCATE')):
            return "Schema Definition"
        elif any(sql_upper.startswith(kw) for kw in ['GRANT', 'REVOKE']):
            return "Security"
        elif any(sql_upper.startswith(kw) for kw in ['BEGIN', 'COMMIT', 'ROLLBACK']):
            return "Transaction Control"
        elif 'UNION' in sql_upper or 'INTERSECT' in sql_upper or 'EXCEPT' in sql_upper:
            return "Set Operations"
        else:
            return "Data Manipulation"
    
    def determine_subcategories(self, sql: str, keywords: List[str]) -> List[str]:
        """Determine subcategories (multi-label)"""
        subcats = []
        
        if pd.isna(sql):
            return ["Read"]
        
        sql_upper = sql.upper()
        
        # Read/Write
        if sql_upper.strip().startswith('SELECT'):
            subcats.append("Read")
        if any(sql_upper.strip().startswith(kw) for kw in ['INSERT', 'UPDATE', 'DELETE']):
            subcats.append("Write")
        
        # Aggregation
        if any(kw in keywords for kw in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
            subcats.append("Aggregation")
        
        # Filtering
        if 'WHERE' in keywords:
            subcats.append("Filtering")
        
        # Sorting
        if 'ORDER BY' in keywords:
            subcats.append("Sorting")
        
        # Joining
        if any(kw in keywords for kw in ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN']):
            subcats.append("Joining")
        
        # Grouping
        if 'GROUP BY' in keywords:
            subcats.append("Grouping")
        
        # Limiting
        if any(kw in keywords for kw in ['LIMIT', 'OFFSET', 'DISTINCT']):
            subcats.append("Limiting")
        
        # Subquery
        if 'SUBQUERY' in keywords:
            subcats.append("Subquery")
        
        # Windowing
        if any(kw in sql_upper for kw in ['OVER(', 'OVER (', 'PARTITION BY']):
            subcats.append("Windowing")
        
        # Set Logic
        if any(kw in keywords for kw in ['UNION', 'INTERSECT', 'EXCEPT']):
            subcats.append("Set Logic")
        
        # DDL
        if any(kw in keywords for kw in ['CREATE', 'ALTER', 'DROP']):
            subcats.append("Object DDL")
        
        if not subcats:
            subcats = ["Read"]
        
        return subcats
    
    def load_and_process_data(self) -> pd.DataFrame:
        """Load data and extract all features"""
        print("Loading data...")
        df = pd.read_csv(config.DATA_PATH)
        print(f"Loaded {len(df)} samples from {config.DATA_PATH.name}")
        
        # Limit samples if configured
        if config.MAX_SAMPLES and len(df) > config.MAX_SAMPLES:
            df = df.sample(n=config.MAX_SAMPLES, random_state=config.RANDOM_SEED)
            print(f"Using {config.MAX_SAMPLES} samples (subset)")
        
        # Extract instruction (handle different formats)
        if 'structured_input' in df.columns:
            print("Detected text-to-sql-100K format")
            
            def extract_prompt(text):
                if pd.isna(text):
                    return ""
                if 'prompt:' in text:
                    text = text.split('prompt:')[1]
                if 'schema:' in text:
                    text = text.split('schema:')[0]
                return text.strip()
            
            df['instruction'] = df['structured_input'].apply(extract_prompt)
            df['sql'] = df['sql']
        else:
            df['instruction'] = df['instruction']
            df['sql'] = df['query']
        
        # Process each row
        print("\nExtracting features...")
        
        all_keywords = []
        all_complexity = []
        all_category = []
        all_subcategory = []
        all_table_count = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            sql = row['sql']
            
            # Extract keywords
            keywords = self.extract_sql_keywords(sql)
            all_keywords.append(keywords)
            
            # Determine complexity
            complexity = self.determine_complexity(sql, keywords)
            all_complexity.append(complexity)
            
            # Determine category
            category = self.determine_category(sql)
            all_category.append(category)
            
            # Determine subcategories (multi-label)
            subcategories = self.determine_subcategories(sql, keywords)
            all_subcategory.append(subcategories)
            
            # Count tables
            table_count = self.count_tables(sql)
            all_table_count.append(table_count)
        
        df['keywords'] = all_keywords
        df['complexity'] = all_complexity
        df['category'] = all_category
        df['subcategory'] = all_subcategory
        df['table_count'] = all_table_count
        
        # Print distributions
        print("\n=== Feature Distributions ===")
        print("\nComplexity:")
        print(df['complexity'].value_counts())
        print("\nCategory:")
        print(df['category'].value_counts())
        print("\nTable Count:")
        print(df['table_count'].value_counts())
        print("\nTop 10 Keywords:")
        all_kw = [kw for kws in df['keywords'] for kw in kws]
        kw_counts = pd.Series(all_kw).value_counts().head(10)
        print(kw_counts)
        
        return df
    
    def prepare_labels(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Encode all labels"""
        
        # Single-label encoders
        complexity_labels = self.complexity_encoder.fit_transform(df['complexity'])
        category_labels = self.category_encoder.fit_transform(df['category'])
        table_count_labels = self.table_count_encoder.fit_transform(df['table_count'])
        
        # Multi-label binarizers
        keyword_labels = self.keyword_binarizer.transform(df['keywords'])
        subcategory_labels = self.subcategory_binarizer.transform(df['subcategory'])
        
        print(f"\n=== Label Shapes ===")
        print(f"Complexity: {complexity_labels.shape} ({len(self.complexity_encoder.classes_)} classes)")
        print(f"Keywords: {keyword_labels.shape} ({len(config.SQL_KEYWORDS)} classes)")
        print(f"Category: {category_labels.shape} ({len(self.category_encoder.classes_)} classes)")
        print(f"Subcategory: {subcategory_labels.shape} ({len(config.SUBCATEGORIES)} classes)")
        print(f"Table Count: {table_count_labels.shape} ({len(self.table_count_encoder.classes_)} classes)")
        
        return {
            'complexity': complexity_labels,
            'keywords': keyword_labels,
            'category': category_labels,
            'subcategory': subcategory_labels,
            'table_count': table_count_labels
        }
    
    def split_data(self, df: pd.DataFrame, labels: Dict[str, np.ndarray]) -> Dict:
        """Split data into train/val/test"""
        texts = df['instruction'].values
        n = len(texts)
        
        # Create indices and split
        indices = np.arange(n)
        
        train_idx, temp_idx = train_test_split(
            indices, 
            test_size=(config.VAL_RATIO + config.TEST_RATIO),
            random_state=config.RANDOM_SEED
        )
        
        val_ratio_adj = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1 - val_ratio_adj),
            random_state=config.RANDOM_SEED
        )
        
        print(f"\n=== Data Split ===")
        print(f"Training:   {len(train_idx)} samples ({len(train_idx)/n*100:.1f}%)")
        print(f"Validation: {len(val_idx)} samples ({len(val_idx)/n*100:.1f}%)")
        print(f"Test:       {len(test_idx)} samples ({len(test_idx)/n*100:.1f}%)")
        
        def get_split(idx):
            return {
                'texts': texts[idx],
                'complexity': labels['complexity'][idx],
                'keywords': labels['keywords'][idx],
                'category': labels['category'][idx],
                'subcategory': labels['subcategory'][idx],
                'table_count': labels['table_count'][idx]
            }
        
        return {
            'train': get_split(train_idx),
            'val': get_split(val_idx),
            'test': get_split(test_idx)
        }
    
    def create_dataloaders(self, split_data: Dict) -> Dict[str, DataLoader]:
        """Create PyTorch DataLoaders"""
        dataloaders = {}
        
        for split_name, data in split_data.items():
            dataset = EnhancedSQLDataset(
                texts=list(data['texts']),
                complexity_labels=data['complexity'],
                keyword_labels=data['keywords'],
                category_labels=data['category'],
                subcategory_labels=data['subcategory'],
                table_count_labels=data['table_count'],
                tokenizer=self.tokenizer
            )
            
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=(split_name == 'train'),
                num_workers=0
            )
        
        return dataloaders
    
    def save_encoders(self):
        """Save all encoders"""
        encoders = {
            'complexity_encoder': self.complexity_encoder,
            'category_encoder': self.category_encoder,
            'table_count_encoder': self.table_count_encoder,
            'keyword_binarizer': self.keyword_binarizer,
            'subcategory_binarizer': self.subcategory_binarizer
        }
        with open(config.LABEL_ENCODERS_PATH, 'wb') as f:
            pickle.dump(encoders, f)
        print(f"Encoders saved to {config.LABEL_ENCODERS_PATH}")
    
    def load_encoders(self):
        """Load all encoders"""
        with open(config.LABEL_ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
        self.complexity_encoder = encoders['complexity_encoder']
        self.category_encoder = encoders['category_encoder']
        self.table_count_encoder = encoders['table_count_encoder']
        self.keyword_binarizer = encoders['keyword_binarizer']
        self.subcategory_binarizer = encoders['subcategory_binarizer']
        print("Encoders loaded successfully")


# Keep old class for backward compatibility
DataProcessor = EnhancedDataProcessor
SQLCategoryDataset = EnhancedSQLDataset


if __name__ == "__main__":
    processor = EnhancedDataProcessor()
    df = processor.load_and_process_data()
    labels = processor.prepare_labels(df)
    split_data = processor.split_data(df, labels)
    dataloaders = processor.create_dataloaders(split_data)
    processor.save_encoders()
    
    print("\n=== Test Batch ===")
    batch = next(iter(dataloaders['train']))
    for key, val in batch.items():
        print(f"{key}: {val.shape}")
