"""
Enhanced SQL Query Analyzer Model
Multi-output classifier with table relevance ranking
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel
from typing import List, Dict, Tuple

from . import config


class EnhancedSQLAnalyzer(nn.Module):
    """
    Multi-output model for SQL query analysis:
    - Complexity (single-label): simple, medium, complex
    - Keywords (multi-label): SQL keywords
    - Category (single-label): Query category
    - Subcategory (multi-label): Query subcategories
    - Table Count (single-label): 1, 2, 3+
    - Table Ranking: Score relevance of tables to query
    """
    
    def __init__(
        self,
        num_complexity_classes: int = 3,
        num_keywords: int = len(config.SQL_KEYWORDS),
        num_categories: int = len(config.CATEGORIES),
        num_subcategories: int = len(config.SUBCATEGORIES),
        num_table_counts: int = 3,
        dropout_rate: float = 0.3,
        freeze_bert_layers: int = 2
    ):
        super().__init__()
        
        # Load pre-trained DistilBERT
        self.bert = DistilBertModel.from_pretrained(config.MODEL_NAME)
        hidden_size = self.bert.config.hidden_size  # 768
        
        # Freeze some layers
        if freeze_bert_layers > 0:
            for i, layer in enumerate(self.bert.transformer.layer):
                if i < freeze_bert_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Shared representation
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # === Classification Heads ===
        
        # Complexity head (single-label)
        self.complexity_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_complexity_classes)
        )
        
        # Keywords head (multi-label)
        self.keywords_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_keywords)
        )
        
        # Category head (single-label)
        self.category_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_categories)
        )
        
        # Subcategory head (multi-label)
        self.subcategory_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_subcategories)
        )
        
        # Table count head (single-label)
        self.table_count_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_table_counts)
        )
        
        # === Table Ranking Head ===
        # Projects query embedding to space for table matching
        self.table_projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256)  # Final embedding dimension
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classification head weights"""
        for module in [self.shared_layer, self.complexity_head, self.keywords_head,
                       self.category_head, self.subcategory_head, self.table_count_head,
                       self.table_projection]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def get_query_embedding(self, input_ids, attention_mask):
        """Get query embedding from BERT"""
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_out.last_hidden_state[:, 0, :]
        shared_repr = self.shared_layer(cls_output)
        return shared_repr
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass for classification heads
        
        Returns dict with all logits
        """
        shared_repr = self.get_query_embedding(input_ids, attention_mask)
        
        return {
            'complexity': self.complexity_head(shared_repr),
            'keywords': self.keywords_head(shared_repr),
            'category': self.category_head(shared_repr),
            'subcategory': self.subcategory_head(shared_repr),
            'table_count': self.table_count_head(shared_repr),
            'query_embedding': self.table_projection(shared_repr)  # For table ranking
        }
    
    def rank_tables(
        self, 
        query_embedding: torch.Tensor,
        table_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Rank tables by relevance to query
        
        Args:
            query_embedding: [batch_size, 256] from forward pass
            table_embeddings: [num_tables, 256] pre-computed table embeddings
        
        Returns:
            scores: [batch_size, num_tables] relevance scores
        """
        # Cosine similarity - normalize both embeddings properly
        query_norm = F.normalize(query_embedding, p=2, dim=-1)  # [batch, 256]
        table_norm = F.normalize(table_embeddings, p=2, dim=-1)  # [num_tables, 256]
        
        scores = torch.mm(query_norm, table_norm.t())  # [batch, num_tables]
        
        # Scale from [-1, 1] to [0, 1] for interpretability
        scores = (scores + 1) / 2
        return scores
    
    def encode_tables(self, table_texts: List[str]) -> torch.Tensor:
        """
        Encode table names/descriptions for ranking
        This should be called once and cached
        """
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
        
        embeddings = []
        self.eval()
        
        with torch.no_grad():
            for table_text in table_texts:
                encoding = tokenizer(
                    table_text,
                    add_special_tokens=True,
                    max_length=64,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(next(self.parameters()).device)
                attention_mask = encoding['attention_mask'].to(next(self.parameters()).device)
                
                bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                cls_output = bert_out.last_hidden_state[:, 0, :]
                shared_repr = self.shared_layer(cls_output)
                table_emb = self.table_projection(shared_repr)
                embeddings.append(table_emb)
        
        return torch.cat(embeddings, dim=0)


class MultiOutputLoss(nn.Module):
    """Combined loss for all outputs"""
    
    def __init__(
        self,
        complexity_weight: float = 1.0,
        keywords_weight: float = 1.0,
        category_weight: float = 1.0,
        subcategory_weight: float = 1.0,
        table_count_weight: float = 0.5
    ):
        super().__init__()
        
        self.weights = {
            'complexity': complexity_weight,
            'keywords': keywords_weight,
            'category': category_weight,
            'subcategory': subcategory_weight,
            'table_count': table_count_weight
        }
        
        # Single-label losses
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Multi-label losses
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs: Dict, labels: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss
        
        Returns:
            total_loss: Combined weighted loss
            losses: Dict of individual losses for logging
        """
        losses = {}
        
        # Complexity (single-label)
        losses['complexity'] = self.ce_loss(
            outputs['complexity'], 
            labels['complexity']
        )
        
        # Keywords (multi-label)
        losses['keywords'] = self.bce_loss(
            outputs['keywords'],
            labels['keywords']
        )
        
        # Category (single-label)
        losses['category'] = self.ce_loss(
            outputs['category'],
            labels['category']
        )
        
        # Subcategory (multi-label)
        losses['subcategory'] = self.bce_loss(
            outputs['subcategory'],
            labels['subcategory']
        )
        
        # Table count (single-label)
        losses['table_count'] = self.ce_loss(
            outputs['table_count'],
            labels['table_count']
        )
        
        # Combined loss
        total_loss = sum(
            self.weights[k] * v for k, v in losses.items()
        )
        
        return total_loss, losses


def create_model() -> EnhancedSQLAnalyzer:
    """Create model with proper configuration"""
    model = EnhancedSQLAnalyzer(
        num_complexity_classes=len(config.COMPLEXITY_LABELS),
        num_keywords=len(config.SQL_KEYWORDS),
        num_categories=len(config.CATEGORIES),
        num_subcategories=len(config.SUBCATEGORIES),
        num_table_counts=len(config.TABLE_COUNT_LABELS),
        dropout_rate=0.3,
        freeze_bert_layers=2
    )
    
    model = model.to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created on: {config.DEVICE}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable: {trainable:,}")
    
    return model


def load_model(model_path: str = None) -> EnhancedSQLAnalyzer:
    """Load trained model"""
    if model_path is None:
        model_path = config.BEST_MODEL_PATH
    
    model = EnhancedSQLAnalyzer()
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Best accuracy: {checkpoint.get('best_accuracy', 'N/A')}")
    
    return model


# Backward compatibility
SQLCategoryClassifier = EnhancedSQLAnalyzer


if __name__ == "__main__":
    model = create_model()
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    
    dummy_input = torch.randint(0, 30000, (batch_size, seq_len)).to(config.DEVICE)
    dummy_mask = torch.ones(batch_size, seq_len).to(config.DEVICE)
    
    outputs = model(dummy_input, dummy_mask)
    
    print("\nTest forward pass:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
    
    # Test table ranking
    print("\nTest table ranking:")
    tables = ["customers", "orders", "products", "employees"]
    table_embeddings = model.encode_tables(tables)
    print(f"  Table embeddings: {table_embeddings.shape}")
    
    scores = model.rank_tables(outputs['query_embedding'], table_embeddings)
    print(f"  Ranking scores: {scores.shape}")
