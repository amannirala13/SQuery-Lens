"""
Enhanced API Server for SQL Query Analyzer
Provides multi-output classification and table relevance ranking
"""
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import config
from model.data_processor import EnhancedDataProcessor
from model.model import load_model, EnhancedSQLAnalyzer


# ============================================================
# Request/Response Models
# ============================================================

class AnalyzeRequest(BaseModel):
    query: str
    tables: Optional[List[str]] = None  # Optional list of tables for ranking

class TableScore(BaseModel):
    table: str
    confidence: float

class AnalyzeResponse(BaseModel):
    query: str
    complexity: str
    complexity_confidence: float
    keywords: List[str]
    category: str
    category_confidence: float
    subcategories: List[str]
    estimated_tables: str
    table_count_confidence: float
    tables: Optional[List[TableScore]] = None  # Ranked tables if provided

class BatchAnalyzeRequest(BaseModel):
    queries: List[str]
    tables: Optional[List[str]] = None

class BatchAnalyzeResponse(BaseModel):
    results: List[AnalyzeResponse]

class ModelInfo(BaseModel):
    complexity_labels: List[str]
    keywords: List[str]
    categories: List[str]
    subcategories: List[str]
    table_count_labels: List[str]
    device: str


# ============================================================
# Service Class
# ============================================================

class SQLAnalyzerService:
    """Service for SQL query analysis"""
    
    def __init__(self):
        self.device = config.DEVICE
        print(f"Loading model on {self.device}...")
        
        # Load processor and encoders
        self.processor = EnhancedDataProcessor()
        self.processor.load_encoders()
        
        # Load model
        self.model = load_model()
        
        # Tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
        
        # Cache for table embeddings
        self._table_embeddings_cache = {}
        
        print("Model loaded successfully!")
    
    def _encode_query(self, text: str):
        """Tokenize and encode query"""
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding['input_ids'].to(self.device), encoding['attention_mask'].to(self.device)
    
    def _get_table_embeddings(self, tables: List[str]) -> torch.Tensor:
        """Get or compute table embeddings"""
        # Create cache key
        cache_key = tuple(sorted(tables))
        
        if cache_key not in self._table_embeddings_cache:
            self._table_embeddings_cache[cache_key] = self.model.encode_tables(tables)
        
        return self._table_embeddings_cache[cache_key]
    
    @torch.no_grad()
    def analyze(self, query: str, tables: Optional[List[str]] = None) -> dict:
        """Analyze a query and optionally rank tables"""
        input_ids, attention_mask = self._encode_query(query)
        
        outputs = self.model(input_ids, attention_mask)
        
        # === Complexity ===
        complexity_probs = torch.softmax(outputs['complexity'], dim=1)
        complexity_idx = torch.argmax(outputs['complexity'], dim=1).item()
        complexity = self.processor.complexity_encoder.inverse_transform([complexity_idx])[0]
        complexity_conf = complexity_probs[0, complexity_idx].item()
        
        # === Keywords (multi-label) ===
        keyword_probs = torch.sigmoid(outputs['keywords'])
        keyword_mask = keyword_probs[0] > 0.5
        keywords = [
            config.SQL_KEYWORDS[i] 
            for i in range(len(config.SQL_KEYWORDS)) 
            if keyword_mask[i]
        ]
        
        # === Category ===
        category_probs = torch.softmax(outputs['category'], dim=1)
        category_idx = torch.argmax(outputs['category'], dim=1).item()
        category = self.processor.category_encoder.inverse_transform([category_idx])[0]
        category_conf = category_probs[0, category_idx].item()
        
        # === Subcategories (multi-label) ===
        subcategory_probs = torch.sigmoid(outputs['subcategory'])
        subcategory_mask = subcategory_probs[0] > 0.5
        subcategories = [
            config.SUBCATEGORIES[i]
            for i in range(len(config.SUBCATEGORIES))
            if subcategory_mask[i]
        ]
        if not subcategories:
            # At least one subcategory
            max_idx = torch.argmax(subcategory_probs[0]).item()
            subcategories = [config.SUBCATEGORIES[max_idx]]
        
        # === Table Count ===
        table_count_probs = torch.softmax(outputs['table_count'], dim=1)
        table_count_idx = torch.argmax(outputs['table_count'], dim=1).item()
        table_count = self.processor.table_count_encoder.inverse_transform([table_count_idx])[0]
        table_count_conf = table_count_probs[0, table_count_idx].item()
        
        result = {
            'query': query,
            'complexity': complexity,
            'complexity_confidence': complexity_conf,
            'keywords': keywords,
            'category': category,
            'category_confidence': category_conf,
            'subcategories': subcategories,
            'estimated_tables': table_count,
            'table_count_confidence': table_count_conf,
            'tables': None
        }
        
        # === Table Ranking (if tables provided) ===
        if tables:
            table_embeddings = self._get_table_embeddings(tables)
            scores = self.model.rank_tables(outputs['query_embedding'], table_embeddings)
            scores = scores[0].cpu().numpy()
            
            # Sort by score
            table_scores = [
                {'table': t, 'confidence': float(s)}
                for t, s in sorted(zip(tables, scores), key=lambda x: x[1], reverse=True)
            ]
            result['tables'] = table_scores
        
        return result
    
    def get_model_info(self) -> dict:
        """Get model metadata"""
        return {
            'complexity_labels': config.COMPLEXITY_LABELS,
            'keywords': config.SQL_KEYWORDS,
            'categories': list(self.processor.category_encoder.classes_),
            'subcategories': config.SUBCATEGORIES,
            'table_count_labels': config.TABLE_COUNT_LABELS,
            'device': str(self.device)
        }


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="SQL Query Analyzer API",
    description="Analyze natural language queries for SQL generation",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service
analyzer = None


@app.on_event("startup")
async def startup():
    global analyzer
    analyzer = SQLAnalyzerService()


@app.get("/")
async def root():
    return {"status": "ok", "message": "SQL Query Analyzer API v2.0"}


@app.get("/info", response_model=ModelInfo)
async def get_info():
    """Get model information"""
    return analyzer.get_model_info()


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_query(request: AnalyzeRequest):
    """
    Analyze a natural language query
    
    Optionally provide table names to get relevance ranking
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    result = analyzer.analyze(request.query, request.tables)
    return result


@app.post("/analyze/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(request: BatchAnalyzeRequest):
    """Analyze multiple queries"""
    if not request.queries:
        raise HTTPException(status_code=400, detail="Queries list cannot be empty")
    
    results = [
        analyzer.analyze(query, request.tables)
        for query in request.queries
    ]
    return {"results": results}


@app.post("/rank-tables")
async def rank_tables(query: str, tables: List[str]):
    """
    Rank tables by relevance to a query
    
    Returns tables sorted by relevance score
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not tables:
        raise HTTPException(status_code=400, detail="Tables list cannot be empty")
    
    result = analyzer.analyze(query, tables)
    return {
        "query": query,
        "ranked_tables": result['tables']
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
