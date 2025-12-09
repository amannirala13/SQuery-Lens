"""
Schema Ranker API - First layer for RAG systems
Provides table hints/filters for semantic search
"""
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class SchemaRanker:
    """
    First-layer schema filter for RAG systems.
    
    Usage:
        ranker = SchemaRanker()
        ranker.load_schema(["customers", "orders", "products", ...])
        
        # Get hints for RAG
        hints = ranker.get_hints("Find customers who paid the most", top_k=20)
        # hints = [("customers", 0.85), ("orders", 0.72), ...]
        
        # Use in RAG
        rag.search(query, filter_tables=hints.table_names)
    """
    
    def __init__(self, model_path: str = None):
        """Initialize Schema Ranker"""
        if model_path is None:
            model_path = str(Path(__file__).parent.parent.parent / "models" / "schema_ranker" / "schema_ranker_model")
        
        self.model = SentenceTransformer(model_path)
        self.tables: List[str] = []
        self.table_embeddings: Optional[np.ndarray] = None
        self._table_names: List[str] = []
    
    def load_schema(self, tables: List[str], descriptions: Dict[str, str] = None):
        """
        Load schema tables for ranking.
        
        Args:
            tables: List of table names
            descriptions: Optional dict of {table_name: description}
        
        Example:
            ranker.load_schema(
                ["customers", "orders", "products"],
                descriptions={
                    "customers": "Customer accounts and contact info",
                    "orders": "Purchase orders with payment data",
                    "products": "Product catalog"
                }
            )
        """
        self.tables = []
        self._table_names = []
        
        for table in tables:
            self._table_names.append(table)
            if descriptions and table in descriptions:
                self.tables.append(f"{table} - {descriptions[table]}")
            else:
                self.tables.append(table)
        
        # Pre-compute embeddings (one-time cost)
        self.table_embeddings = self.model.encode(self.tables)
        return self
    
    def load_schema_with_columns(self, schema: Dict[str, List[str]]):
        """
        Load schema with column information.
        
        Args:
            schema: Dict of {table_name: [column1, column2, ...]}
        
        Example:
            ranker.load_schema_with_columns({
                "customers": ["id", "name", "email"],
                "orders": ["id", "customer_id", "total", "date"]
            })
        """
        self.tables = []
        self._table_names = []
        
        for table, columns in schema.items():
            self._table_names.append(table)
            self.tables.append(f"{table} ({', '.join(columns)})")
        
        self.table_embeddings = self.model.encode(self.tables)
        return self
    
    def get_hints(
        self, 
        query: str, 
        top_k: int = 20,
        min_score: float = 0.0
    ) -> "SchemaHints":
        """
        Get table hints for RAG filtering.
        
        Args:
            query: Natural language query
            top_k: Number of top tables to return
            min_score: Minimum similarity score threshold
        
        Returns:
            SchemaHints object with ranked tables
        """
        if self.table_embeddings is None:
            raise ValueError("Schema not loaded. Call load_schema() first.")
        
        query_emb = self.model.encode(query)
        scores = cos_sim(query_emb, self.table_embeddings)[0].numpy()
        
        # Rank by score
        indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in indices:
            score = float(scores[idx])
            if score >= min_score:
                results.append((self._table_names[idx], score))
        
        return SchemaHints(query, results)
    
    def get_table_filter(self, query: str, top_k: int = 20) -> List[str]:
        """
        Get just table names for RAG filtering (convenience method).
        
        Returns:
            List of table names to filter on
        """
        hints = self.get_hints(query, top_k)
        return hints.table_names


class SchemaHints:
    """Container for schema ranking results"""
    
    def __init__(self, query: str, ranked_tables: List[Tuple[str, float]]):
        self.query = query
        self.ranked_tables = ranked_tables
    
    @property
    def table_names(self) -> List[str]:
        """Get just the table names"""
        return [t for t, _ in self.ranked_tables]
    
    @property
    def scores(self) -> Dict[str, float]:
        """Get scores as dict for boosting"""
        return {t: s for t, s in self.ranked_tables}
    
    def top(self, k: int) -> "SchemaHints":
        """Get top-k results"""
        return SchemaHints(self.query, self.ranked_tables[:k])
    
    def above_threshold(self, threshold: float) -> "SchemaHints":
        """Get results above score threshold"""
        filtered = [(t, s) for t, s in self.ranked_tables if s >= threshold]
        return SchemaHints(self.query, filtered)
    
    def to_rag_filter(self) -> Dict:
        """
        Convert to RAG-compatible filter format.
        Can be used with various vector DBs.
        """
        return {
            "table_names": self.table_names,
            "scores": self.scores,
            "query": self.query
        }
    
    def to_boost_weights(self, base_weight: float = 1.0) -> Dict[str, float]:
        """
        Convert to boost weights for RAG ranking.
        Higher scored tables get higher boost.
        """
        if not self.ranked_tables:
            return {}
        
        max_score = max(s for _, s in self.ranked_tables)
        return {
            table: base_weight + (score / max_score if max_score > 0 else 0)
            for table, score in self.ranked_tables
        }
    
    def __repr__(self):
        top_3 = self.ranked_tables[:3]
        return f"SchemaHints({len(self.ranked_tables)} tables, top: {top_3})"
    
    def __iter__(self):
        return iter(self.ranked_tables)
    
    def __len__(self):
        return len(self.ranked_tables)


# Convenience function
def create_ranker(model_path: str = None) -> SchemaRanker:
    """Create a new SchemaRanker instance"""
    return SchemaRanker(model_path)


if __name__ == "__main__":
    # Demo usage
    print("="*60)
    print("Schema Ranker - RAG Integration Demo")
    print("="*60)
    
    # Create ranker
    ranker = SchemaRanker()
    
    # Load schema (with descriptions for better ranking)
    ranker.load_schema(
        ["customers", "orders", "products", "payments", "shipments", 
         "reviews", "inventory", "categories", "users", "sessions",
         "support_tickets", "invoices", "returns", "coupons", "campaigns"],
        descriptions={
            "customers": "Customer accounts and contact information",
            "orders": "Purchase orders with totals and status",
            "products": "Product catalog with pricing",
            "payments": "Payment transactions and methods",
            "sessions": "User charging sessions with usage data"
        }
    )
    
    # Get hints for RAG
    query = "Find customers who paid the most last month"
    hints = ranker.get_hints(query, top_k=10)
    
    print(f"\nQuery: '{query}'")
    print(f"\nTop 10 table hints for RAG:")
    for table, score in hints:
        print(f"  {table:20s} {score:.3f}")
    
    print(f"\n--- RAG Integration ---")
    print(f"Filter tables: {hints.table_names[:5]}")
    print(f"Boost weights: {hints.to_boost_weights()}")
    print(f"RAG filter: {hints.to_rag_filter()}")
