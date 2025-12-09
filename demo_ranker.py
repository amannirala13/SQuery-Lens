#!/usr/bin/env python3
"""
Schema Ranker Demo - Test the API for RAG integration
Run: python demo_ranker.py
"""
import sys
sys.path.insert(0, 'src')

from schema_ranker import SchemaRanker

def main():
    print("="*60)
    print("Schema Ranker Demo - RAG Integration Layer")
    print("="*60)
    
    # Initialize ranker
    print("\n[1] Loading Schema Ranker model...")
    ranker = SchemaRanker()
    print("    ✓ Model loaded")
    
    # Example schema (EV charging domain)
    schema = {
        "drivers": ["id", "email", "name", "payment_modes", "created_at"],
        "sessions": ["id", "driver_id", "tariff_id", "energy_used", "duration", "total_cost", "date"],
        "tariffs": ["id", "name", "cost_per_kwh", "valid_from", "valid_to"],
        "chargers": ["id", "address", "connector_types", "status", "group_id"],
        "charger_groups": ["id", "name", "owner_id", "region"],
        "support_tickets": ["id", "user_id", "charger_id", "session_id", "issue", "status"],
        "reviews": ["id", "user_id", "charger_id", "rating", "comment"],
        "payments": ["id", "session_id", "amount", "method", "status", "date"],
        "invoices": ["id", "driver_id", "month", "total_amount", "due_date", "paid_date"],
        "usage_stats": ["id", "charger_id", "date", "total_sessions", "total_energy"],
        "maintenance_logs": ["id", "charger_id", "issue", "resolved_at", "technician"],
        "promotions": ["id", "code", "discount_pct", "valid_from", "valid_to"],
        "user_promotions": ["id", "user_id", "promotion_id", "used_at"],
        "audit_logs": ["id", "user_id", "action", "resource", "timestamp"],
        "settings": ["id", "key", "value", "updated_at"],
    }
    
    # Load with column info
    print("\n[2] Loading schema (15 tables)...")
    ranker.load_schema_with_columns(schema)
    print(f"    ✓ {len(schema)} tables loaded and embedded")
    
    # Test queries
    queries = [
        "Which customer paid the most?",
        "Show chargers that need maintenance",
        "List all sessions from last week",
        "Find users who used discount codes",
        "Get revenue by charger location",
        "Show unresolved support issues",
    ]
    
    print("\n[3] Testing queries...")
    print("="*60)
    
    for query in queries:
        hints = ranker.get_hints(query, top_k=5)
        
        print(f"\nQuery: \"{query}\"")
        print(f"Top 5 hints for RAG:")
        for table, score in hints:
            bar = "█" * int(score * 20)
            print(f"  {table:20s} {score:.3f} {bar}")
        print(f"→ Filter: {hints.table_names[:3]}")
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter queries to test (or 'quit' to exit):\n")
    
    while True:
        try:
            query = input("Query> ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if not query:
                continue
            
            hints = ranker.get_hints(query, top_k=10)
            
            print(f"\nTop 10 table hints:")
            for i, (table, score) in enumerate(hints, 1):
                bar = "█" * int(score * 20)
                print(f"  {i:2d}. {table:20s} {score:.3f} {bar}")
            
            print(f"\n→ RAG filter: {hints.table_names[:5]}")
            print(f"→ Boost weights: { {t: round(w, 2) for t, w in list(hints.to_boost_weights().items())[:3]} }...")
            print()
            
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    print("\nDone!")


if __name__ == "__main__":
    main()
