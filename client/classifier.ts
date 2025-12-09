/**
 * SQL Query Analyzer - TypeScript Client
 * Enhanced client for multi-output classification and table ranking
 */

// ============================================================
// Types
// ============================================================

export interface TableScore {
    table: string;
    confidence: number;
}

export interface AnalyzeResult {
    query: string;
    complexity: "simple" | "medium" | "complex";
    complexity_confidence: number;
    keywords: string[];
    category: string;
    category_confidence: number;
    subcategories: string[];
    estimated_tables: "1" | "2" | "3+";
    table_count_confidence: number;
    tables?: TableScore[];
}

export interface ModelInfo {
    complexity_labels: string[];
    keywords: string[];
    categories: string[];
    subcategories: string[];
    table_count_labels: string[];
    device: string;
}

interface AnalyzeRequest {
    query: string;
    tables?: string[];
}

// ============================================================
// Client
// ============================================================

export class SQLQueryAnalyzer {
    private baseUrl: string;

    constructor(baseUrl: string = "http://localhost:8000") {
        this.baseUrl = baseUrl;
    }

    /**
     * Check if API server is running
     */
    async healthCheck(): Promise<boolean> {
        try {
            const response = await fetch(`${this.baseUrl}/`);
            const data = await response.json();
            return data.status === "ok";
        } catch {
            return false;
        }
    }

    /**
     * Get model info and available labels
     */
    async getModelInfo(): Promise<ModelInfo> {
        const response = await fetch(`${this.baseUrl}/info`);
        if (!response.ok) throw new Error(`Failed: ${response.statusText}`);
        return response.json();
    }

    /**
     * Analyze a natural language query
     * 
     * @param query - Natural language query
     * @param tables - Optional list of table names for relevance ranking
     */
    async analyze(query: string, tables?: string[]): Promise<AnalyzeResult> {
        const response = await fetch(`${this.baseUrl}/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, tables } as AnalyzeRequest),
        });

        if (!response.ok) throw new Error(`Failed: ${response.statusText}`);
        return response.json();
    }

    /**
     * Analyze multiple queries
     */
    async analyzeBatch(queries: string[], tables?: string[]): Promise<AnalyzeResult[]> {
        const response = await fetch(`${this.baseUrl}/analyze/batch`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ queries, tables }),
        });

        if (!response.ok) throw new Error(`Failed: ${response.statusText}`);
        const data = await response.json();
        return data.results;
    }

    /**
     * Rank tables by relevance to a query
     * 
     * @param query - Natural language query
     * @param tables - List of table names to rank
     * @returns Sorted list of tables with confidence scores
     */
    async rankTables(query: string, tables: string[]): Promise<TableScore[]> {
        const result = await this.analyze(query, tables);
        return result.tables || [];
    }

    /**
     * Get top-K most relevant tables for a query
     */
    async getTopTables(query: string, tables: string[], k: number = 5): Promise<string[]> {
        const ranked = await this.rankTables(query, tables);
        return ranked.slice(0, k).map(t => t.table);
    }
}

// ============================================================
// Example Usage
// ============================================================

async function main() {
    const analyzer = new SQLQueryAnalyzer("http://localhost:8000");

    // Check server
    console.log("Checking API server...");
    if (!(await analyzer.healthCheck())) {
        console.error("Server not running. Start with: ./run_server.sh");
        process.exit(1);
    }
    console.log("✓ Server is running\n");

    // Get model info
    console.log("Model Info:");
    const info = await analyzer.getModelInfo();
    console.log(`  Categories: ${info.categories.length}`);
    console.log(`  Keywords: ${info.keywords.length}`);
    console.log(`  Device: ${info.device}\n`);

    // === Example 1: Basic Analysis ===
    console.log("=".repeat(60));
    console.log("Example 1: Basic Query Analysis");
    console.log("=".repeat(60));

    const result = await analyzer.analyze(
        "Find customers who spent more than $1000 last month"
    );

    console.log(`Query: "${result.query}"`);
    console.log(`\nClassification:`);
    console.log(`  Complexity: ${result.complexity} (${(result.complexity_confidence * 100).toFixed(1)}%)`);
    console.log(`  Category: ${result.category} (${(result.category_confidence * 100).toFixed(1)}%)`);
    console.log(`  Subcategories: ${result.subcategories.join(", ")}`);
    console.log(`  Keywords: ${result.keywords.join(", ")}`);
    console.log(`  Estimated Tables: ${result.estimated_tables}`);

    // === Example 2: Table Ranking ===
    console.log("\n" + "=".repeat(60));
    console.log("Example 2: Table Ranking");
    console.log("=".repeat(60));

    const tables = [
        "customers",
        "orders",
        "products",
        "inventory",
        "employees",
        "departments",
        "logs",
        "config"
    ];

    const resultWithTables = await analyzer.analyze(
        "Find customers who spent more than $1000 last month",
        tables
    );

    console.log(`Query: "${resultWithTables.query}"`);
    console.log(`\nTable Relevance Ranking:`);
    resultWithTables.tables?.forEach((t, i) => {
        console.log(`  ${i + 1}. ${t.table}: ${(t.confidence * 100).toFixed(1)}%`);
    });

    // === Example 3: Multiple Queries ===
    console.log("\n" + "=".repeat(60));
    console.log("Example 3: Batch Analysis");
    console.log("=".repeat(60));

    const queries = [
        "Show all users",
        "Calculate total revenue by product category",
        "Delete expired sessions",
    ];

    const batchResults = await analyzer.analyzeBatch(queries);

    for (const r of batchResults) {
        console.log(`\n"${r.query}"`);
        console.log(`  Complexity: ${r.complexity}`);
        console.log(`  Keywords: ${r.keywords.slice(0, 5).join(", ")}`);
    }
}

main().catch(console.error);
