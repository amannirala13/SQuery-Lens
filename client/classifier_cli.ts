/**
 * SQL Category Classifier - TypeScript with CLI (SLOW - for testing only)
 * 
 * ⚠️ WARNING: This spawns Python for each request, loading the model every time.
 * For production, use the API server approach instead (api_server.py + classifier.ts)
 */

import { spawn } from "child_process";
import * as path from "path";

interface ClassificationResult {
    query: string;
    category: string;
    subcategory: string;
    category_confidence: number;
    subcategory_confidence: number;
}

/**
 * Classify a query using the Python CLI
 * NOTE: This is SLOW (~5-10 seconds per call) because it loads the model each time
 */
async function classifyWithCLI(query: string): Promise<ClassificationResult> {
    return new Promise((resolve, reject) => {
        const pythonPath = path.join(__dirname, "venv", "bin", "python3");
        const scriptPath = path.join(__dirname, "classify_cli.py");

        const process = spawn(pythonPath, [scriptPath, query, "--json"]);

        let stdout = "";
        let stderr = "";

        process.stdout.on("data", (data) => {
            stdout += data.toString();
        });

        process.stderr.on("data", (data) => {
            stderr += data.toString();
        });

        process.on("close", (code) => {
            if (code !== 0) {
                reject(new Error(`Python process exited with code ${code}: ${stderr}`));
                return;
            }

            try {
                // Find the JSON line in output (skip any logging)
                const lines = stdout.trim().split("\n");
                const jsonLine = lines.find((line) => line.startsWith("{"));
                if (!jsonLine) {
                    reject(new Error("No JSON output found"));
                    return;
                }
                const result = JSON.parse(jsonLine);
                resolve(result);
            } catch (e) {
                reject(new Error(`Failed to parse output: ${stdout}`));
            }
        });
    });
}

// Example usage
async function main() {
    console.log("⚠️  CLI approach - loads model each time (slow)");
    console.log("   For production, use: python api_server.py\n");

    const queries = [
        "Calculate total sales by region",
        "Delete expired sessions",
        "Show all customers",
    ];

    for (const query of queries) {
        console.log(`\nClassifying: "${query}"`);
        console.time("Time");

        try {
            const result = await classifyWithCLI(query);
            console.log(`  → ${result.subcategory} (${(result.subcategory_confidence * 100).toFixed(1)}%)`);
        } catch (error) {
            console.error("Error:", error);
        }

        console.timeEnd("Time");
    }
}

main().catch(console.error);

export { classifyWithCLI, ClassificationResult };
