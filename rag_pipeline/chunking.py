"""
RAG Chunking Module
Loads vehicle maintenance knowledge from text_enriched.csv and instruction_dataset.jsonl,
and creates text chunks suitable for embedding and retrieval.
"""
import os
import json
import csv


def load_instruction_chunks(jsonl_path: str) -> list[str]:
    """Load instruction-output pairs from the JSONL dataset."""
    chunks = []
    if not os.path.exists(jsonl_path):
        return chunks
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            instruction = record.get("instruction", "")
            inp = record.get("input", "")
            output = record.get("output", "")
            chunk = f"Q: {instruction} {inp}\nA: {output}"
            chunks.append(chunk)
    return chunks


def load_enriched_chunks(csv_path: str, max_rows: int = 500) -> list[str]:
    """Load vehicle summaries and recommendations from text_enriched.csv.
    
    Samples up to max_rows representative records to keep the index manageable.
    """
    chunks = []
    if not os.path.exists(csv_path):
        return chunks
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            summary = row.get("vehicle_summary", "").strip()
            recommendation = row.get("maintenance_recommendation", "").strip()
            model = row.get("Vehicle_Model", "")
            risk = row.get("risk_level", "")
            mileage = row.get("Mileage", "")
            age = row.get("Vehicle_Age", "")
            tire = row.get("Tire_Condition", "")
            brake = row.get("Brake_Condition", "")
            battery = row.get("Battery_Status", "")
            issues = row.get("Reported_Issues", "")

            chunk = (
                f"Vehicle: {model}, Age: {age} years, Mileage: {mileage} km, "
                f"Issues: {issues}, Risk: {risk}. "
                f"Tire: {tire}, Brake: {brake}, Battery: {battery}. "
                f"{summary} {recommendation}"
            )
            chunks.append(chunk)
    return chunks


# General vehicle maintenance knowledge base
MAINTENANCE_KNOWLEDGE = [
    "Regular oil changes should be performed every 5,000 to 7,500 km or every 6 months, whichever comes first. Synthetic oil can extend this interval to 10,000 km.",
    "Tire pressure should be checked monthly. Under-inflated tires reduce fuel efficiency by up to 3% and cause uneven wear. Recommended pressure is usually 30-35 PSI.",
    "Brake pads typically need replacement every 40,000 to 70,000 km. Warning signs include squeaking, grinding noises, and longer stopping distances.",
    "Battery life averages 3-5 years. Signs of a weak battery include slow engine cranking, dim headlights, and electrical issues. Test battery voltage regularly.",
    "Coolant should be flushed and replaced every 50,000 km or every 2 years. Low coolant levels can cause engine overheating and severe damage.",
    "Air filters should be replaced every 20,000 to 30,000 km. A clogged air filter reduces engine performance and fuel efficiency.",
    "Transmission fluid should be checked every 50,000 km. Dark or burnt-smelling fluid indicates it needs replacement.",
    "Worn out brakes are the most critical safety concern. Brake condition is one of the top predictors of maintenance needs according to our ML model.",
    "Weak battery status is a strong indicator of maintenance requirement. Battery_Status_Weak is one of the top 3 features driving maintenance predictions.",
    "The number of reported issues is the strongest single predictor of maintenance needs. Vehicles with 3+ issues have very high maintenance probability.",
    "Service history frequency matters: vehicles with fewer past services tend to need more urgent maintenance.",
    "Accident history increases maintenance risk. Each past accident adds to the cumulative wear on vehicle components.",
    "Poor maintenance history significantly increases the probability of needing maintenance compared to good or average maintenance history.",
    "Electric vehicles generally have lower mechanical maintenance needs but battery health monitoring is critical.",
    "For fleet management, prioritize vehicles with high mileage-to-age ratio (usage intensity) for proactive maintenance scheduling.",
    "SHAP analysis shows the top factors driving maintenance prediction: Reported Issues, Brake Condition (Worn Out), Battery Status (Weak), Service History, and Maintenance History.",
]


def get_all_chunks(
    project_root: str | None = None,
) -> list[str]:
    """Build and return all chunks from available data sources."""
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    jsonl_path = os.path.join(project_root, "llm_data", "instruction_dataset.jsonl")
    csv_path = os.path.join(project_root, "llm_data", "text_enriched.csv")

    chunks = list(MAINTENANCE_KNOWLEDGE)
    chunks.extend(load_instruction_chunks(jsonl_path))
    chunks.extend(load_enriched_chunks(csv_path, max_rows=500))

    return chunks


if __name__ == "__main__":
    chunks = get_all_chunks()
    print(f"Total chunks: {len(chunks)}")
    print(f"Sample chunk: {chunks[0][:200]}...")
