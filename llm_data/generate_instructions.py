import pandas as pd
import json
import os

def create_instruction_dataset(df_path='llm_data/text_enriched.csv'):
    """
    Phase 7: Instruction Dataset Generation
    Generates Instruction -> Input -> Output pairs for fine-tuning.
    """
    if not os.path.exists(df_path): return
    df = pd.read_csv(df_path)
    
    dataset = []
    for _, row in df.head(100).iterrows():
        sample = {
            "instruction": "Evaluate the maintenance risk for this vehicle.",
            "input": row['vehicle_summary'],
            "output": row['maintenance_recommendation']
        }
        dataset.append(sample)
        
    os.makedirs('llm_data', exist_ok=True)
    with open('llm_data/instruction_dataset.jsonl', 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')
            
    print("Instruction dataset saved to llm_data/instruction_dataset.jsonl")

if __name__ == "__main__":
    create_instruction_dataset()
