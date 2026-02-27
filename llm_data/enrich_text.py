import pandas as pd
import os

def enrich_text(df_path='features/vehicle_features.csv'):
    """
    Enriches structured data with natural language descriptions.
    - vehicle_summary: A readable summary of the vehicle's state.
    - maintenance_recommendation: LLM-ready instruction field.
    """
    if not os.path.exists(df_path):
        print("Featured data not found. Run engineer_features.py first.")
        return
        
    df = pd.read_csv(df_path)
    
    def generate_summary(row):
        return f"A {row['Vehicle_Age']}-year-old {row['Vehicle_Model']} with {row['Mileage']} km mileage. Last service was on {row['Last_Service_Date']}."
        
    def generate_recommendation(row):
        if row['Need_Maintenance'] == 1:
            return f"Immediate maintenance required for {row['Vehicle_Model']} due to {row['Reported_Issues']} reported issues and high mileage intensity."
        else:
            return f"Vehicle is in stable condition. Next routine check recommended after 5,000 km."

    df['vehicle_summary'] = df.apply(generate_summary, axis=1)
    df['maintenance_recommendation'] = df.apply(generate_recommendation, axis=1)
    
    # Save enriched data
    os.makedirs('llm_data', exist_ok=True)
    df.to_csv('llm_data/text_enriched.csv', index=False)
    print("Enriched data saved to llm_data/text_enriched.csv")

if __name__ == "__main__":
    enrich_text()
