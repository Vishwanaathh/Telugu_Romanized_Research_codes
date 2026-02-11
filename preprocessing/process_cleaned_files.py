import pandas as pd
import re
import os
import string

def classical_nlp_cleanup(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Remove Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 4. Remove Numbers
    text = re.sub(r'\d+', '', text)
    
    # 5. Remove Extra Whitespaces
    text = " ".join(text.split())
    
    return text

def process_cleaned_files():
    # Set the path to the Datasets folder
    dataset_path = "../Datasets"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Directory '{dataset_path}' not found!")
        return

    # Identify all files starting with 'cleaned_' in the Datasets folder
    files = [f for f in os.listdir(dataset_path) if f.startswith('cleaned_') and f.endswith('.csv')]
    
    if not files:
        print(f"No 'cleaned_' files found in {dataset_path}!")
        return

    for file_name in files:
        full_path = os.path.join(dataset_path, file_name)
        print(f"--- Refining Text In-Place: {file_name} ---")
        
        # Load the file
        df = pd.read_csv(full_path)
        
        # Find the comment column
        target_col = next((col for col in df.columns if 'comment' in col.lower()), None)
        
        if target_col:
            # Apply NLP cleanup
            df[target_col] = df[target_col].apply(classical_nlp_cleanup)
            
            # Remove rows that are now empty (e.g., they only contained punctuation or numbers)
            initial_count = len(df)
            df = df[df[target_col].str.strip() != ""]
            final_count = len(df)
            
            # Save back to the same path (In-Place)
            df.to_csv(full_path, index=False, encoding='utf-8')
            print(f"✅ Cleanup complete. Kept {final_count} rows (Removed {initial_count - final_count} empty rows).")
        else:
            print(f"⚠️ Skipping: Could not find comment column in {file_name}")

if __name__ == "__main__":
    process_cleaned_files()