import pandas as pd
import re
import os

def clean_file(file_path):
    # 1. Check if path exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"--- Processing: {os.path.basename(file_path)} ---")
    
    # 2. Automatically choose the right reader based on extension
    df = None
    try:
        if file_path.lower().endswith('.xlsx'):
            # Requires: pip install openpyxl
            df = pd.read_excel(file_path)
            print(f"✅ Successfully read as EXCEL")
        else:
            # Try multiple encodings for CSV
            for enc in ['utf-8-sig', 'latin1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    print(f"✅ Successfully read as CSV ({enc})")
                    break
                except:
                    continue
    except Exception as e:
        print(f"❌ Failed to read {file_path}: {e}")
        return

    if df is None: return

    # 3. Target the 'Comments' column
    # Some files might have 'comments' or 'comment' (case insensitive)
    target_col = None
    for col in df.columns:
        if 'comment' in col.lower():
            target_col = col
            break
    
    if not target_col:
        print(f"⚠️ Column 'Comments' not found. Skipping. Columns: {list(df.columns)}")
        return

    # 4. Cleaning Logic: Remove rows with Telugu script
    def is_romanized(text):
        if pd.isna(text): return False
        text = str(text)
        # Search for Telugu Unicode range: \u0c00 to \u0c7f
        if re.search(r'[\u0c00-\u0c7f]', text):
            return False # Contains Telugu script
        return True # Strictly Romanized/English

    before = len(df)
    df_cleaned = df[df[target_col].apply(is_romanized)]
    after = len(df_cleaned)
    
    # 5. Save output
    output_name = f"cleaned_{os.path.basename(file_path).replace('.xlsx', '.csv')}"
    df_cleaned.to_csv(output_name, index=False, encoding='utf-8')
    print(f"📊 Rows kept: {after} | Rows removed: {before - after}")
    print(f"💾 Saved to: {output_name}\n")

# Your exact file paths
files_to_clean = [
    "../Datasets/telugu-hate-speech-train.xlsx - Sheet1.csv",
    "../Datasets/telugu-english-test-data-with-labels.xlsx",
    "../Datasets/telugu-hate-speech-test.xlsx",
    "../Datasets/training_data_telugu-hate.xlsx"
]

for f in files_to_clean:
    clean_file(f)