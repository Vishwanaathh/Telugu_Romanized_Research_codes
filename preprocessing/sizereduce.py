import os

def trim_file_to_size(input_path, output_path, max_mb=10):
    max_bytes = max_mb * 1024 * 1024
    current_bytes = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line_bytes = len(line.encode('utf-8'))
            if current_bytes + line_bytes > max_bytes:
                print(f"Reached limit. Trimmed file size: {current_bytes / (1024*1024):.2f} MB")
                break
            
            outfile.write(line)
            current_bytes += line_bytes

# Usage
input_csv = '../Datasets/gpteacher_trimmed.csv'
output_csv = '../Datasets/gpteacher_trimmedd.csv'

trim_file_to_size(input_csv, output_csv)