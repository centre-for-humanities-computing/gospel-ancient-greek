#from ../processing.utils import save_csv_as_html
import glob

import pandas as pd
from pathlib import Path

def save_csv_as_html(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    basename = Path(file_path).stem

    # Convert DataFrame to HTML table
    html_table = df.to_html(index=False)

    # Save the HTML table to a file in your docs folder (e.g., docs/raw_data.html)
    with open(f"docs/results/{basename}.html", "w") as f:
        f.write(html_table)
    
    return

csv_files = glob.glob("results/*.csv")

for file_path in csv_files:
    save_csv_as_html(file_path)
