import os
import glob

def check_text_lengths(directory):
    # Get a list of all text files in the directory
    files = glob.glob(os.path.join(directory, '*.txt'))

    # Iterate over the files and print their names and lengths
    for file in files:
        with open(file, 'r') as f:
            content = f.read()
            print(f'File: {os.path.basename(file)}, Length: {len(content)}')

# Use the function
check_text_lengths('/Users/au619572/Documents/git_repos/gospel-ancient-greek/data/raw_single_file/greek_histeriographies')