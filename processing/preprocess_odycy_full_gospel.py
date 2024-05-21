from glob import glob
from pathlib import Path
from typing import Iterable

import spacy
from tqdm import tqdm

nlp = spacy.load("grc_odycy_joint_trf")

def iterate_gospel(file_path: Path) -> Iterable[tuple[str, str]]:
    with open(file_path) as in_file:
        contents = in_file.read()
        yield 'gospel_of_mark', contents


out_path = Path("data/spacy_objects")
out_path.mkdir(exist_ok=True, parents=True)

files = Path("data/raw/NA28-041MRK_full.txt")
file_id = files.stem
current_folder = out_path.joinpath(file_id)
current_folder.mkdir(exist_ok=True, parents=True)
for gospel_name, gospel_content in iterate_gospel(files):
    out_file_path = current_folder.joinpath(f"{gospel_name}.spacy")
    if not out_file_path.is_file():
        doc = nlp(gospel_content)
        doc.to_disk(out_file_path)