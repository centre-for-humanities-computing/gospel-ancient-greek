from glob import glob
from pathlib import Path
from typing import Iterable

import spacy
from tqdm import tqdm

nlp = spacy.load("grc_odycy_joint_trf")


def iterate_fables(file_path: Path) -> Iterable[tuple[str, str]]:
    with open(file_path) as in_file:
        contents = in_file.read()
        fables = contents.split("\n\n\n\n")
        for fable in fables:
            fable_name, fable_text = fable.split("\n\n\n")
            yield fable_name, fable_text


out_path = Path("data/spacy_objects")
out_path.mkdir(exist_ok=True, parents=True)

files = glob("data/raw/*.txt")
files = [Path(file) for file in files]
for file in tqdm(files, desc="Going through all fables."):
    file_id = file.stem
    current_folder = out_path.joinpath(file_id)
    current_folder.mkdir(exist_ok=True, parents=True)
    for fable_name, fable_content in iterate_fables(file):
        out_file_path = current_folder.joinpath(f"{fable_name}.spacy")
        if not out_file_path.is_file():
            doc = nlp(fable_content)
            doc.to_disk(out_file_path)
