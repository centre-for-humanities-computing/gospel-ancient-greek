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

files = glob("data/raw_single_file/*/*.txt")
files = [Path(file) for file in files]
for file in tqdm(files, desc="Going through all texts."):
    file_id = file.stem
    out_file_path = out_path.joinpath(f"{file_id}.spacy")
    with open(file) as in_file:
        doc_content = in_file.read()
        if len(doc_content) < 1000000:
            if not out_file_path.is_file():
                doc = nlp(doc_content)
                doc.to_disk(out_file_path)
        else:
            print(f"File {file} is too large.")
