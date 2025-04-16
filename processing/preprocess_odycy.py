from glob import glob
from pathlib import Path
from typing import Iterable

import spacy
from tqdm import tqdm

nlp = spacy.load("grc_odycy_joint_trf")
nlp.max_length = 2100000  # Increase max_length to 2 million


def iterate_fables(file_path: Path) -> Iterable[tuple[str, str]]:
    with open(file_path) as in_file:
        contents = in_file.read()
        fables = contents.split("\n\n\n\n")
        for fable in fables:
            fable_name, fable_text = fable.split("\n\n\n")
            yield fable_name, fable_text




dat_path = Path("/work/text_reuse/raw_single_file/")
files = list(dat_path.rglob("*.txt"))

for file in tqdm(files, desc="Going through all texts."):
    file_id = file.stem
    group = str(file.parent).split("/")[-1]

    out_path = Path(f"/work/gospel-ancient-greek/gospel-ancient-greek/data/spacy_objects/{group}/")
    out_path.mkdir(exist_ok=True, parents=True)

    out_file_path = out_path.joinpath(f"{file_id}.spacy")

    if not out_file_path.is_file():
        with open(file) as in_file:
            doc_content = in_file.read()
            if len(doc_content) < 2100000:
                doc = nlp(doc_content)
                doc.to_disk(out_file_path)
            else:
                print(f"File {file} is too large.")
