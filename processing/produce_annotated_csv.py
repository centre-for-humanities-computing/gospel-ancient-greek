import glob
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import Doc, Token

nlp = spacy.load("grc_odycy_joint_trf")


def load_works() -> list[dict]:
    works = glob.glob("data/spacy_objects/*")
    works = map(Path, works)
    works = [work for work in works if work.is_dir()]
    records = []
    for work in works:
        work_id = work.stem
        files = glob.glob(str(work.joinpath("*.spacy")))
        files = map(Path, files)
        for file in files:
            text_name = file.stem
            doc = Doc(nlp.vocab).from_disk(file)
            records.append(dict(work_id=work_id, text_name=text_name, doc=doc))
    return records


def get_token_features(token: Token) -> dict:
    return dict(
        token=token.orth_,
        lemma=token.lemma_,
        norm=token.norm_,
        is_stop=token.is_stop,
        upos_tag=token.pos_,
        fine_grained_tag=token.tag_,
        dependency_relation=token.dep_,
    )


out_path = Path("annotations")
out_path.mkdir(exist_ok=True)

print("Loading data.")
texts = load_works()

print("Exporting text annotations as csv.")
for text in texts:
    work_id = text["work_id"]
    text_name = text["text_name"]
    doc = text["doc"]
    out_file = out_path.joinpath(f"{work_id}__{text_name}.csv")
    entries = [get_token_features(token) for token in doc]
    table = pd.DataFrame.from_records(entries)
    table.to_csv(out_file)

print("Done.")
