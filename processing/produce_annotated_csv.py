import glob
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import Doc, Token
from tqdm import tqdm

nlp = spacy.load("grc_odycy_joint_trf")


def load_files(dat_path) -> list[dict]:
    files = list(dat_path.rglob("*.spacy"))
    records = []
    for file in tqdm(files):
        text_name = file.stem
        doc = Doc(nlp.vocab).from_disk(file)
        work = str(file.parent).split("/")[-1]
        records.append(dict(text_name=text_name, doc=doc, work=work))
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

dat_path = Path("/work/gospel-ancient-greek/gospel-ancient-greek/data/")

out_path = dat_path.joinpath("annotations/")
out_path.mkdir(exist_ok=True, parents=True)


print("Loading data.")
texts = load_files(dat_path)

print("Exporting text annotations as csv.")
for text in texts:
    work_id = text["work"]
    text_name = text["text_name"]
    doc = text["doc"]
    out_file = out_path.joinpath(f"{work_id}__{text_name}.csv")
    entries = [get_token_features(token) for token in doc]
    table = pd.DataFrame.from_records(entries)
    table.to_csv(out_file)

print("Done.")
