import glob
from collections import Counter
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Doc
from tqdm import tqdm

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


def moving_ttr(tokens: list[str], window_size: int = 50) -> list[float]:
    """Calculates moving type-token-ratios for each window in a text."""
    counter = Counter(tokens[:window_size])
    n_types = len(counter)
    ttrs = [n_types / window_size]
    for i in range(len(tokens) - window_size):
        old_word = tokens[i]
        new_word = tokens[i + window_size]
        counter[old_word] -= 1
        if not counter[old_word]:
            del counter[old_word]
        if new_word in counter:
            counter[new_word] += 1
        else:
            counter[new_word] = 1
        n_types = len(counter)
        ttrs.append(n_types / window_size)
    return ttrs


def mattr(tokens: list[str], window_size: int = 50) -> float:
    ttrs = moving_ttr(tokens, window_size)
    return np.mean(ttrs)


def ttr(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens)


def vocabulary_richness(doc: Doc) -> dict[str, float]:
    lemmata = [token.lemma_ for token in doc]
    return dict(
        mattr_10=mattr(lemmata, 10), mattr_50=mattr(lemmata, 50), ttr=ttr(lemmata)
    )


def lengths(doc: Doc) -> dict[str, Union[int, float]]:
    sentence_lengths = [len(sent) for sent in doc.sents]
    token_lenghts = [len(token.orth_) for token in doc]
    return dict(
        length=len(doc),
        mean_sentence_length=np.mean(sentence_lengths),  # type: ignore
        mean_token_length=np.mean(token_lenghts),
        n_sentences=len(list(doc.sents)),
    )


def n_question_marks(doc: Doc) -> int:
    return len([tok for tok in doc if tok.orth_ == ";"])


def genre_marker(doc: Doc) -> bool:
    """Looks at whether the following words
    occur in the first three sentences: μῦθος, αἶνος, λόγος, παραβολή
    """
    for sent in list(doc.sents)[:3]:
        for tok in sent:
            if tok.lemma_ in {"μῦθος", "αἶνος", "λόγος", "παραβολή"}:
                return True
    return False


def man_occurs(doc: Doc) -> bool:
    """Looks at whether the following words
    occur in the first three sentences: τίς, ἀνήρ, ἄνθρωπος
    """
    for sent in list(doc.sents)[:3]:
        for tok in sent:
            if tok.lemma_ in {"τίς", "ἀνήρ", "ἄνθρωπος"}:
                return True
    return False


out_path = Path("results/stylistic_features.csv")
out_path.parent.mkdir(exist_ok=True)

print("Calculating vocabulary richness.")
data = pd.DataFrame(load_works())

records = []
for doc in tqdm(data["doc"], desc="Processing documents."):
    record = {
        "n_question_marks": n_question_marks(doc),
        "man_occurs": man_occurs(doc),
        "genre_occurs": genre_marker(doc),
        **vocabulary_richness(doc),
        **lengths(doc),
    }
    records.append(record)
data = pd.concat([data, pd.DataFrame.from_records(records)], axis=1)

print("Saving results")
res = data.drop(columns=["doc"])
res.to_csv(out_path)

print("Done.")
