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
kai_list = ["καὶ", "και", "καί"]

def load_files(dat_path) -> list[dict]:
    files = list(dat_path.rglob("*.spacy"))
    records = []
    for file in tqdm(files):
        text_name = file.stem
        doc = Doc(nlp.vocab).from_disk(file)
        work = str(file.parent).split("/")[-1]
        records.append(dict(text_name=text_name, doc=doc, work=work))
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
        ttr=ttr(lemmata), 
        mattr_10=mattr(lemmata, 10), 
        mattr_50=mattr(lemmata, 50), 
        mattr_500=mattr(lemmata, 500),
        mattr_1000=mattr(lemmata,1000),
        mattr_2500=mattr(lemmata, 2500),
        n_types = len(set(doc)),
        n_lemmata = len(set(lemmata)),
    )

def segment_by_and(doc):
    segments = []
    current_start = 0  # Track the start index of the current segment

    for i, token in enumerate(doc):
        if token.text.lower() in kai_list:
            # Add the current segment as a Span
            segments.append(doc[current_start:i])
            current_start = i + 1  # Move the start to after "καί" or "και"

    # Append the last segment if there's remaining text
    if current_start < len(doc):
        segments.append(doc[current_start:])
    
    return segments

Doc.set_extension("segments", getter=segment_by_and, force=True)

def lengths(doc: Doc) -> dict[str, Union[int, float]]:
    sentence_lengths = [len(sent) for sent in doc.sents]
    token_lenghts = [len(token.orth_) for token in doc]
    kai_lengths = [len(segment) for segment in doc._.segments]
    # remove kai_lengths that are 4 or below
    kai_lengths = [length for length in kai_lengths if length > 4]

    # Initialize counters
    kai_after_punct_count = 0
    punct_count = 0

    # Iterate through the tokens in the Doc
    for i, token in enumerate(doc):
        if token.is_punct:
            punct_count += 1
            # Check if the next token is "καὶ"
            if i + 1 < len(doc) and doc[i + 1].text.lower() in kai_list:
                kai_after_punct_count += 1

    return dict(
        length=len(doc),
        mean_sentence_length=np.mean(sentence_lengths),  # type: ignore
        mean_token_length=np.mean(token_lenghts),
        n_sentences=len([i for i in doc.sents]),
        mean_kai_length=np.mean(kai_lengths),
        n_kai=len(kai_lengths),
        kai_token_ratio = len(kai_lengths) / len(doc),
        kai_after_punct_count = kai_after_punct_count,
        punct_count = punct_count,
    )

def kai_richness(doc: Doc) -> dict[str, float]:
    lemmata = [token.lemma_ for token in doc]
    return dict(
        maktr_500=moving_average_kai_token_ratio(lemmata, 500), maktr_1000=moving_average_kai_token_ratio(lemmata, 1000)
    )

def moving_average_kai_token_ratio(tokens: list[str], window_size: int = 50) -> list[float]:
    """Calculates moving type-token-ratios for each window in a text."""
    counter = Counter(tokens[:window_size])
    n_kai = sum(counter.get(kai, 0) for kai in kai_list)
    ttrs = [n_kai / window_size]
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
        n_kai = sum(counter.get(kai, 0) for kai in kai_list)
        ttrs.append(n_kai / window_size)
    mattrs = np.mean(ttrs)
    return mattrs
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

dat_path = Path("/work/gospel-ancient-greek/gospel-ancient-greek/data/")

out_path = dat_path.joinpath("results/stylistic_features.csv")
out_path.parent.mkdir(exist_ok=True, parents=True)

print("Calculating vocabulary richness.")
data = pd.DataFrame(load_files(dat_path))

records = []
for doc in tqdm(data["doc"], desc="Processing documents."):
    record = {
        "n_question_marks": n_question_marks(doc),
        "man_occurs": man_occurs(doc),
        "genre_occurs": genre_marker(doc),
        **vocabulary_richness(doc),
        **lengths(doc),
        **kai_richness(doc),
    }
    records.append(record)
data = pd.concat([data, pd.DataFrame.from_records(records)], axis=1)

print("Saving results")
res = data.drop(columns=["doc"])
res.to_csv(out_path)

print("Done.")
