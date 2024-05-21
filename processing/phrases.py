import glob
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from spacy.tokens import Doc

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
            fable_name = file.stem
            doc = Doc(nlp.vocab).from_disk(file)
            records.append(dict(work_id=work_id, fable_name=fable_name, doc=doc))
    return records


def top_freq_group(
    labels, doc_term_matrix, vocab, top_k=10
) -> dict[str, tuple[str, float]]:
    unique_labels = np.unique(labels)
    res = {}
    for label in unique_labels:
        freq = np.squeeze(np.asarray(doc_term_matrix[labels == label].sum(axis=0)))
        high = np.argpartition(-freq, top_k)[:top_k]
        importance = freq[high]
        high = high[np.argsort(-importance)]
        res[label] = list(zip(vocab[high], freq[high]))
    return res


out_path = Path("results/phrases.csv")
out_path.parent.mkdir(exist_ok=True)

print("Calculating vocabulary richness.")
data = pd.DataFrame(load_works())

print("Removing stop words, lowercasing.")
lemmatized_text = data["doc"].map(
    lambda d: " ".join(tok.lemma_ for tok in d if not tok.is_stop)
)

print("Counting frequencies.")
vectorizer = CountVectorizer(ngram_range=(2, 4))
dtm = vectorizer.fit_transform(lemmatized_text)
vocab = vectorizer.get_feature_names_out()

print("Calculating top words in works and fables.")
top_freq_per_class = top_freq_group(data["work_id"], dtm, vocab)

print("Saving results")
res = pd.DataFrame(top_freq_per_class)
res.to_csv(out_path)

print("Done.")
