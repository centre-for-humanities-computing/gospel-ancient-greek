import glob
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from spacy.tokens import Doc
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


def top_freq_group(
    labels, doc_term_matrix, vocab, top_k=10
) -> dict[str, tuple[str, float]]:
    unique_labels = np.unique(labels)
    res = {}
    for label in unique_labels:
        mask = (labels == label).values  # if labels is a pandas Series
        freq = np.squeeze(np.asarray(doc_term_matrix[mask].sum(axis=0)))
        # freq = np.squeeze(np.asarray(doc_term_matrix[labels == label].sum(axis=0)))
        high = np.argpartition(-freq, top_k)[:top_k]
        importance = freq[high]
        high = high[np.argsort(-importance)]
        res[label] = list(zip(vocab[high], freq[high]))
    return res

dat_path = Path("/work/gospel-ancient-greek/gospel-ancient-greek/data/")

out_path = dat_path.joinpath("results/phrases.csv")
out_path.parent.mkdir(exist_ok=True, parents=True)


print("Calculating vocabulary richness.")
data = pd.DataFrame(load_files(dat_path))

print("Removing stop words, lowercasing.")
lemmatized_text = data["doc"].map(
    lambda d: " ".join(tok.lemma_ for tok in d if not tok.is_stop)
)

print("Counting frequencies.")
vectorizer = CountVectorizer(ngram_range=(2, 4))
dtm = vectorizer.fit_transform(lemmatized_text)
vocab = vectorizer.get_feature_names_out()

print("Calculating top words in works and texts.")
top_freq_per_class = top_freq_group(data["work"], dtm, vocab)

print("Saving results")
res = pd.DataFrame(top_freq_per_class)
res.to_csv(out_path)

print("Done.")
