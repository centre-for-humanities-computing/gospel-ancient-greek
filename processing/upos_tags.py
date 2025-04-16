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
) -> dict[str, tuple[str, int, float]]:
    unique_labels = np.unique(labels)
    res = {}
    for label in unique_labels:
        mask = (labels == label).values  # if labels is a pandas Series
        freq = np.squeeze(np.asarray(doc_term_matrix[mask].sum(axis=0)))
        # freq = np.squeeze(np.asarray(doc_term_matrix[labels == label].sum(axis=0)))
        rel_freq = freq / freq.sum()
        high = np.argpartition(-freq, top_k)[:top_k]
        importance = freq[high]
        high = high[np.argsort(-importance)]
        res[label] = list(zip(vocab[high], freq[high], rel_freq[high]))
    return res


def extract_upos(doc: Doc) -> str:
    tags = []
    for tok in doc:
        if (tok.pos_ == "PUNCT") and (tok.orth_ in [".", ";"]):
            tags.append("PUNCT")
        else:
            tags.append(tok.pos_)
    return " ".join(tags)

dat_path = Path("/work/gospel-ancient-greek/gospel-ancient-greek/data/")

out_path = dat_path.joinpath("results/upos_tags.csv")
out_path.parent.mkdir(exist_ok=True, parents=True)


print("Loading data.")
data = pd.DataFrame(load_files(dat_path))

print("Extracting UPOS tags.")
upos_docs = data["doc"].map(
    extract_upos,
)

print("Counting UPOS frequencies.")
vectorizer = CountVectorizer()
upos_vecs = np.asarray(vectorizer.fit_transform(upos_docs).todense())
upos_vocab = vectorizer.get_feature_names_out()

print("Saving UPOS frequencies")
freq_df = pd.DataFrame(upos_vecs, columns=upos_vocab)
freq_df["text_name"] = data["text_name"]
freq_df["work"] = data["work"]
freq_df.to_csv(out_path)

print("Collecting UPOS n-grams")
n_gram_vectorizer = CountVectorizer(ngram_range=(2, 4))
upos_ngrams = n_gram_vectorizer.fit_transform(upos_docs)
ngram_vocab = n_gram_vectorizer.get_feature_names_out()

print("Calculating top n-gram patterns.")
top_freq_per_class = top_freq_group(data["work"], upos_ngrams, ngram_vocab)

print("Saving results")
out_path = dat_path.joinpath("results/upos_patterns.csv")

res = pd.DataFrame(top_freq_per_class)
res.to_csv(out_path)

print("Done.")

print("Collecting UPOS n-grams")
n_gram_vectorizer = CountVectorizer(ngram_range=(4, 4))
upos_ngrams = n_gram_vectorizer.fit_transform(upos_docs)
ngram_vocab = n_gram_vectorizer.get_feature_names_out()

print("Calculating top n-gram patterns.")
top_freq_per_class = top_freq_group(data["work"], upos_ngrams, ngram_vocab)

print("Saving results")
out_path = dat_path.joinpath("results/upos_patterns_4.csv")

res = pd.DataFrame(top_freq_per_class)
res.to_csv(out_path)

print("Done.")
