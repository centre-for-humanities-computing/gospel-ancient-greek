from ast import literal_eval
from pathlib import Path
import shutil
import os
import pandas as pd

SHEET_URL = "https://docs.google.com/spreadsheets/d/15WIzk2aV3vCQLnDihdnNCLxMbDmJZiZKmuiM_xRKbwk/edit#gid=282554525"


def fetch_metadata(url: str) -> pd.DataFrame:
    """Loads metadata from Google Sheets url."""
    url = url.replace("/edit#gid=", "/export?format=csv&gid=")
    metadata = pd.read_csv(url)
    metadata.skal_fjernes = metadata.skal_fjernes == "True"
    return metadata


def find_work(work_id: str, md: pd.DataFrame) -> str:
    md = md.dropna(subset=["work", "document_id"])
    md = md[md["document_id"].str.contains(work_id)]
    #id_plus_name = md["document_id"].iloc[0] + md["work"].iloc[0]
    return md["document_id"].iloc[0], md["author"].iloc[0], md["work"].iloc[0]


def wrap_text(text: str) -> str:
    if len(text) > 10:
        text = "<br>".join(text.split())
    return "<b>" + text


# remove perseus_tlg0007.tlg024.perseus-grc from list

data = [
    "perseus_tlg0085.tlg005.perseus-grc2",
    "perseus_tlg0085.tlg007.perseus-grc2",
    "perseus_tlg0085.tlg006.perseus-grc2",
    "perseus_tlg0085.tlg002.perseus-grc2",
    "perseus_tlg0085.tlg003.perseus-grc2",
    "perseus_tlg0085.tlg004.perseus-grc2",
    "perseus_tlg0085.tlg001.perseus-grc2",
    "perseus_tlg0006.tlg007.perseus-grc2",
    "perseus_tlg0006.tlg002.perseus-grc2",
    "perseus_tlg0006.tlg006.perseus-grc2",
    "perseus_tlg0006.tlg017.perseus-grc2",
    "perseus_tlg0006.tlg001.perseus-grc2",
    "perseus_tlg0006.tlg012.perseus-grc2",
    "perseus_tlg0006.tlg014.perseus-grc2",
    "perseus_tlg0006.tlg004.perseus-grc2",
    "perseus_tlg0006.tlg009.perseus-grc2",
    "perseus_tlg0006.tlg005.perseus-grc2",
    "perseus_tlg0006.tlg010.perseus-grc2",
    "perseus_tlg0006.tlg018.perseus-grc2",
    "perseus_tlg0006.tlg013.perseus-grc2",
    "perseus_tlg0006.tlg003.perseus-grc2",
    "perseus_tlg0006.tlg016.perseus-grc2",
    "perseus_tlg0006.tlg019.perseus-grc2",
    "perseus_tlg0006.tlg015.perseus-grc2",
    "perseus_tlg0006.tlg008.perseus-grc2",
    "perseus_tlg0006.tlg011.perseus-grc2",
    "perseus_tlg0011.tlg003.perseus-grc2",
    "perseus_tlg0011.tlg002.perseus-grc2",
    "perseus_tlg0011.tlg005.perseus-grc2",
    "perseus_tlg0011.tlg008.perseus-grc2",
    "perseus_tlg0011.tlg007.perseus-grc2",
    "perseus_tlg0011.tlg004.perseus-grc2",
    "perseus_tlg0011.tlg006.perseus-grc2",
    "perseus_tlg0011.tlg001.perseus-grc2"
]

md = fetch_metadata(SHEET_URL)
data = [find_work(work_id, md) for work_id in data]

# data contains tuples of (document_id, work) divide into two columsn
data = {"document_id": [d[0] for d in data], "author": [d[1] for d in data],  "work": [d[2] for d in data]}

# make data into pandas
data = pd.DataFrame(data)

# make column that combines author and work with " - " between
data["author_work"] = data["author"] + " - " + data["work"]
# data["author_work"] can be max 40 characters
data["author_work"] = data["author_work"].str[:40] + ".txt"

# make () in column author_work to _
data["author_work"] = data["author_work"].str.replace("(", "_")
data["author_work"] = data["author_work"].str.replace(")", "_")

# save to csv
data.to_csv("results/works.csv")

# Specify the source and destination directories
src_dir = '/Users/au619572/Downloads/computing-antiquity/dat/greek/exported_data/with_stopwords'
dst_dir = '/Users/au619572/Documents/git_repos/gospel-ancient-greek/data/raw_single_file/greek_tragedies'

# Assume the file names are in a column named 'filename'
for file_name in data['author_work']:
    src_file = os.path.join(src_dir, file_name)
    dst_file = os.path.join(dst_dir, file_name)

    # Copy the file if it exists, otherwise return a warning
    if os.path.exists(src_file):
        shutil.copy(src_file, dst_file)
    else:
        print(f"Warning: File '{src_file}' does not exist.")