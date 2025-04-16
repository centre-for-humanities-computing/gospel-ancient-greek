from ast import literal_eval
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    return md["work"].iloc[0]


def wrap_text(text: str) -> str:
    if len(text) > 10:
        text = "<br>".join(text.split())
    return "<b>" + text


print("Producing UPOS frequency visualizations in 3D.")
dat_path = Path("/work/gospel-ancient-greek/gospel-ancient-greek/data")

data = pd.read_csv(dat_path.joinpath("results/upos_tags.csv"),index_col=0)

# md = fetch_metadata(SHEET_URL)
# data["work_name"] = data["work_id"].map(partial(find_work, md=md))
data = data.set_index(["work", "text_name"])
freq = data.to_numpy()
rel_freq = pd.DataFrame(
    (freq.T / freq.sum(axis=1)).T, columns=data.columns, index=data.index
)
# Meaning words
fig = px.scatter_3d(
    rel_freq.reset_index(),
    x="noun", y = "adj", z = "verb",
    hover_name="text_name",
    color="work",
)
fig.update_layout(legend=dict(
    y=-0.3,
    xanchor="left",
    x=0
))
out_path = Path("docs/_static/upos_scatter_matrix_3d.html")
fig.write_html(out_path)