from ast import literal_eval
from pathlib import Path

import pandas as pd
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


data = pd.read_csv("results/phrases.csv", index_col=0)
md = fetch_metadata(SHEET_URL)
data.columns = [find_work(work_id, md) for work_id in data.columns]
z = data.applymap(lambda elem: literal_eval(elem)[1])
data = data.applymap(lambda elem: literal_eval(elem)[0])
data = data.applymap(wrap_text)

trace = go.Heatmap(
    z=z,
    text=data,
    texttemplate="%{text}",
    textfont=dict(size=14),
    x=data.columns,
    y=data.index,
    colorbar=dict(title="Number of occurrences"),
)
fig = go.Figure(
    data=trace,
)
fig = fig.update_layout(
    width=1000,
    height=1400,
)
fig = fig.update_yaxes(autorange="reversed")

out_path = Path("docs/_static/phrases.html")
out_path.parent.mkdir(exist_ok=True, parents=True)

fig.write_html(out_path)
