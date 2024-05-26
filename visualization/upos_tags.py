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


print("Producing Patterns heatmap (2-4).")
data = pd.read_csv("results/upos_patterns.csv", index_col=0)
md = fetch_metadata(SHEET_URL)
#data.columns = [find_work(work_id, md) for work_id in data.columns]
rel_freq = data.applymap(lambda elem: literal_eval(elem)[2])
counts = data.applymap(lambda elem: literal_eval(elem)[1])
data = data.applymap(lambda elem: literal_eval(elem)[0])
data = data.applymap(wrap_text)
data = data + "<br> [" + counts.applymap(str) + "]"
trace = go.Heatmap(
    z=rel_freq,
    text=data,
    texttemplate="%{text}",
    textfont=dict(size=14),
    x=data.columns,
    y=data.index,
    colorbar=dict(title="Relative Frequency"),
)
fig = go.Figure(
    data=trace,
)
fig = fig.update_layout(
    width=1000,
    height=1400,
)
fig = fig.update_yaxes(autorange="reversed")
out_path = Path("docs/_static/upos_patterns.html")
out_path.parent.mkdir(exist_ok=True, parents=True)
fig.write_html(out_path)

print("Producing Patterns heatmap (4).")
data = pd.read_csv("results/upos_patterns_4.csv", index_col=0)
md = fetch_metadata(SHEET_URL)
#data.columns = [find_work(work_id, md) for work_id in data.columns]
rel_freq = data.applymap(lambda elem: literal_eval(elem)[2])
counts = data.applymap(lambda elem: literal_eval(elem)[1])
data = data.applymap(lambda elem: literal_eval(elem)[0])
data = data.applymap(wrap_text)
data = data + "<br> [" + counts.applymap(str) + "]"
trace = go.Heatmap(
    z=rel_freq,
    text=data,
    texttemplate="%{text}",
    textfont=dict(size=14),
    x=data.columns,
    y=data.index,
    colorbar=dict(title="Relative Frequency"),
)
fig = go.Figure(
    data=trace,
)
fig = fig.update_layout(
    width=1000,
    height=1400,
)
fig = fig.update_yaxes(autorange="reversed")
out_path = Path("docs/_static/upos_patterns_4.html")
out_path.parent.mkdir(exist_ok=True, parents=True)
fig.write_html(out_path)

print("Producing UPOS frequency visualizations.")
data = pd.read_csv("results/upos_tags.csv", index_col=0)
#data["work_name"] = data["work_id"].map(partial(find_work, md=md))
data["work_name"] = data["work_id"]
data = data.set_index(["work_name", "text_name"]).drop(columns=["work_id"])
freq = data.to_numpy()
rel_freq = pd.DataFrame(
    (freq.T / freq.sum(axis=1)).T, columns=data.columns, index=data.index
)
# Meaning words
fig = px.scatter_matrix(
    rel_freq.reset_index(),
    dimensions=["noun", "adj", "verb", "aux"],
    hover_name="text_name",
    color="work_name",
)
fig.update_layout(legend=dict(
    y=-0.3,
    xanchor="left",
    x=0
))
out_path = Path("docs/_static/upos_scatter_matrix.html")
fig.write_html(out_path)

# Function words
fig = px.scatter_matrix(
    rel_freq.reset_index(),
    dimensions=["adp", "adv", "cconj", "det", "part", "sconj"],
    # dimensions=set(rel_freq.columns) - set(["noun", "adj", "verb", "propn", "num", "aux"]),
    hover_name="text_name",
    color="work_name",
)
fig.update_layout(legend=dict(
    y=-0.3,
    xanchor="left",
    x=0
))
out_path = Path("docs/_static/upos_scatter_matrix_function.html")
fig.write_html(out_path)

print("Producing wave plot")
unique_works = rel_freq.reset_index()["work_name"].unique()
colors = px.colors.qualitative.Safe
work_to_color = dict(zip(unique_works, colors))
tag_order = rel_freq.columns[np.argsort(-rel_freq.sum(axis=0))]
rel_freq = rel_freq[tag_order]
rel_freq = rel_freq.sort_index(level="work_name")
fig = go.Figure()
legendgroups = set()
for (work_name, text_name), data in rel_freq.iterrows():
    fig.add_trace(
        go.Scatter(
            name=work_name if work_name not in legendgroups else "",
            legendgroup=work_name,
            marker=dict(color=work_to_color[work_name]),
            x=tag_order,
            y=data,
            mode="lines",
            opacity=0.8,
            showlegend=work_name not in legendgroups,
        )
    )
    legendgroups |= set([work_name])
fig.add_trace(
    go.Scatter(
        name="Average across all works",
        marker=dict(color="black"),
        line=dict(width=5),
        x=tag_order,
        y=np.mean(rel_freq, axis = 0),
        mode="lines",
        opacity=1,
    )
)

fig.update_layout(legend=dict(
    y=-0.4,
    xanchor="left",
    x=0
))
out_path = Path("docs/_static/upos_wave_plot.html")
fig.write_html(out_path)
