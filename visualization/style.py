from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


data = pd.read_csv("results/stylistic_features.csv")
md = fetch_metadata(SHEET_URL)
#data["text"] = data["text_name"].map(lambda s: s.split(" - ")[1])
data["text"] = data["text_name"]
#data["work"] = data["work_id"].map(lambda id: find_work(id, md))
data["work"] = data["work_id"]

out_path = Path("docs/_static/vocabulary_richness.html")
out_path.parent.mkdir(exist_ok=True, parents=True)
fig = make_subplots(
    rows=3, cols=1, subplot_titles=["Overall TTR", "MATTR-10", "MATTR-50"]
)
unique_works = data["work"].unique()
colors = px.colors.qualitative.Pastel
work_to_color = dict(zip(unique_works, colors))
for i_feature, feature in enumerate(["ttr", "mattr_10", "mattr_50"]):
    row = i_feature + 1
    for work, color in work_to_color.items():
        subset = data[data["work"] == work]
        trace = go.Box(
            x=subset[feature],
            name=work,
            legendgroup=work,
            showlegend=row == 1,
            boxpoints="all",
            text=subset["text"],
            hovertemplate="<b>%{text}",
            marker=dict(color=color),
        )
        fig.add_trace(trace, row=row, col=1)
fig.update_layout(template="plotly_white", width=1200, height=1000)
fig.update_yaxes(visible=False)
fig.write_html(out_path)

fig = px.scatter_matrix(
    data,
    dimensions=["length", "mean_sentence_length", "mean_token_length", "n_sentences"],
    hover_name="text_name",
    color="work",
)
out_path = Path("docs/_static/length_scatter_matrix.html")
fig.write_html(out_path)
