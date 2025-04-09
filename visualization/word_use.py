from pathlib import Path
import pandas as pd
import plotly.express as px


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


data = pd.read_csv("results/word_use.csv")
#data = pd.read_csv("/work/gospel-ancient-greek/gospel-ancient-greek/results/word_use.csv")
md = fetch_metadata(SHEET_URL)
#data["text"] = data["text_name"].map(lambda s: s.split(" - ")[1])
data["text"] = data["text_name"]
#data["work"] = data["work_id"].map(lambda id: find_work(id, md))
data["work"] = data["work_id"]
data["group"] = (
    "<b>"
    + data["work"]
    + "</b>"
    + " <br> top c-tf-idf: <i>"
    + data["top_ctf-idf"]
    + "</i> <br> top freq: <i>"
    + data["top_frequency_in_work"]
)

out_path = Path("docs/_static/word_use.html")
out_path.parent.mkdir(exist_ok=True, parents=True)

fig = px.scatter(
    data,
    x="x_umap",
    y="y_umap",
    color="group",
    #text="text",
    hover_data={
        "text_name": True,
        "top_tf-idf": True,
        "top_frequency": True,
        "x_umap": False,
        "y_umap": False,
        "group": False,
        "text": True,
    },
    #hover_name="text_name",
    template="plotly_white",
    height=1200,
    width=1000,
)

# Force the mode to markers so that no text is rendered directly.
fig.update_traces(mode="markers", marker=dict(size=14, line=dict(width=2, color="black")))
fig.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

fig.write_html(out_path)
