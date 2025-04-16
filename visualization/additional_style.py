from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

dat_path = Path("/work/gospel-ancient-greek/gospel-ancient-greek/data/results")

data = pd.read_csv(dat_path.joinpath("style_noun_adj_verb.csv"))

out_path = Path("docs/_static/vocabulary_richness_noun_adj_verb.html")
out_path.parent.mkdir(exist_ok=True, parents=True)
fig = make_subplots(
    rows=5, cols=1, subplot_titles=["Overall TTR", "MATTR-500", "MATTR-1000"]
)
unique_works = data["work"].unique()
colors = px.colors.qualitative.Pastel
work_to_color = dict(zip(unique_works, colors))
for i_feature, feature in enumerate(["ttr", "mattr_500", "mattr_1000"]):
    row = i_feature + 1
    for work, color in work_to_color.items():
        subset = data[data["work"] == work]
        trace = go.Box(
            x=subset[feature],
            name=work,
            legendgroup=work,
            showlegend=row == 1,
            boxpoints="all",
            text=subset["text_name"],
            hovertemplate="<b>%{text}",
            marker=dict(color=color),
        )
        fig.add_trace(trace, row=row, col=1)
fig.update_layout(template="plotly_white", width=1200, height=1000)
fig.update_yaxes(visible=False)
fig.write_html(out_path)

data = pd.read_csv(dat_path.joinpath("style_adv_intj_adp_cconj_sconj_det_part_pron.csv"))


out_path = Path("docs/_static/vocabulary_richness_adv_intj_adp_cconj_sconj_det_part_pron.html")
out_path.parent.mkdir(exist_ok=True, parents=True)
fig = make_subplots(
    rows=5, cols=1, subplot_titles=["Overall TTR", "MATTR-500", "MATTR-1000"]
)
unique_works = data["work"].unique()
colors = px.colors.qualitative.Pastel
work_to_color = dict(zip(unique_works, colors))
for i_feature, feature in enumerate(["ttr", "mattr_500", "mattr_1000"]):
    row = i_feature + 1
    for work, color in work_to_color.items():
        subset = data[data["work"] == work]
        trace = go.Box(
            x=subset[feature],
            name=work,
            legendgroup=work,
            showlegend=row == 1,
            boxpoints="all",
            text=subset["text_name"],
            hovertemplate="<b>%{text}",
            marker=dict(color=color),
        )
        fig.add_trace(trace, row=row, col=1)
fig.update_layout(template="plotly_white", width=1200, height=1000)
fig.update_yaxes(visible=False)
fig.write_html(out_path)
