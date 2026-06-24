"""Streamlit demo: search for NFL draft QB prospects similar to a given college quarterback.

Run with: streamlit run app.py
"""

import os

# faiss and numba (a UMAP dependency, imported via qb_similarity.clustering) each bundle
# their own OpenMP runtime, which can segfault on macOS if both spin up worker threads in
# the same process. Set before any heavy imports.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import streamlit as st

from qb_similarity import cleaning, clustering, data_loading, features, similarity, viz

MIN_BLUE_CHIP_CLUSTER_SIZE = 20
COMPARISON_METRICS = ["completion_pct", "td_pct", "int_pct"]

st.set_page_config(page_title="QB Draft Prospect Similarity Search", layout="centered")


@st.cache_resource
def load_pipeline():
    results_df, cfb_df, combine_df = data_loading.load_raw_data()
    cfb_nfl_dataset = cleaning.build_cfb_nfl_dataset(results_df, cfb_df, combine_df)
    standard_data = features.build_standard_data(cfb_nfl_dataset)
    index = similarity.build_similarity_index(standard_data)
    return cfb_nfl_dataset, standard_data, index


@st.cache_resource
def find_blue_chip_cluster(standard_data: pd.DataFrame):
    """Spectral-cluster the standardized features and return the QBs in whichever
    sufficiently-large cluster has the highest historical NFL success rate."""
    cluster_data = standard_data.drop(columns=["athlete_id"] + features.COMBINE_COLUMNS)
    clustered = clustering.run_spectral_clustering(cluster_data, n_clusters=15, n_neighbors=50, random_state=0)
    clustered = pd.concat([standard_data["athlete_id"].reset_index(drop=True), clustered.reset_index(drop=True)], axis=1)

    cluster_stats = clustered.groupby("cluster_assignments")["nfl_success"].agg(["mean", "size"])
    eligible = cluster_stats[cluster_stats["size"] >= MIN_BLUE_CHIP_CLUSTER_SIZE]
    best_cluster = eligible["mean"].idxmax()

    blue_chip_ids = set(clustered.loc[clustered["cluster_assignments"] == best_cluster, "athlete_id"])
    return blue_chip_ids, eligible.loc[best_cluster, "mean"], int(eligible.loc[best_cluster, "size"])


st.title("College to Pro: QB Similarity Search")
st.write(
    "Find historical college quarterbacks whose college stats, draft profile, and "
    "school/conference most closely resemble a given prospect. Built on a flat-index "
    "FAISS similarity search over ~1,000 QBs who played between 2004 and 2020."
)

with st.spinner("Loading data and clustering players (first run only takes a few seconds)..."):
    cfb_nfl_dataset, standard_data, index = load_pipeline()
    blue_chip_ids, blue_chip_success_rate, blue_chip_size = find_blue_chip_cluster(standard_data)

with st.expander("Filter the player list"):
    all_conferences = sorted(
        {c.strip() for entry in cfb_nfl_dataset["conference"].dropna() for c in entry.split(",")}
    )
    selected_conferences = st.multiselect("Conference", all_conferences)

    round_display = cfb_nfl_dataset["Round"].fillna("Undrafted").astype(str)
    all_rounds = sorted(round_display.unique(), key=lambda r: (r == "Undrafted", r))
    selected_rounds = st.multiselect("Draft round", all_rounds)

filtered_dataset = cfb_nfl_dataset.copy()
filtered_dataset["round_display"] = round_display
if selected_conferences:
    filtered_dataset = filtered_dataset[
        filtered_dataset["conference"].fillna("").apply(lambda c: any(conf in c for conf in selected_conferences))
    ]
if selected_rounds:
    filtered_dataset = filtered_dataset[filtered_dataset["round_display"].isin(selected_rounds)]

name_options = sorted(filtered_dataset["Clean_Name"].dropna().unique())
if not name_options:
    st.warning("No players match these filters — try removing one.")
    st.stop()

default_name = "trevor lawrence" if "trevor lawrence" in name_options else name_options[0]
selected_name = st.selectbox(
    "Choose a quarterback", name_options, index=name_options.index(default_name), format_func=str.title
)
k = st.slider("Number of similar players to return", min_value=2, max_value=15, value=5)

results = similarity.find_similar_players(selected_name, cfb_nfl_dataset, standard_data, index, k=k)

if results.iloc[0]["athlete_id"] in blue_chip_ids:
    st.success(
        f"🏆 This QB falls in a {blue_chip_size}-player cluster with a historical "
        f"~{blue_chip_success_rate:.0%} NFL success rate."
    )

display_results = results.copy()
display_results["Round"] = display_results["Round"].fillna("Undrafted").astype(str)
display_results["Similarity"] = (100 / (1 + display_results["distance"])).round(1).astype(str) + "%"
display_results["Clean_Name"] = display_results["Clean_Name"].str.title()
display_results = display_results.rename(columns={"Clean_Name": "Name"})

st.dataframe(
    display_results[["Name", "AV", "Round", "Similarity"]],
    width="stretch",
    hide_index=True,
)
st.caption("Similarity is derived from L2 distance in the standardized feature space — 100% is the searched player themselves.")

if len(results) > 1:
    top_comp_name = results.iloc[1]["Clean_Name"]
    st.subheader(f"{selected_name.title()} vs. closest comp: {top_comp_name.title()}")
    fig = viz.plot_qb_comparison(cfb_nfl_dataset, selected_name, top_comp_name, COMPARISON_METRICS)
    st.pyplot(fig)

st.divider()
st.caption(
    "Data: cfbfastR (college stats, 2004-2020), nflreadr (combine data), and Pro Football "
    "Reference StatHead (NFL draft/results data). See the analysis notebook in `notebooks/` "
    "for the full methodology."
)
