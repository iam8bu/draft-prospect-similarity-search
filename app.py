"""Streamlit demo: search for NFL draft QB prospects similar to a given college quarterback.

Run with: streamlit run app.py
"""

import os

# faiss and scikit-learn each bundle their own OpenMP runtime, which can conflict on macOS
# when both spin up worker threads in the same process. Set before any heavy imports.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from qb_similarity import cleaning, data_loading, features, similarity

st.set_page_config(page_title="QB Draft Prospect Similarity Search", layout="centered")


@st.cache_resource
def load_pipeline():
    results_df, cfb_df, combine_df = data_loading.load_raw_data()
    cfb_nfl_dataset = cleaning.build_cfb_nfl_dataset(results_df, cfb_df, combine_df)
    standard_data = features.build_standard_data(cfb_nfl_dataset)
    index = similarity.build_similarity_index(standard_data)
    return cfb_nfl_dataset, standard_data, index


st.title("College to Pro: QB Similarity Search")
st.write(
    "Find historical college quarterbacks whose college stats, draft profile, and "
    "school/conference most closely resemble a given prospect. Built on a flat-index "
    "FAISS similarity search over ~1,000 QBs who played between 2004 and 2020."
)

cfb_nfl_dataset, standard_data, index = load_pipeline()
name_options = sorted(cfb_nfl_dataset["Clean_Name"].dropna().unique())

selected_name = st.selectbox("Choose a quarterback", name_options, index=name_options.index("trevor lawrence") if "trevor lawrence" in name_options else 0)
k = st.slider("Number of similar players to return", min_value=1, max_value=15, value=5)

if st.button("Find similar players", type="primary"):
    results = similarity.find_similar_players(selected_name, cfb_nfl_dataset, standard_data, index, k=k)
    results["Round"] = results["Round"].fillna("Undrafted").astype(str)
    st.dataframe(results, width="stretch", hide_index=True)
    st.caption("`distance` is L2 distance in the standardized feature space — 0 is an exact match (the player themselves).")

st.divider()
st.caption(
    "Data: cfbfastR (college stats, 2004-2020), nflreadr (combine data), and Pro Football "
    "Reference StatHead (NFL draft/results data). See the analysis notebook in `notebooks/` "
    "for the full methodology."
)
