"""Flat-index FAISS similarity search over the standardized QB feature matrix."""

import faiss
import pandas as pd

INDEX_EXCLUDE_COLUMNS = ["athlete_id", "nfl_success"]


def build_similarity_index(standard_data: pd.DataFrame) -> faiss.IndexFlatL2:
    """Build an L2 flat index over the standardized feature columns.

    A flat index is exact (100% recall) and fast enough here because the dataset
    is only a few hundred QBs.
    """
    feature_data = standard_data.drop(columns=INDEX_EXCLUDE_COLUMNS, errors="ignore").values.astype("float32")
    index = faiss.IndexFlatL2(feature_data.shape[1])
    index.add(feature_data)
    return index


def find_similar_players(
    name: str, cfb_nfl_dataset: pd.DataFrame, standard_data: pd.DataFrame, index: faiss.IndexFlatL2, k: int = 5
) -> pd.DataFrame:
    """Look up a QB by (partial) cleaned name and return their `k` nearest neighbors.

    Returns athlete_id, name, career AV, draft round, and distance for each neighbor
    (the query player itself is typically the closest match, at distance 0).
    """
    matches = cfb_nfl_dataset[cfb_nfl_dataset["Clean_Name"].str.contains(name)]
    if matches.empty:
        raise ValueError(f"No QB found matching '{name}'")

    player_id = matches["athlete_id"].values[0]
    player_row = standard_data[standard_data["athlete_id"] == player_id]
    player_vector = player_row.drop(columns=INDEX_EXCLUDE_COLUMNS, errors="ignore").values.astype("float32").reshape(1, -1)

    distances, indices = index.search(player_vector, k=k)
    similar_players = standard_data.iloc[indices[0]].copy()
    similar_players["distance"] = distances[0]

    result = similar_players.merge(
        cfb_nfl_dataset[["athlete_id", "Clean_Name", "AV", "Round"]], on="athlete_id", how="left"
    )
    return result[["athlete_id", "Clean_Name", "AV", "Round", "distance"]]
