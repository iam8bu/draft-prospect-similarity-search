"""Turning the merged career dataset into a standardized feature matrix for clustering/similarity search."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

RESULT_COLUMNS = [
    "av_per_year", "years_played", "G", "GS", "from_year_nfl", "to_year_nfl",
    "Draft Year", "Draft Team", "draft_age", "Round", "Pick", "nfl_team",
]
NON_FEATURE_COLUMNS = ["combine_season", "from_year", "to_year", "Clean_Name"]
COMBINE_COLUMNS = ["ht", "wt", "forty", "vertical", "broad_jump", "cone", "shuttle"]

SUCCESS_AV_THRESHOLD = 30
SKEW_THRESHOLD = 1.0
CONFERENCE_HASH_FEATURES = 128
COLLEGE_HASH_FEATURES = 256
PCA_COMPONENTS = 50
PCA_RANDOM_STATE = 42


def height_to_inches(height_str):
    """Convert a "6-0" feet-inches string into total inches."""
    try:
        feet, inches = height_str.split("-")
        return int(feet) * 12 + int(inches)
    except (AttributeError, ValueError):
        return None


def _hash_and_reduce_categoricals(standard_data: pd.DataFrame) -> pd.DataFrame:
    """Feature-hash conference/college_team, then PCA the combined hash space down to a fixed size."""
    conference_dict = standard_data[["conference"]].astype(str).to_dict(orient="records")
    hashed_conference = FeatureHasher(n_features=CONFERENCE_HASH_FEATURES, input_type="dict").transform(
        conference_dict
    ).toarray()

    college_dict = standard_data[["college_team"]].astype(str).to_dict(orient="records")
    hashed_college = FeatureHasher(n_features=COLLEGE_HASH_FEATURES, input_type="dict").transform(
        college_dict
    ).toarray()

    hashed_combined = np.hstack([hashed_college, hashed_conference])
    pca = PCA(n_components=PCA_COMPONENTS, random_state=PCA_RANDOM_STATE)
    college_pca = pca.fit_transform(hashed_combined)
    hashed_df = pd.DataFrame(college_pca, columns=[f"pca_feature_{i}" for i in range(PCA_COMPONENTS)])

    return pd.concat(
        [standard_data.drop(columns=["conference", "college_team"]).reset_index(drop=True), hashed_df.reset_index(drop=True)],
        axis=1,
    )


def build_standard_data(cfb_nfl_dataset: pd.DataFrame) -> pd.DataFrame:
    """Build the standardized, model-ready feature matrix used for clustering and similarity search.

    Drops NFL outcome columns (so results can't leak into the model), labels each QB with
    `nfl_success` (AV >= 30), unskews and 0-1 scales numeric columns, and hashes+PCA-reduces
    the conference/college categoricals.
    """
    standard_data = cfb_nfl_dataset.drop(columns=RESULT_COLUMNS)
    standard_data = standard_data.drop(columns=NON_FEATURE_COLUMNS)
    standard_data["ht"] = standard_data["ht"].apply(height_to_inches)

    numeric_columns = standard_data.select_dtypes(include="number").columns.tolist()
    numeric_columns.remove("athlete_id")
    numeric_columns.remove("AV")

    standard_data["nfl_success"] = (standard_data["AV"] >= SUCCESS_AV_THRESHOLD).astype(int)
    standard_data = standard_data.drop(columns="AV")

    skewed_columns = standard_data[numeric_columns].skew()
    high_skew_columns = skewed_columns[abs(skewed_columns) > SKEW_THRESHOLD].index.tolist()
    pt = PowerTransformer(method="yeo-johnson")
    standard_data.loc[:, high_skew_columns] = pt.fit_transform(standard_data[high_skew_columns])

    standard_data.loc[:, numeric_columns] = MinMaxScaler().fit_transform(standard_data[numeric_columns])

    standard_data = _hash_and_reduce_categoricals(standard_data)
    standard_data.columns = standard_data.columns.astype(str)
    standard_data = standard_data.astype("float64")

    for col in COMBINE_COLUMNS:
        standard_data[col] = standard_data[col].fillna(-1)

    return standard_data
