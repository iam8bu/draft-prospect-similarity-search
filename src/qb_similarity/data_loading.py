"""Loading the three raw source files into dataframes."""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def load_raw_data(data_dir: Path | str = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the NFL results, college stats, and combine datasets.

    Returns (results_df, cfb_df, combine_df).
    """
    data_dir = Path(data_dir)
    results_df = pd.read_excel(data_dir / "NFL Results Data.xlsx")
    cfb_df = pd.read_csv(data_dir / "cfb_qb_stats_full_2004_2020.csv")
    combine_df = pd.read_csv(data_dir / "combine_data.csv")
    return results_df, cfb_df, combine_df
