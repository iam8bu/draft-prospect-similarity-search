"""Cleaning and merging the three raw QB datasets into one career-level dataset."""

import string

import pandas as pd

COLLEGE_COLUMNS_TO_DROP = [
    "year", "passing_pct", "passing_ypa", "rushing_ypc",
    "receiving_rec", "receiving_yds", "receiving_td", "receiving_ypr",
    "receiving_long", "fumbles_rec", "defensive_solo", "defensive_tot", "defensive_tfl", "defensive_sacks",
    "defensive_qb_hur", "interceptions_int", "interceptions_yds",
    "interceptions_avg", "interceptions_td", "defensive_pd", "defensive_td",
    "kicking_fgm", "kicking_fga", "kicking_pct", "kicking_xpa",
    "kicking_xpm", "kicking_pts", "kicking_long", "kick_returns_no",
    "kick_returns_yds", "kick_returns_avg", "kick_returns_td",
    "kick_returns_long", "punting_no", "punting_yds", "punting_ypp",
    "punting_long", "punting_in_20", "punting_tb", "punt_returns_no",
    "punt_returns_yds", "punt_returns_avg", "punt_returns_td",
    "punt_returns_long", "position", "stat_category",
]

NAME_SUFFIXES = (" jr", " sr", " iii", " ii", " iv")


def clean_name(name: str) -> str:
    """Lower-case, strip punctuation/suffixes so names match across datasets."""
    if pd.isna(name):
        return name
    name = name.lower().strip()
    for punct in string.punctuation:
        name = name.replace(punct, "")
    for suffix in NAME_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def clean_nfl_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Add years-played/AV-per-year and drop redundant Pro Football Reference columns."""
    results_df = results_df.copy()
    results_df["years_played"] = results_df["To"] - results_df["From"] + 1
    results_df["av_per_year"] = results_df["AV"] / results_df["years_played"]
    return results_df.drop(["Rk", "AV.1", "Pos"], axis=1)


def clean_college_stats(cfb_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate season-level college stats into one row per QB career, with rate stats and passer rating."""
    cfb_df = cfb_df.drop(COLLEGE_COLUMNS_TO_DROP, axis=1)

    college_seasons = cfb_df.groupby("athlete_id", as_index=False).agg(
        from_year=("season", "min"), to_year=("season", "max")
    )
    team_list = (
        cfb_df.groupby("athlete_id")["team"]
        .agg(lambda x: ", ".join(sorted(set(x))))
        .reset_index()
    )
    conference_list = (
        cfb_df.groupby("athlete_id")["conference"]
        .agg(lambda x: ", ".join(sorted(set(x))))
        .reset_index()
    )

    career_stats = cfb_df.groupby("athlete_id", as_index=False).agg(
        {
            "player": "last",
            "passing_completions": "sum",
            "passing_att": "sum",
            "passing_yds": "sum",
            "passing_td": "sum",
            "passing_int": "sum",
            "rushing_car": "sum",
            "rushing_yds": "sum",
            "rushing_td": "sum",
            "rushing_long": "max",
            "fumbles_fum": "sum",
            "fumbles_lost": "sum",
        }
    )
    career_stats = career_stats.merge(college_seasons, on="athlete_id", how="left")
    career_stats = career_stats.merge(team_list, on="athlete_id", how="left")
    career_stats = career_stats.merge(conference_list, on="athlete_id", how="left")

    career_stats["completion_pct"] = career_stats["passing_completions"] / career_stats["passing_att"]
    career_stats["ypa"] = career_stats["passing_yds"] / career_stats["passing_att"]
    career_stats["td_pct"] = career_stats["passing_td"] / career_stats["passing_att"]
    career_stats["int_pct"] = career_stats["passing_int"] / career_stats["passing_att"]
    career_stats["ypr"] = career_stats["rushing_yds"] / career_stats["rushing_car"]

    # Standard NFL/NCAA passer rating formula, components capped to [0, 2.375].
    a = ((career_stats["completion_pct"] - 0.3) * 5).clip(0, 2.375)
    b = ((career_stats["ypa"] - 3) * 0.25).clip(0, 2.375)
    c = (career_stats["td_pct"] * 20).clip(0, 2.375)
    d = (2.375 - (career_stats["int_pct"] * 25)).clip(0, 2.375)
    career_stats["passer_rating"] = ((a + b + c + d) / 6) * 100

    return career_stats


def clean_combine_data(combine_df: pd.DataFrame) -> pd.DataFrame:
    """Filter combine data down to QBs from the relevant draft window."""
    combine_df = combine_df[combine_df["season"].between(2005, 2021)].reset_index(drop=True)
    combine_df = combine_df[combine_df["pos"] == "QB"].reset_index(drop=True)
    return combine_df.drop(["pfr_id", "cfb_id", "pos"], axis=1)


def merge_datasets(
    cfb_merged: pd.DataFrame, results_df: pd.DataFrame, combine_df: pd.DataFrame
) -> pd.DataFrame:
    """Name-match and merge the three cleaned datasets into one career-level QB table."""
    results_df = results_df.copy()
    cfb_merged = cfb_merged.copy()
    combine_df = combine_df.copy()

    results_df["Clean_Name"] = results_df["Player"].apply(clean_name)
    results_df = results_df.drop(["Player"], axis=1)

    cfb_merged["Clean_Name"] = cfb_merged["player"].apply(clean_name)
    cfb_merged = cfb_merged.drop(["player"], axis=1)

    combine_df["Clean_Name"] = combine_df["player_name"].apply(clean_name)
    combine_df = combine_df.drop(["player_name"], axis=1)

    cfb_nfl_dataset = cfb_merged.merge(results_df, on="Clean_Name", how="left", suffixes=("", "_result"))
    cfb_nfl_dataset = cfb_nfl_dataset.merge(combine_df, on="Clean_Name", how="left", suffixes=("", "_combine"))

    # College QBs with no NFL match did not play in the league.
    cfb_nfl_dataset["years_played"] = cfb_nfl_dataset["years_played"].fillna(0)
    cfb_nfl_dataset["av_per_year"] = cfb_nfl_dataset["av_per_year"].fillna(0)
    cfb_nfl_dataset["AV"] = cfb_nfl_dataset["AV"].fillna(0)
    cfb_nfl_dataset["GS"] = cfb_nfl_dataset["GS"].fillna(0)
    cfb_nfl_dataset["G"] = cfb_nfl_dataset["G"].fillna(0)

    # Drop QBs with too few college attempts to be meaningfully comparable.
    cfb_nfl_dataset = cfb_nfl_dataset[cfb_nfl_dataset["passing_att"] >= 100]

    cfb_nfl_dataset = cfb_nfl_dataset.drop(
        ["fumbles_fum", "fumbles_lost", "bench", "College", "draft_year", "draft_team", "draft_round", "draft_ovr", "school"],
        axis=1,
    )
    cfb_nfl_dataset = cfb_nfl_dataset.rename(
        columns={
            "team": "college_team",
            "From": "from_year_nfl",
            "To": "to_year_nfl",
            "Team": "nfl_team",
            "season": "combine_season",
            "Age": "draft_age",
        }
    )

    # Drop ambiguous duplicate names (can't tell if it's the same player or two players sharing a name).
    cfb_nfl_dataset = cfb_nfl_dataset[~cfb_nfl_dataset["Clean_Name"].duplicated(keep=False)]

    return cfb_nfl_dataset


def build_cfb_nfl_dataset(
    results_df: pd.DataFrame, cfb_df: pd.DataFrame, combine_df: pd.DataFrame
) -> pd.DataFrame:
    """Run the full cleaning + merge pipeline on the three raw dataframes."""
    cfb_merged = clean_college_stats(cfb_df)
    results_df = clean_nfl_results(results_df)
    combine_df = clean_combine_data(combine_df)
    return merge_datasets(cfb_merged, results_df, combine_df)
