# College to Pro: NFL Draft QB Prospect Similarity Search

A similarity-search and clustering analysis of NFL quarterback draft prospects. Given a college quarterback, it finds the historical QBs whose college stats, draft profile, and school/conference most closely resemble them — and surfaces the traits that actually separate NFL hits from busts.

According to [The Athletic](https://www.nytimes.com/athletic/5377539/2024/03/29/odds-a-top-10-qb-busts-scoop-city/), 41% of quarterbacks drafted in the first round can be classified as a reach or a bust. Quarterback is the most expensive, highest-stakes position to evaluate in professional sports — this project builds a data-driven "comp finder" to help separate signal from scouting-report noise.

This project began as a final project for a Harvard Extension School data mining course; this repo is a restructured, portfolio-ready version of that original analysis. ChatGPT was used at points during development.

## What's here

- **`src/qb_similarity/`** — a reusable Python package with the full pipeline: data loading, cleaning/merging, feature engineering, clustering, and FAISS similarity search. Both the notebook and the app import from here, so there's a single source of truth for the logic.
- **`notebooks/college_to_pro_analysis.ipynb`** — the full analytical writeup: EDA, feature engineering, three clustering approaches, and the similarity search, with commentary throughout.
- **`app.py`** — a small Streamlit app that wraps the similarity search in a "pick a QB, see his comps" interface.
- **`archive/`** — historical artifacts kept for reference but superseded by the above: the original project proposal, the original (unrefactored) final-analysis notebook, and a StatHead export of college passing stats (`unused_college_data_stathead_export.xlsx`) that was pulled early on but never actually used in the analysis — `cfb_qb_stats_full_2004_2020.csv` became the college data source instead.

## Data

Three datasets, merged on cleaned player name:

1. **College stats** ([cfbfastR](https://cfbfastr.sportsdataverse.org)): every FBS quarterback who recorded a stat from 2004–2020.
2. **NFL combine data** ([nflreadr](https://nflreadr.nflverse.com/reference/index.html)): combine measurables from 2000–2025 (filtered to QBs from the relevant draft window). Too sparse to use in clustering, so it's dropped there — it's still part of the similarity-search feature set.
3. **NFL draft/career results** ([Pro Football Reference](https://www.pro-football-reference.com) StatHead): every QB drafted 2005–2021, with draft position and [Approximate Value](https://www.pro-football-reference.com/about/approximate_value.htm) (AV) as a proxy for career quality.

Raw files live in `data/`. See the notebook's appendix for a full column-by-column explanation.

## Methodology

1. **Clean & merge** (`qb_similarity.cleaning`) — aggregate season-level college stats into one row per career, compute rate stats and passer rating, filter to QBs with 100+ career attempts, and merge in NFL outcomes by name.
2. **Feature engineering** (`qb_similarity.features`) — label each QB `nfl_success` (career AV ≥ 30), correct skew with a Yeo-Johnson transform, scale to 0–1, and feature-hash + PCA-reduce the conference/college categoricals (384 hashed columns → 50 components) so they don't drown out the numeric stats.
3. **Clustering** (`qb_similarity.clustering`) — agglomerative, spectral, and OPTICS clustering, comparing silhouette score and cluster diameter to pick a cluster count.
4. **Similarity search** (`qb_similarity.similarity`) — an exact flat-index FAISS search (`IndexFlatL2`) over the standardized features. The dataset (~1,000 QBs) is small enough that an exact index is both fast and 100% recall — no need for an approximate index.

## Key findings

- Spectral clustering (15 clusters) found one cluster of 34 QBs that was **~91% NFL-successful** — and it included QBs who slipped well past the first round (Russell Wilson, Derek Carr, Dak Prescott, Case Keenum, and others), suggesting real value could be found by spotting this profile early.
- For that cluster, **passing volume mattered more than efficiency** — and interception rate was surprisingly *not* among the most significant differences from the rest of the dataset.
- Feature hashing on conference/school works as intended: searching "Mac Jones" (Alabama, SEC) returns other Alabama and SEC quarterbacks.
- Box-score stats alone can mislead — Josh Allen (NFL MVP) and Josh Rosen (a notorious bust) compare unfavorably for Allen on completion % and interception rate, despite the gap in NFL outcomes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the analysis notebook:

```bash
jupyter notebook notebooks/college_to_pro_analysis.ipynb
```

Run the interactive demo:

```bash
streamlit run app.py
```

> **Note (macOS):** `faiss-cpu` and `umap-learn` (via numba) each bundle their own OpenMP runtime, which can segfault if both spin up worker threads in the same process. Both the notebook and `app.py` set `OMP_NUM_THREADS=1` and related env vars at import time to avoid this — no extra setup needed, but if you hit a kernel crash in a different entry point, set `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` in your shell before launching.

## Limitations & future work

- Height/weight and advanced college statistics would likely improve the model, but no public dataset had enough coverage of them for this QB population.
- The dataset is small even though it's the full population of relevant QBs — results for cluster 12 are encouraging but it's only 34 players, and OPTICS in particular struggled to find structure given so few positive examples.
- Next steps: capture college-team-level features (scheme, competition quality) directly rather than only through hashed conference/school; analyze combine-invited athletes as a separate cohort; and explore how much NFL outcomes depend on situation/team quality rather than the QB's own traits.
