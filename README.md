# Beyond Simple Popularity — thesis analysis companion

Supporting code and inputs for *Beyond Simple Popularity: Category-Aware Long Tail Analysis and Item Exposure in Recommender Systems*.

## Setup

From the repository root, use Python 3.10 or later:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p Figures/Results
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Regenerate figures

| Command | Input | Output under `Figures/Results/` |
| --- | --- | --- |
| `python Code/plot_readable_product_examples.py` | Product records and grouped display data in `Results/Descriptive/` | Positive and negative percentile displacement panels |
| `python Code/plot_apparel_displacement.py` | Apparel box-plot summaries in `Results/Descriptive/` | Signed displacement by apparel status |
| `python Code/plot_rq2_spending_activity.py` | Ordered decile summaries in `Results/Descriptive/` | Spending and activity per thirty observed days |
| `python Code/plot_exposure_and_popularity.py` | Category scatter layers in `Inputs/category_availability_map.pdf` and reported aggregate percentages | Availability scatter panels and the Top-20 popularity mix comparison |

The exposure plotting script preserves the scatter layers and reconstructs the axes and labels. It does not infer underlying point records from the graphic or fit a recommender.

## Reconstruct the training reference

Obtain `amazon-purchases.csv` from [Open E-Commerce 1.0](https://doi.org/10.1038/s41597-024-03329-6). Keep the CSV outside the repository. Run:

```sh
python Code/training_mix_reference.py --purchases /path/to/amazon-purchases.csv
```

The script checks the chronological split and 5-by-5 training core and writes the equal-buyer mean Head/Mid/Tail mix to `Results/TrainingReference/`. The reference uses the 4,012 buyers eligible for novel catalog-known validation; it should round to 25.78%, 26.32%, and 47.90%.

## Scope

This package contains the supporting scripts available for the thesis figures and training reference. It does not contain the complete code for fitting the five recommendation models or reconstructing every empirical result. The thesis methodology and computational reproducibility appendix describe those protocols. The raw purchases, buyer-level histories, and recommendation lists are not distributed here.
