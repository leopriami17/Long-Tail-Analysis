# Fixed chronological recommender experiment

This package contains executable analysis code, the complete reconstructed outputs of the five documented models, saved latent parameters, and a comparison with the values reported in the thesis. It does not alter the LaTeX thesis.

## Important verification result

Global quantity, Category quantity, ItemKNN, and BPR-MF match all available reported reference values at their stated display precision. NCF has small differences in several reported values. The package must not be described as identical to every reported result. See `RAPPORTO_VERIFICA.md` and `Results/Chronological8020/compatibility_report.csv`.

The original individual five-model lists are unavailable, so record-by-record identity with those lists cannot be tested. The independently reconstructed Global quantity Top-20 lists match the separate 80,240-slot audit exactly.

## Run

Install `requirements.txt`, then run:

```bash
python Code/chronological_8020.py --purchases /absolute/path/amazon-purchases.csv
```

The wrapper fixes OpenBLAS/OMP/MKL to one thread, pins the Python hash seed, and reconstructs the dated split and catalog. Models use the documented fixed settings: two latent epochs, seed 42, learning rate 0.001, batch size 4,096, 32-dimensional BPR factors, 16-dimensional NCF embeddings, 64/32 ReLU hidden layers, and one uniform unseen negative per positive. It performs no hyperparameter optimization. Training arrays are float32; numerical settings and parameter hashes are recorded with the saved fits.

The command creates `prepared.pkl` as a local cache; it is not required to inspect the archived outputs and is not distributed. The raw purchase file is an external input and is not included.

Add `--require-exact-reference` to make the command fail if any reconstructed point estimate differs at the published precision. A discrepancy is reported rather than overwritten or rounded away.

## Complete outputs

All numerical metric columns in CSVs use their natural 0-1 scale unless their names explicitly end in `_pp` or indicate counts. The thesis comparison applies the documented percentage conversion.

- `recommendation_slots.csv.gz`: 601,800 rows: five models x 4,012 buyers x 30 ordered positions; buyer, model, product, rank, HTB class, primary category, and binary validation relevance.
- `per_buyer_metrics.csv.gz`: 60,180 rows: five models x 4,012 buyers x three list lengths.
- `summary.csv`: all five models at K=10, 20, and 30.
- `training_catalog.csv.gz` and `training_users.csv`: categories, quantities, labels, titles, and embedding-index mappings.
- `buyer_history.csv`, `target_audit.csv`, `manifest.json`: buyer profiles, target eligibility, split counts, and data/cohort fingerprints.
- `category_exposure.csv`: recommendation, candidate-availability, and historical shares for all categories and models at all three list lengths.
- `product_exposure_top20.csv.gz`: all 30,360 products for every model, including products with zero recommendation exposure.
- `tail_terciles.csv`, `fairness_groups.csv`: group comparisons.
- `training_losses.csv`, `uncertainty.csv`, `verification.json`: training diagnostics, seeded bootstrap intervals, and verification status.
- `compatibility_report.csv`: every checked reference value, reconstructed value, difference, precision, and status.

`Fits/BPR-MF_raw_float32_io/` and `Fits/NCF_raw_float32_io/` contain the selected weights and their training metadata. `Reference/` supplies the numeric thesis reference and the independent Global quantity audit. The runner creates `by_model/` checkpoints during execution; the archive distributes the consolidated files to avoid duplicating the complete data.

For category-exposure analysis, use the combined slot, product-exposure, and category-exposure files from this one run. Candidate availability is a descriptive reference, not an assumption of equal relevance or a demonstrated fairness objective.
