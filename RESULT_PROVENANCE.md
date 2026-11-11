# Result provenance

## Descriptive analysis

The cleaned category-complete population has 1,726,977 purchase rows, 5,022 buyers, and 875,367 products. Purchased units determine popularity and HTB labels. The primary global/local comparison uses 869,135 eligible products. Buyer composition counts purchase rows; expenditure is unit price multiplied by quantity.

`Results/Descriptive/` supplies the product examples with source identifiers and titles, apparel box summaries, and numerically ordered buyer-decile summaries. The methodology and figure captions define their populations and denominators.

## Chronological recommendation evaluation

Training has 1,381,836 rows through 18 April 2022; validation has 345,141 rows from 19 April 2022. The training-only core contains 4,507 buyers and 30,360 products. All 4,012 buyers with a novel catalog-known validation target form the common evaluated cohort. Retrieval and exposure use the same lists; Top-10 and Top-20 are prefixes of Top-30. Training quantities, modal categories, and HTB labels are frozen before validation. BPR-MF and NCF use the stated NumPy Adam implementations, two epochs, and seed 42; validation does not select their settings.

## Training purchase reference

`Results/TrainingReference/` records mean within-buyer Head/Mid/Tail shares of catalog-classifiable training purchase rows: 25.779114%, 26.321685%, and 47.899201%. The cohort is the same 4,012 buyers; rank-defined Tail is 68.642623%. Coverage does not apply to this reference. `Code/training_mix_reference.py` reconstructs these values from the external purchase CSV.

## Scope

The project contains the figures required for LaTeX compilation and the supporting scripts listed in README.md. Empirical reconstruction requires the same external source records. Offline recommendation positions do not identify causal effects of Amazon recommendations or feedback processes.
