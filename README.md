# UIDAI Data Hackathon (Analysis Notebooks)

This repository contains Jupyter notebooks (and their `jupytext`-exported `.py` equivalents) used for analysis during the UIDAI data hackathon.

## What’s inside

- **`books/book1.ipynb`**: Operational load hotspots (enrolment + demographic updates + biometric updates), aggregated at pincode level with volatility metrics.
- **`books/book2.ipynb`**: Update-heavy but enrolment-light regions (district-level “update pressure” ratios). Depends on outputs from `book1`.
- **`books/book3.ipynb`**: Age-driven service pressure (district-level adult vs child update activity). Depends on dataframes from `book1`.

The corresponding `books/book*.py` files are the same notebooks saved in a script-friendly format.

## Repository structure

```
.
├── README.md
└── books/
    ├── book1.ipynb
    ├── book2.ipynb
    ├── book3.ipynb
    ├── book1.py
    ├── book2.py
    ├── book3.py
    └── data/
        ├── raw/
        └── parquet/
```

## Data layout (expected)

The notebooks expect raw CSVs under `books/data/raw/`.

This repository also includes **cleaned parquet outputs** under `books/data/parquet/`:

```
data/parquet/enrol_clean.parquet
data/parquet/demo_clean.parquet
data/parquet/bio_clean.parquet
```

However, the notebooks are currently written to load the **raw CSV slices** (see below). If you want to run purely from parquet, you’ll need to update `book1` to read from `data/parquet/*.parquet` instead of `data/raw/...`.

`book1` reads files using paths like:

```
data/raw/api_data_aadhar_enrolment/api_data_aadhar_enrolment_0_500000.csv
data/raw/api_data_aadhar_demographic/api_data_aadhar_demographic_0_500000.csv
data/raw/api_data_aadhar_biometric/api_data_aadhar_biometric_0_500000.csv
```

Important:

- Run notebooks with the **working directory set to `books/`** so the relative paths above resolve correctly.
- Raw datasets are typically not committed (see `.gitignore`). Place your CSVs locally under the paths above.

## Setup

This repo doesn’t pin dependencies yet. A minimal environment to run the notebooks is:

- Python 3.10+
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `jupyter` (or `jupyterlab`)
- `adjustText` (used in `book3`)
- `pyarrow` (recommended if you want to load the included parquet files)

Example setup (macOS/Linux):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install pandas numpy matplotlib seaborn jupyterlab adjustText

# If you want to load parquet inputs
pip install pyarrow
```

## Running the notebooks

1. Ensure your raw CSVs exist under `books/data/raw/...` (see above).
2. Start Jupyter from the `books/` folder:

```bash
cd books
jupyter lab
```

Then open `book1.ipynb` first (since `book2` and `book3` import from it).

## Notes

- `book2.py` and `book3.py` import objects from `book1` (for example `from book1 import pincode_df`). If you run scripts directly, ensure:
  - you run them from `books/`, and
  - `book1` can execute end-to-end (it loads the raw CSVs).

## License

Internal hackathon work-in-progress.
