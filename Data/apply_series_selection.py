"""
apply_series_selection.py
Build the reduced-information dataset used in the series-selection
robustness check.

Reads data_tf.csv and meta_data_v2.csv from an input Data folder, drops every
series flagged include == 0 in series_selection.csv, and writes both files to
an output Data folder. The target (GDP) is never dropped.

Usage
-----
python Data/apply_series_selection.py <input_data_dir> <output_data_dir>

Author : Gabriela Aquino
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TARGET = "GDP"


def main(in_dir: Path, out_dir: Path) -> None:
    selection = pd.read_csv(HERE / "series_selection.csv")
    excluded = selection.loc[selection["include"] == 0, "series"].tolist()
    if TARGET in excluded:
        raise ValueError("The target variable cannot be excluded.")

    data = pd.read_csv(in_dir / "data_tf.csv")
    meta = pd.read_csv(in_dir / "meta_data_v2.csv")

    missing = [s for s in excluded if s not in data.columns]
    if missing:
        raise KeyError(f"Excluded series not found in data_tf.csv: {missing}")

    data_red = data.drop(columns=excluded)
    meta_red = meta.loc[~meta["series"].isin(excluded)].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    data_red.to_csv(out_dir / "data_tf.csv", index=False)
    meta_red.to_csv(out_dir / "meta_data_v2.csv", index=False)
    # Copy the selection file so downstream scripts (e.g. MIDAS.R) know
    # which missing series were dropped on purpose.
    selection.to_csv(out_dir / "series_selection.csv", index=False)

    print(f"Excluded {len(excluded)} series: {', '.join(excluded)}")
    print(f"Series kept: {len(meta_red)} of {len(meta)}")
    print(f"Written to: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
