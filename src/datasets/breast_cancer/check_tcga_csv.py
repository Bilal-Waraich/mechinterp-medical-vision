"""Inspect tcga_brca_all_clean.csv — verify column matching and label coverage.

Run from the repo root:
    python src/datasets/breast_cancer/check_tcga_csv.py \
        --csv \\fileserver\share\deep_learning\input\data\mmm\tcga_breast\tcga_brca_all_clean.csv
"""

import argparse
import csv
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Must mirror tcga_breast_dataset.py exactly
_PATIENT_COL_CANDIDATES = [
    "case_id", "submitter_id", "PATIENT_ID", "patient_id",
    "bcr_patient_barcode", "PATIENT",
]
_SUBTYPE_COL_CANDIDATES = [
    "oncotree_code", "ONCOTREE_CODE",
    "SUBTYPE", "subtype", "PAM50", "pam50",
    "BRCA_Subtype_PAM50", "brca_subtype_pam50",
    "paper_BRCA_Subtype_PAM50",
]
_LABEL_MAP = {
    "idc": 0, "ilc": 1, "mdlc": 2, "ibc": 3,
    "brca": 4, "mpt": 4, "clis": 4, "immc": 4, "pure": 4,
}
CLASS_NAMES = ["IDC", "ILC", "MDLC", "IBC", "Other"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify tcga_brca_all_clean.csv column matching")
    ap.add_argument("--csv", required=True, help="Path to tcga_brca_all_clean.csv")
    ap.add_argument("--rows", type=int, default=5, help="Sample rows to show (default 5)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader  = csv.DictReader(fh)
        headers = list(reader.fieldnames or [])
        rows    = [row for row, _ in zip(reader, range(args.rows))]

    print("=" * 60)
    print(f"File   : {csv_path}")
    print(f"Columns: {len(headers)}")
    print()

    patient_col = next((c for c in _PATIENT_COL_CANDIDATES if c in headers), None)
    subtype_col = next((c for c in _SUBTYPE_COL_CANDIDATES if c in headers), None)

    print("Patient column candidates:")
    for c in _PATIENT_COL_CANDIDATES:
        hit = "[MATCH]" if c == patient_col else ("[found]" if c in headers else "   -")
        print(f"  {c:<35} {hit}")

    print()
    print("Subtype column candidates:")
    for c in _SUBTYPE_COL_CANDIDATES:
        hit = "[MATCH]" if c == subtype_col else ("[found]" if c in headers else "   -")
        print(f"  {c:<35} {hit}")

    print()
    if patient_col and subtype_col:
        print(f"OK  Will use  patient='{patient_col}'  subtype='{subtype_col}'")
    else:
        print("MISMATCH -- labels will all be -1")
        print()
        print("First 15 column names in the CSV:")
        for h in headers[:15]:
            print(f"  {h!r}")
        sys.exit(1)

    # Sample rows + label mapping check
    print()
    print(f"Sample ({args.rows} rows):")
    print(f"  {'patient_id ([:12])':<22} {'subtype raw':<20} -> label")
    print("  " + "-" * 55)
    for r in rows:
        pid     = r[patient_col].strip()[:12]
        subtype = r[subtype_col].strip()
        label   = _LABEL_MAP.get(subtype.lower(), -1)
        lname   = CLASS_NAMES[label] if label >= 0 else "unknown"
        print(f"  {pid:<22} {subtype:<20} -> {label} ({lname})")

    # Full coverage stats
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        all_rows = list(csv.DictReader(fh))

    total = len(all_rows)
    from collections import Counter
    code_counts = Counter(r[subtype_col].strip().lower() for r in all_rows)
    unmapped    = {c for c in code_counts if _LABEL_MAP.get(c, -1) == -1 and c}

    print()
    print(f"Total rows: {total}")
    print(f"Label distribution:")
    for name in CLASS_NAMES:
        idx   = CLASS_NAMES.index(name)
        codes = [c for c, v in _LABEL_MAP.items() if v == idx]
        n     = sum(code_counts.get(c, 0) for c in codes)
        print(f"  {name:<10}: {n:>6,}")
    print(f"  {'unknown':<10}: {sum(code_counts.get(c, 0) for c in unmapped):>6,}")
    if unmapped:
        print(f"  Unmapped codes: {sorted(unmapped)}")
        print("  -> Add these to _LABEL_MAP in tcga_breast_dataset.py")


if __name__ == "__main__":
    main()