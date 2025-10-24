# Name: Luke Burden
# Student ID: 46484947
# Email: lukeburd@umich.edu
# Course: SI 201  •  Project 1
# Dataset: Agriculture Crop Yield (Kaggle)
# GenAI usage:
# ChatGPT: (structuring, scaffolding, and example tests)
# No Collaborators.
 
from __future__ import annotations
import csv
import argparse
import os
from typing import List, Dict, Any, Iterable, Optional


# -----------------------------
# ---------- Utilities --------
# -----------------------------
def _to_float(x: Any) -> Optional[float]:
    """Safely convert to float; returns None on failure."""
    if x is None:
        return None
    try:
        s = str(x).strip()
        if s == "" or s.lower() in {"na", "n/a", "null"}:
            return None
        return float(s)
    except Exception:
        return None


def _norm_str(x: Any) -> str:
    """Normalize strings for case-insensitive matching."""
    return str(x).strip().lower() if x is not None else ""


def filter_rows(
    rows: Iterable[Dict[str, Any]],
    **equals: str
) -> List[Dict[str, Any]]:
    """
    Return rows where ALL provided key=value pairs match (case-insensitive string match).
    Example: filter_rows(data, Crop="Soybean", Region="East")
    """
    out = []
    eq_norm = {k: _norm_str(v) for k, v in equals.items()}
    for r in rows:
        keep = True
        for k, want in eq_norm.items():
            have = _norm_str(r.get(k))
            if have != want:
                keep = False
                break
        if keep:
            out.append(r)
    return out


# -----------------------------------
# ---------- Core I/O Functions -----
# -----------------------------------
def read_csv_file(filename: str) -> List[Dict[str, Any]]:
    """
    Read a CSV into a list of dictionaries.
    Raises FileNotFoundError if file is missing.
    """
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def write_txt_results(path: str, content: str) -> None:
    """Write plain-text output."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def write_csv_from_mapping(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    """Write a CSV given a list of row dicts and explicit headers."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


# -------------------------------------------------
# ---------- Calculation 1 (uses 3+ cols) ---------
# -------------------------------------------------
def calculate_avg_days_and_rainfall_by_crop(
    data: List[Dict[str, Any]],
    crops: Iterable[str],
    crop_col: str = "Crop",
    days_col: str = "Days_to_Harvest",
    rainfall_col: str = "Rainfall_mm",
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    For each crop in `crops`, compute:
      - avg_days_to_harvest
      - avg_rainfall_mm
      - count_used
    Uses ≥3 columns: Crop (category), Days_to_Harvest (numeric), Rainfall_mm (numeric).
    Returns a mapping like:
      {
        "soybean": {"avg_days": 87.3, "avg_rainfall": 112.0, "count": 54},
        ...
      }
    Crop keys are lowercased to make output consistent.
    """
    wanted = {_norm_str(c) for c in crops}
    sums_days = {c: 0.0 for c in wanted}
    sums_rain = {c: 0.0 for c in wanted}
    counts    = {c: 0   for c in wanted}

    for row in data:
        crop_val = _norm_str(row.get(crop_col))
        if crop_val not in wanted:
            continue

        days = _to_float(row.get(days_col))
        rain = _to_float(row.get(rainfall_col))

        # Only include rows where BOTH key numerics are present
        if days is None or rain is None:
            continue

        sums_days[crop_val] += days
        sums_rain[crop_val] += rain
        counts[crop_val]    += 1

    result: Dict[str, Dict[str, Optional[float]]] = {}
    for c in wanted:
        n = counts[c]
        result[c] = {
            "avg_days": (sums_days[c] / n) if n > 0 else None,
            "avg_rainfall": (sums_rain[c] / n) if n > 0 else None,
            "count": n,
        }
    return result


# -------------------------------------------------
# ---------- Calculation 2 (uses 3+ cols) ---------
# -------------------------------------------------
def calculate_avg_temperature_for_crop_in_region(
    data: List[Dict[str, Any]],
    target_crop: str = "Soybean",
    target_region: str = "East",
    temp_col: str = "Temperature_Celsius",
    crop_col: str = "Crop",
    region_col: str = "Region",
) -> Optional[float]:
    """
    Compute the average Temperature_Celsius for rows where Crop==target_crop AND Region==target_region.
    Uses ≥3 columns: Temperature_Celsius (numeric), Crop (category), Region (category).
    Returns None if there are no valid numeric temperatures among matching rows.
    """
    total = 0.0
    n = 0
    target_crop_norm = _norm_str(target_crop)
    target_region_norm = _norm_str(target_region)

    for row in data:
        if _norm_str(row.get(crop_col)) != target_crop_norm:
            continue
        if _norm_str(row.get(region_col)) != target_region_norm:
            continue

        t = _to_float(row.get(temp_col))
        if t is None:
            continue
        total += t
        n += 1

    return (total / n) if n > 0 else None


# -----------------------------------------
# ---------- Pretty Formatting ------------
# -----------------------------------------
def format_results_text(
    per_crop: Dict[str, Dict[str, Optional[float]]],
    soybean_east_avg_temp: Optional[float],
) -> str:
    lines = []
    lines.append("SI 201 • Project 1 Results")
    lines.append("Dataset: Agriculture Crop Yield (Kaggle)")
    lines.append("------------------------------------------------------------")
    lines.append("Calculation 1: Average Days to Harvest AND Average Rainfall by Crop")
    lines.append("  (Crops: Cotton, Wheat, Soybean)")
    for crop in ["cotton", "wheat", "soybean"]:
        stats = per_crop.get(crop, {})
        lines.append(
            f"  - {crop.title():<7} | avg days: {stats.get('avg_days')} days | "
            f"avg rainfall: {stats.get('avg_rainfall')} mm | count: {stats.get('count')}"
        )
    lines.append("")
    lines.append("Calculation 2: Average Temperature for Soybean in East")
    lines.append(f"  - Avg Temperature (Soybean, East): {soybean_east_avg_temp} °C")
    lines.append("------------------------------------------------------------")
    lines.append("Notes:")
    lines.append("  • Rows with missing or invalid numeric values are excluded from averages.")
    lines.append("  • Matching is case-insensitive for categorical fields (Crop, Region).")
    return "\n".join(lines)


# -----------------------------------------
# ---------- Example Tests (4 each) -------
# -----------------------------------------
def _make_mock_data() -> List[Dict[str, Any]]:
    # Minimal realistic rows for testing (matching your exact column names)
    return [
        {"Crop": "Soybean", "Region": "East", "Days_to_Harvest": "90", "Rainfall_mm": "110", "Temperature_Celsius": "22.5"},
        {"Crop": "Soybean", "Region": "EAST", "Days_to_Harvest": "95", "Rainfall_mm": "108", "Temperature_Celsius": "23.0"},
        {"Crop": "Wheat",   "Region": "West", "Days_to_Harvest": "120","Rainfall_mm": "85",  "Temperature_Celsius": "19.5"},
        {"Crop": "Cotton",  "Region": "South","Days_to_Harvest": "100","Rainfall_mm": "120", "Temperature_Celsius": "25.0"},
        # Edge: missing numeric -> excluded
        {"Crop": "Soybean", "Region": "East", "Days_to_Harvest": "",   "Rainfall_mm": "117", "Temperature_Celsius": "22.0"},
        # Edge: invalid numeric -> excluded
        {"Crop": "Wheat",   "Region": "East", "Days_to_Harvest": "abc","Rainfall_mm": "70",  "Temperature_Celsius": "18.0"},
        # Edge: different crop not requested
        {"Crop": "Barley",  "Region": "East", "Days_to_Harvest": "75", "Rainfall_mm": "95",  "Temperature_Celsius": "17.0"},
    ]


def run_tests() -> None:
    data = _make_mock_data()

    # ---- Tests for Calculation 1 ----
    crops = ["Cotton", "Wheat", "Soybean"]
    per_crop = calculate_avg_days_and_rainfall_by_crop(
        data, crops, crop_col="Crop", days_col="Days_to_Harvest", rainfall_col="Rainfall_mm"
    )

    # General case 1: Soybean averages (only valid numerics counted)
    s = per_crop["soybean"]
    assert s["count"] == 2, "Soybean count should be 2 (exclude missing days row)"
    assert round(s["avg_days"], 2) == 92.5, "Soybean avg days should be (90+95)/2 = 92.5"
    assert round(s["avg_rainfall"], 2) == 109.0, "Soybean avg rainfall (110+108)/2 = 109.0"

    # General case 2: Cotton averages
    c = per_crop["cotton"]
    assert c["count"] == 1, "Cotton count should be 1"
    assert c["avg_days"] == 100.0, "Cotton avg days should be 100"
    assert c["avg_rainfall"] == 120.0, "Cotton avg rainfall should be 120"

    # Edge case 1: Wheat row with invalid days -> excluded from avg days & rainfall
    w = per_crop["wheat"]
    # There is one valid Wheat row (West) with valid numerics
    assert w["count"] == 1, "Wheat valid count should be 1 (exclude invalid 'abc')"
    assert w["avg_days"] == 120.0, "Wheat avg days should be 120"
    assert w["avg_rainfall"] == 85.0, "Wheat avg rainfall should be 85"

    # Edge case 2: Unrequested crop not included
    assert "barley" not in per_crop, "Unrequested crop (barley) should not appear"

    # ---- Tests for Calculation 2 ----
    avg_temp_se = calculate_avg_temperature_for_crop_in_region(
        data,
        target_crop="Soybean",
        target_region="East",
        temp_col="Temperature_Celsius",
        crop_col="Crop",
        region_col="Region",
    )

    # General case 1: Two valid soybean-east temps 22.5 and 23.0
    assert round(avg_temp_se, 2) == 22.75, "Avg temp should be (22.5+23.0)/2 = 22.75"

    # General case 2: Case-insensitive matching for region
    assert avg_temp_se is not None, "Average should be computed case-insensitively"

    # Edge case 1: No matches (change target crop)
    avg_none = calculate_avg_temperature_for_crop_in_region(
        data, target_crop="Rice", target_region="East"
    )
    assert avg_none is None, "Should return None when there are no matching rows"

    # Edge case 2: Matching rows but all invalid temperature -> craft a small set
    invalid_temps = [
        {"Crop": "Soybean", "Region": "East", "Temperature_Celsius": ""},
        {"Crop": "Soybean", "Region": "East", "Temperature_Celsius": "N/A"},
    ]
    avg_invalid = calculate_avg_temperature_for_crop_in_region(invalid_temps)
    assert avg_invalid is None, "Should return None when all temps are invalid"

    print("✅ All tests passed.")


# -----------------------------------------
# ---------- Main / CLI -------------------
# -----------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SI 201 • Project 1: Agriculture Crop Yield Analysis")
    parser.add_argument("--csv", type=str, default="crop_yield 2.csv",
                        help='Path to the Kaggle CSV file (default: "crop_yield 2.csv")')
    parser.add_argument("--out_txt", type=str, default="project1_results.txt",
                        help="Path to write the human-readable results (.txt)")
    parser.add_argument("--out_csv", type=str, default="avg_by_crop.csv",
                        help="Path to write per-crop averages (.csv)")
    parser.add_argument("--run_tests", action="store_true",
                        help="Run built-in tests (no CSV required) and exit")
    args = parser.parse_args()

    if args.run_tests:
        run_tests()
        return

    # 1) Resolve CSV path and auto-detect if missing
    csv_path = args.csv
    script_dir = os.path.dirname(__file__)
    # If path is relative and doesn't exist in CWD, try relative to script directory
    if not os.path.isabs(csv_path):
        rel_try = os.path.join(script_dir, csv_path)
        if os.path.exists(rel_try):
            csv_path = rel_try

    if not os.path.exists(csv_path):
        # Try to auto-detect a CSV in the project folder or data/ subfolder
        candidates = []
        for folder in (script_dir, os.path.join(script_dir, "data")):
            if os.path.isdir(folder):
                for name in os.listdir(folder):
                    if name.lower().endswith(".csv"):
                        candidates.append(os.path.join(folder, name))

        # Prefer CSVs with 'yield' or 'crop' in the filename
        preferred = [p for p in candidates if any(k in os.path.basename(p).lower() for k in ("yield", "crop"))]
        chosen = preferred[0] if preferred else (candidates[0] if candidates else None)

        if chosen:
            # Use detected CSV silently to avoid confusing warnings
            csv_path = chosen
        else:
            print(f"⚠️ CSV file not found: {args.csv}")
            print('   Provide the correct path with --csv "PATH/TO/crop_yield 2.csv"')
            return

    data = read_csv_file(csv_path)

    # 2) Perform calculations per checkpoint
    crops = ["Cotton", "Wheat", "Soybean"]
    per_crop = calculate_avg_days_and_rainfall_by_crop(
        data, crops, crop_col="Crop", days_col="Days_to_Harvest", rainfall_col="Rainfall_mm"
    )

    soybean_east_avg_temp = calculate_avg_temperature_for_crop_in_region(
        data,
        target_crop="Soybean",
        target_region="East",
        temp_col="Temperature_Celsius",
        crop_col="Crop",
        region_col="Region",
    )

    # 3) Write outputs
    txt = format_results_text(per_crop, soybean_east_avg_temp)
    write_txt_results(args.out_txt, txt)

    rows_for_csv = []
    for crop_key in ["cotton", "wheat", "soybean"]:
        stats = per_crop.get(crop_key, {})
        rows_for_csv.append({
            "crop": crop_key,
            "avg_days_to_harvest": stats.get("avg_days"),
            "avg_rainfall_mm": stats.get("avg_rainfall"),
            "count": stats.get("count"),
        })
    write_csv_from_mapping(
        args.out_csv,
        fieldnames=["crop", "avg_days_to_harvest", "avg_rainfall_mm", "count"],
        rows=rows_for_csv
    )

    # 4) Console summary (nice for quick verification)
    print("✅ Analysis complete.")
    print(f"   Wrote text summary to: {args.out_txt}")
    print(f"   Wrote per-crop CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()