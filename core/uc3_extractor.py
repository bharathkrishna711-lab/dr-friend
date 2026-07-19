"""
uc3_extractor.py

UC3 lab report extraction pipeline. Built in stages:
1. Raw text extraction (pdfplumber) -- this stage
2. Structured value parsing (regex + LLM fallback) -- next
3. Abnormal-value flagging -- next
"""

import pdfplumber
import os
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_REPORT = os.path.join(CURRENT_DIR, "..", "data", "sample_report", "Sunrise_Diagnostics_Lab_Report.pdf")

def extract_raw_text(pdf_path: str) -> str:
    """
    Stage 1: Extract raw text from a digital PDF lab report using
    pdfplumber. This is the foundation everything else builds on --
    we inspect the raw output first to understand what structure
    (or lack of structure) we're working with before writing any
    parsing logic on top of it.
    """
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            full_text += f"\n--- PAGE {i + 1} ---\n"
            full_text += page_text if page_text else "[No text extracted]"
    return full_text


def extract_raw_tables(pdf_path: str) -> list:
    """
    pdfplumber can also extract tables directly as structured lists,
    which may work better than raw text for our table-heavy lab
    report layout. Testing this alongside raw text extraction to see
    which gives cleaner, more parseable output.
    """
    all_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for j, table in enumerate(tables):
                all_tables.append({
                    "page": i + 1,
                    "table_index": j,
                    "rows": table,
                })
    return all_tables




def parse_numeric_value(result_str: str) -> float | None:
    """
    Extracts the numeric value from a result string like "10.2 g/dL"
    or "12,800 /cumm" -- strips commas (thousands separators) before
    matching, since Python's float() can't parse "12,800" directly.
    """
    cleaned = result_str.replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+", cleaned)
    return float(match.group()) if match else None


def parse_reference_range(range_str: str) -> tuple:
    """
    Handles four real-world reference range formats, checked in order:
    - "less than 140 mg/dL"  -> (None, 140.0)
    - "< 200 mg/dL"          -> (None, 200.0)   -- symbol form, same meaning
    - "greater than 40"      -> (40.0, None)
    - "> 40 mg/dL"           -> (40.0, None)     -- symbol form
    - "13.0 - 17.0 g/dL"     -> (13.0, 17.0)     -- standard dash range

    WHY BOTH WORD AND SYMBOL FORMS: testing against a third, differently
    formatted sample report revealed that some labs use "<"/">" symbols
    instead of "less than"/"greater than" text. Without handling both,
    any range using symbols silently parses as (None, None), causing
    compute_flag() to default every value in that table to "Normal"
    regardless of the actual result -- a serious, silent failure mode
    that a fixed regex pattern would not catch or warn about.
    """
    cleaned = range_str.replace(",", "")

    less_word = re.search(r"less than\s*([\d.]+)", cleaned, re.IGNORECASE)
    less_symbol = re.search(r"<\s*([\d.]+)", cleaned)
    if less_word:
        return (None, float(less_word.group(1)))
    if less_symbol:
        return (None, float(less_symbol.group(1)))

    greater_word = re.search(r"greater than\s*([\d.]+)", cleaned, re.IGNORECASE)
    greater_symbol = re.search(r">\s*([\d.]+)", cleaned)
    if greater_word:
        return (float(greater_word.group(1)), None)
    if greater_symbol:
        return (float(greater_symbol.group(1)), None)

    dash_match = re.search(r"([\d.]+)\s*-\s*([\d.]+)", cleaned)
    if dash_match:
        return (float(dash_match.group(1)), float(dash_match.group(2)))

    return (None, None)

def compute_flag(value: float, range_low: float, range_high: float) -> str:
    """
    Stage 3: independently computes Normal/High/Low from the parsed
    numeric value and reference bounds -- NOT read from any flag the
    lab report itself printed. This is deterministic, same philosophy
    as UC1's rule-based urgency engine: the system reasons from the
    actual numbers, not from an external, unverified label.

    Handles one-sided ranges (e.g. "less than 140" has no lower bound,
    so a low value there is never flagged Low -- there is no clinical
    lower limit being tested).
    """
    if value is None:
        return "Unknown"

    if range_low is not None and value < range_low:
        return "Low"
    if range_high is not None and value > range_high:
        return "High"
    return "Normal"

# Recognized header keywords per column type. Matched case-insensitively
# as substrings, so "Test Name", "Test", "Parameter" all map to "name";
# "Reference Range", "Biological Reference Interval", "Normal Range" all
# map to "range", etc. This is what lets the parser handle real-world
# reports that don't all use identical column headers.
COLUMN_KEYWORDS = {
    "name": ["test name", "parameter", "investigation", "test"],
    "result": ["result", "value"],
    "unit": ["unit"],
    "range": ["reference range", "reference interval", "normal range",
              "biological reference", "range", "interval"],
    "flag": ["flag", "interpretation", "remark", "status"],
}


def detect_columns(header_row: list) -> dict:
    """
    Maps each column index in a table's header row to a semantic role
    (name/result/unit/range/flag) by matching header text against known
    keyword synonyms, rather than assuming a fixed column order.

    WHY THIS IS NECESSARY: real-world lab reports vary column order,
    naming, and count between labs (confirmed by testing against 3
    differently-formatted sample reports). A parser that assumes
    column 0 is always the test name and column 3 is always the flag
    will silently misread reports that don't match that exact layout --
    which is exactly what happened before this fix: a report with
    columns in a different order had its Unit column misread as the
    reference range, causing every abnormal value to compute as
    "Normal" with no error or warning.

    Returns a dict like {"name": 0, "result": 1, "unit": 2, "range": 3}.
    Keys for column types not found in this particular table are omitted
    -- "unit" and "flag" are optional, "name"/"result"/"range" are
    required for a table to be considered parseable.
    """
    column_map = {}
    for idx, header_cell in enumerate(header_row):
        if header_cell is None:
            continue
        header_text = header_cell.strip().lower()
        for role, keywords in COLUMN_KEYWORDS.items():
            if role in column_map:
                continue  # first match wins if a role appears twice
            if any(keyword in header_text for keyword in keywords):
                column_map[role] = idx
    return column_map


def extract_structured_values(pdf_path: str) -> list:
    """
    Full Stage 2+3 pipeline: extracts tables via pdfplumber, identifies
    each table's column layout from its own header row (see
    detect_columns), and parses each data row into a structured record
    with an independently computed flag.

    Tables where a header row cannot be confidently identified (missing
    a name, result, or range column) are skipped WITH AN EXPLICIT WARNING
    printed -- never silently, since a silently-dropped table (as
    happened before this fix) can mean clinically important values never
    reach the rest of the system with no indication anything went wrong.
    """
    tables = extract_raw_tables(pdf_path)
    structured = []

    for table in tables:
        rows = table["rows"]
        if not rows:
            continue

        header_row = rows[0]
        columns = detect_columns(header_row)

        required = {"name", "result", "range"}
        if not required.issubset(columns.keys()):
            print(
                f"WARNING: Skipping table on page {table['page']} "
                f"(table {table['table_index']}) -- could not identify "
                f"required columns from header: {header_row}"
            )
            continue

        for row in rows[1:]:
            if not row or all(cell is None for cell in row):
                continue

            name_idx = columns["name"]
            result_idx = columns["result"]
            range_idx = columns["range"]
            unit_idx = columns.get("unit")
            flag_idx = columns.get("flag")

            if name_idx >= len(row) or result_idx >= len(row) or range_idx >= len(row):
                continue

            test_name = row[name_idx]
            result_str = row[result_idx]
            range_str = row[range_idx]
            unit_str = row[unit_idx] if unit_idx is not None and unit_idx < len(row) else ""
            printed_flag = row[flag_idx] if flag_idx is not None and flag_idx < len(row) else None

            if test_name is None or result_str is None or range_str is None:
                continue

            # Combine result + unit into one display string when they're
            # separate columns (e.g. Value="8.9", Unit="uIU/mL" becomes
            # "8.9 uIU/mL"), matching the format used when a report
            # already combines them into one cell.
            raw_result = result_str.strip()
            if unit_str and unit_str.strip() and unit_str.strip() not in raw_result:
                raw_result = f"{raw_result} {unit_str.strip()}"

            value = parse_numeric_value(result_str)
            range_low, range_high = parse_reference_range(range_str)
            computed_flag = compute_flag(value, range_low, range_high)

            printed_flag_clean = printed_flag.strip() if printed_flag else None

            structured.append({
                "test_name": test_name.strip(),
                "raw_result": raw_result,
                "value": value,
                "raw_range": range_str.strip(),
                "range_low": range_low,
                "range_high": range_high,
                "computed_flag": computed_flag,
                "printed_flag": printed_flag_clean,
                "flags_agree": computed_flag == printed_flag_clean if printed_flag_clean else None,
            })

    return structured



if __name__ == "__main__":
    import os as _os

    reports = {
        "Sunrise (original format)": "Sunrise_Diagnostics_Lab_Report.pdf",
        "Metropolis (2nd format)": "Metropolis_City_Lab_Report.pdf",
        "Apex Path Labs (3rd, different layout)": "Apex_Path_Labs_Report.pdf",
    }

    for label, filename in reports.items():
        path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "sample_report", filename)
        print("=" * 70)
        print(f"REPORT: {label}")
        print("=" * 70)
        results = extract_structured_values(path)
        for r in results:
            print(
                f"{r['test_name']:20s} | value={r['value']} | "
                f"range=({r['range_low']}, {r['range_high']}) | "
                f"computed={r['computed_flag']}"
            )
        print(f"Total rows parsed: {len(results)}\n")