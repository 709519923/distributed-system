"""CSV input and output helpers.

Rank 0 is the only rank that reads prompts and writes generated outputs. If
prompt counts, header handling, prompt-column selection, or output rows look
wrong, this file is the smallest place to debug.
"""

import csv
from pathlib import Path


def read_prompts(csv_path, has_header, prompt_column):
    """Read prompts from CSV.

    Supported formats:
    - Without header: read the first column by default, or --prompt-column as a
      zero-based column index.
    - With header: read --prompt-column as a column name, or use the first header
      column if --prompt-column is omitted.
    """
    prompts = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        if has_header:
            reader = csv.DictReader(f)
            if prompt_column is None:
                if not reader.fieldnames:
                    return []
                prompt_column = reader.fieldnames[0]
            for row in reader:
                value = (row.get(prompt_column) or "").strip()
                if value:
                    prompts.append(value)
        else:
            column_index = int(prompt_column) if prompt_column is not None else 0
            reader = csv.reader(f)
            for row in reader:
                if len(row) <= column_index:
                    continue
                value = row[column_index].strip()
                if value:
                    prompts.append(value)
    return prompts


def read_contextual_controlled_rows(csv_path, prompt_column):
    """Read observable request lengths plus audit labels for controlled DEF."""
    records = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("contextual_controlled requires a CSV header.")

        prompt_key = prompt_column or reader.fieldnames[0]
        required = {
            prompt_key,
            "scenario",
            "request_type",
            "target_output_tokens",
            "phase",
            "source_row",
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(
                "contextual_controlled CSV is missing columns: " + ", ".join(missing)
            )

        for row_number, row in enumerate(reader, start=2):
            prompt = (row.get(prompt_key) or "").strip()
            if not prompt:
                raise ValueError(
                    f"contextual_controlled CSV row {row_number} has an empty prompt."
                )
            try:
                target_output_tokens = int(
                    (row.get("target_output_tokens") or "").strip()
                )
                source_row = int((row.get("source_row") or "").strip())
            except ValueError as exc:
                raise ValueError(
                    f"contextual_controlled CSV row {row_number} has invalid "
                    "target_output_tokens/source_row."
                ) from exc
            if target_output_tokens < 1 or source_row < 1:
                raise ValueError(
                    f"contextual_controlled CSV row {row_number} has non-positive "
                    "target_output_tokens/source_row."
                )
            records.append(
                {
                    "prompt": prompt,
                    "scenario": (row.get("scenario") or "").strip(),
                    "request_type": (row.get("request_type") or "").strip(),
                    "target_output_tokens": target_output_tokens,
                    "phase": (row.get("phase") or "").strip().lower(),
                    "source_row": source_row,
                }
            )
    return records


def read_def_interleaved_rows(csv_path):
    """Read and validate the 1,500-row DEF sidecar manifest.

    Prompts remain in the normal input CSV. This sidecar describes the
    observable request budget for each prompt and provides audit-only D/E/F
    metadata, so the normal prompt reader remains compatible with older runs.
    """
    records = []
    required = {
        "batch",
        "scenario",
        "request_type",
        "target_output_tokens",
        "source_row",
    }
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("DEF context manifest requires a CSV header.")
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError("DEF context manifest is missing columns: " + ", ".join(missing))

        for row_number, row in enumerate(reader, start=2):
            try:
                batch = int(str(row.get("batch") or "").strip())
                source_row = int(str(row.get("source_row") or "").strip())
                target_output_tokens = int(
                    str(row.get("target_output_tokens") or "").strip()
                )
            except ValueError as exc:
                raise ValueError(
                    f"DEF context manifest row {row_number} has invalid numeric fields."
                ) from exc
            scenario = str(row.get("scenario") or "").strip().upper()
            request_type = str(row.get("request_type") or "").strip()
            expected_batch = len(records) + 1
            expected_scenario = ("D", "E", "F")[(expected_batch - 1) % 3]
            expected_source_row = (expected_batch - 1) // 3 + 1
            if batch != expected_batch:
                raise ValueError(
                    f"DEF context manifest row {row_number} must have batch="
                    f"{expected_batch}, got {batch}."
                )
            if scenario != expected_scenario or source_row != expected_source_row:
                raise ValueError(
                    f"DEF context manifest row {row_number} must be "
                    f"{expected_scenario}{expected_source_row}, got {scenario}{source_row}."
                )
            if not request_type or target_output_tokens < 1:
                raise ValueError(
                    f"DEF context manifest row {row_number} has invalid request settings."
                )
            records.append(
                {
                    "batch": batch,
                    "scenario": scenario,
                    "request_type": request_type,
                    "target_output_tokens": target_output_tokens,
                    "source_row": source_row,
                }
            )
    return records


def chunk_items(items, batch_size):
    """Yield (batch_number, start_index, chunk) for dynamic prompt batching.

    batch_number is 1-based because it is written to allocation.csv and is meant
    to be read by humans. start_index is 0-based and is only used for progress
    logging.
    """
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    for start in range(0, len(items), batch_size):
        batch_number = start // batch_size + 1
        yield batch_number, start, items[start:start + batch_size]


def write_output_rows(output_csv, rows):
    """Write Rank 0 generation results to CSV."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "generated_text", "full_text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Rank 0] Wrote {len(rows)} rows to {output_path}")
