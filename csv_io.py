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
    """Read the labeled 600-learning/300-evaluation contextual dataset."""
    records = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("contextual_controlled requires a CSV header.")

        prompt_key = prompt_column or reader.fieldnames[0]
        required = {prompt_key, "scenario", "request_type", "phase"}
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
            records.append(
                {
                    "prompt": prompt,
                    "scenario": (row.get("scenario") or "").strip(),
                    "request_type": (row.get("request_type") or "").strip(),
                    "phase": (row.get("phase") or "").strip().lower(),
                }
            )
    return records


def read_lipschitz_validation_rows(csv_path, prompt_column):
    """Read label-controlled rows for exhaustive Lipschitz validation."""
    records = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("lipschitz_validation requires a CSV header.")

        prompt_key = prompt_column or reader.fieldnames[0]
        required = {
            prompt_key,
            "scenario",
            "request_type",
            "target_output_tokens",
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(
                "lipschitz_validation CSV is missing columns: "
                + ", ".join(missing)
            )

        for row_number, row in enumerate(reader, start=2):
            prompt = (row.get(prompt_key) or "").strip()
            scenario = (row.get("scenario") or "").strip().upper()
            request_type = (row.get("request_type") or "").strip()
            if not prompt:
                raise ValueError(
                    f"lipschitz_validation CSV row {row_number} has an empty prompt."
                )
            if not scenario or not request_type:
                raise ValueError(
                    f"lipschitz_validation CSV row {row_number} has an empty label."
                )
            try:
                target_output_tokens = int(row.get("target_output_tokens", ""))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "lipschitz_validation CSV row "
                    f"{row_number} has invalid target_output_tokens."
                ) from exc
            if target_output_tokens <= 0:
                raise ValueError(
                    "lipschitz_validation CSV row "
                    f"{row_number} target_output_tokens must be positive."
                )
            records.append(
                {
                    "prompt": prompt,
                    "scenario": scenario,
                    "request_type": request_type,
                    "target_output_tokens": target_output_tokens,
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
