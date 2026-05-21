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
