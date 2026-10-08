import json
import math
from pathlib import Path
from typing import Any

from src.retrieve import retrieve

QUESTION_PATH = Path(__file__).with_name("questions.jsonl")
K_VALUES = (1, 3, 5, 10)


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be greater than zero.")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total.")
    if z <= 0:
        raise ValueError("z must be greater than zero.")

    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return center - half_width, center + half_width


def load_questions(path: Path = QUESTION_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as questions_file:
        for line_number, line in enumerate(questions_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}.") from error
            if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not isinstance(row.get("source"), str):
                raise ValueError(f"Line {line_number} must contain string 'question' and 'source' fields.")
            if not row["question"].strip() or not row["source"].strip():
                raise ValueError(f"Line {line_number} has an empty question or source.")
            rows.append(row)
    if not rows:
        raise ValueError(f"No evaluation questions found in {path}.")
    return rows


def evaluate(rows: list[dict[str, Any]], k_values: tuple[int, ...] = K_VALUES) -> None:
    if not rows:
        raise ValueError("At least one evaluation question is required.")
    for k in k_values:
        if k <= 0:
            raise ValueError("Every k value must be greater than zero.")
        hits = 0
        reciprocal_rank = 0.0
        for row in rows:
            results = retrieve(row["question"], k=k)
            sources = [metadata["source"] for _, metadata in results]
            expected_source = row["source"].removesuffix(".pdf")
            if expected_source in sources:
                hits += 1
                reciprocal_rank += 1 / (sources.index(expected_source) + 1)

        lower, upper = wilson(hits, len(rows))
        print(
            f"k={k}: hit@k={hits / len(rows):.2f} "
            f"[{lower:.2f}-{upper:.2f}]  MRR={reciprocal_rank / len(rows):.2f}"
        )


if __name__ == "__main__":
    evaluate(load_questions())