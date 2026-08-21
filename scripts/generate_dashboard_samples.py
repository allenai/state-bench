"""Build the deterministic sample pack used by the local inspection dashboard."""

from __future__ import annotations

import json
import random
from pathlib import Path

from state_bench.config import FillerDistraction, load_formatter
from state_bench.distractors import generate_incorrect_states
from state_bench.generation import derive_instance, generate_periodic_source
from state_bench.io import derive_seed
from state_bench.model import Complexity
from state_bench.rendering import render_instance

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dashboard/public/samples.json"
FORMATTER_PATHS = tuple(sorted((ROOT / "configs/formats").glob("*.yaml")))
PADDING_VARIANTS: tuple[tuple[float, FillerDistraction | None], ...] = (
    (0.0, None),
    (0.25, "easy"),
    (0.25, "medium"),
    (0.25, "hard"),
    (0.5, "easy"),
    (0.5, "medium"),
    (0.5, "hard"),
    (0.75, "easy"),
    (0.75, "medium"),
    (0.75, "hard"),
    (0.9, "easy"),
    (0.9, "medium"),
    (0.9, "hard"),
)
NUM_DISTRACTORS = 4


def main() -> None:
    formatters = tuple(load_formatter(path) for path in FORMATTER_PATHS)
    samples: list[dict[str, object]] = []
    for sample_index in range(2):
        source = generate_periodic_source(
            num_variables=3,
            num_values=6,
            min_extra_assignments=4,
            max_extra_assignments=4,
            min_aperiodic_constants=0,
            max_aperiodic_constants=0,
            rng=random.Random(derive_seed(29, "dashboard-source", str(sample_index))),
        )
        for proportion, distraction in PADDING_VARIANTS:
            for complexity in Complexity:
                instance = derive_instance(source, complexity=complexity)
                incorrect_states = generate_incorrect_states(
                    instance,
                    num_distractors=NUM_DISTRACTORS,
                    rng=random.Random(
                        derive_seed(
                            29,
                            "dashboard-distractors",
                            str(sample_index),
                            complexity.value,
                            str(proportion),
                            distraction or "none",
                        )
                    ),
                )
                for formatter in formatters:
                    rendered = render_instance(
                        instance,
                        formatter,
                        rng=random.Random(
                            derive_seed(
                                29,
                                "dashboard-surface",
                                str(sample_index),
                                formatter.name,
                                str(proportion),
                                distraction or "none",
                            )
                        ),
                        incorrect_states=incorrect_states,
                        shuffle_choices=True,
                        filler_proportion=proportion,
                        filler_distraction=distraction,
                    )
                    rendered_record = rendered.to_record()
                    metadata = {
                        "sample_id": (
                            f"{sample_index}-{formatter.name}-{complexity.value}-"
                            f"{distraction or 'none'}-{proportion:g}"
                        ),
                        "sample_index": sample_index + 1,
                        "source_instance_id": source.instance_id,
                        "instance_id": instance.instance_id,
                        "query": list(instance.query),
                        "final_state": list(instance.final_state),
                        **{
                            key: rendered_record[key]
                            for key in (
                                "formatter",
                                "formatter_config_hash",
                                "complexity",
                                "num_variables",
                                "num_values",
                                "surface_variables",
                                "surface_values",
                                "num_assignments",
                                "incorrect_completions",
                                "choices",
                                "correct_choice_index",
                                "filler_proportion",
                                "actual_filler_proportion",
                                "filler_distraction",
                                "num_filler_lines",
                                "final_dependency_lengths",
                                "max_dependency_length",
                                "mean_dependency_length",
                                "modal_dependency_length",
                            )
                        },
                    }
                    samples.append(
                        {
                            "id": (
                                f"{sample_index}-{formatter.name}-{complexity.value}-"
                                f"{distraction or 'none'}-{proportion:g}"
                            ),
                            "sampleIndex": sample_index + 1,
                            "sourceInstanceId": source.instance_id,
                            "formatter": formatter.name,
                            "complexity": complexity.value,
                            "paddingComplexity": distraction or "none",
                            "paddingPercent": round(proportion * 100),
                            "sourceOperations": [
                                operation.to_record() for operation in source.operations
                            ],
                            "operations": [operation.to_record() for operation in instance.operations],
                            "query": list(instance.query),
                            "finalState": list(instance.final_state),
                            "prefix": rendered.prefix,
                            "completion": rendered.completion,
                            "numFillerLines": rendered.num_filler_lines,
                            "metadata": metadata,
                        }
                    )

    payload = {
        "schemaVersion": 1,
        "samplesPerCondition": 2,
        "samples": samples,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(samples)} samples to {OUTPUT}")


if __name__ == "__main__":
    main()
