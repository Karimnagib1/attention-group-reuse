"""
ICL Prompt Generator
--------------------
Creates in-context learning prompts from a country dataset.

Usage:
    from icl_prompt_generator import generate_icl_prompt, DATA
    prompt = generate_icl_prompt(
        source_type="city",
        target_type="capital",
        num_examples=3,
        seed=42
    )
    print(prompt)
"""
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import random
import json


DATA_PATH = Path(__file__).resolve().parent / "data" / "en_data.json"
with DATA_PATH.open("r", encoding="utf-8") as f:
    DATA = json.load(f)


ALLOWED_SOURCE_TYPES = {"city", "university", "landmark"}
ALLOWED_TARGET_TYPES = {"capital", "country calling code"}
ALLOWED_DIRECTIONS = {"ltr", "rtl"}

def generate_icl_prompt(
    source_type: str,
    target_type: str,
    num_examples: int = 3,
    dir: str = "ltr",
    seed: Optional[int] = None,
    dataset: Optional[Dict] = None,
) -> tuple[str, str, str]:
    """
    Create an in-context learning prompt like:
    'Source1, Target1. Source2, Target2. ... SourceN, TargetN. Source(N+1), '
    - source_type in {'city','university','landmark'}
    - target_type in {'capital','country calling code'}
    - num_examples controls how many (Source, Target) pairs before the final query.
    - No repeats; the last 'query' source is distinct from the examples.
    """
    if dataset is None:
        dataset = DATA

    if source_type not in ALLOWED_SOURCE_TYPES:
        raise ValueError(f"source_type must be one of {sorted(ALLOWED_SOURCE_TYPES)}")
    if target_type not in ALLOWED_TARGET_TYPES:
        raise ValueError(f"target_type must be one of {sorted(ALLOWED_TARGET_TYPES)}")
    if dir not in ALLOWED_DIRECTIONS:
        raise ValueError(f"dir must be one of {sorted(ALLOWED_DIRECTIONS)}")
    if num_examples < 0:
        raise ValueError("num_examples must be non-negative")

    rnd = random.Random(seed)

    # Collect all candidate sources by type, along with their country
    candidates: List[Tuple[str, str]] = []  # (source, country)
    plural_map = {"city": "cities", "university": "universities", "landmark": "landmarks"}
    for country, info in dataset.items():
        items = info[plural_map[source_type]]
        for item in items:
            candidates.append((item, country))

    # Ensure uniqueness of source strings
    unique_map: Dict[str, str] = {}
    for s, c in candidates:
        if s not in unique_map:
            unique_map[s] = c
    unique_items = list(unique_map.items())  # [(source, country), ...]

    needed = num_examples + 1  # examples + 1 query
    if len(unique_items) < needed:
        raise ValueError(
            f"Not enough unique {source_type} items to create {needed} entries. "
            f"Have {len(unique_items)}. Reduce num_examples or extend dataset."
        )

    rnd.shuffle(unique_items)
    picked = unique_items[:needed]

    def target_for(country: str) -> str:
        if target_type == "capital":
            return dataset[country]["capital"]
        elif target_type == "country calling code":
            return dataset[country]["country_code"]
        else:
            raise ValueError("Unsupported target type")

    pair_separator = ", " if dir == "ltr" else "، "
    query_suffix = "," if dir == "ltr" else "،"

    # Build examples
    examples = [
        f"{src}{pair_separator}{target_for(country)}"
        for src, country in picked[:num_examples]
    ]

    # Final query (source only)
    query_src, bridge = picked[num_examples]
    answer_src = " " + target_for(bridge)

    # Build prompt string
    if examples:
        prompt = ". ".join(examples) + ". " + f"{query_src}{query_suffix}"
    else:
        prompt = f"{query_src}{query_suffix}"

    return prompt, bridge, answer_src

if __name__ == "__main__":
    # Demo
    print(generate_icl_prompt("city", "capital", num_examples=3, seed=3, dir="rtl"))
    print(generate_icl_prompt("university", "country calling code", num_examples=3, seed=2, dir="rtl"))
    # from typing import Any, Literal
    # NUM_PAIRS = 100
    # normal_src, normal_tgt = "city", "capital"
    # alt_src, alt_tgt = "landmark", "country calling code"

    # def get_alt_answer(
    #     data: Any, bridge: str, target_type: Literal["capital", "country_code"]
    # ) -> str:
    #     # leading space for tokenization consistency
    #     return " " + data[bridge][target_type]
    # normal_prompts: List[str] = []
    # normal_answers: List[str] = []
    # alt_prompts: List[str] = []
    # alt_answers: List[str] = []

    # for i in range(NUM_PAIRS):
    #     normal_prompt, normal_bridge, normal_answer = generate_icl_prompt(
    #         normal_src, normal_tgt, num_examples=20, seed=i, dir = "rtl"
    #     )
    #     normal_prompts.append(normal_prompt)
    #     normal_answers.append(normal_answer)  # ensure leading space

    #     alt_prompt, alt_bridge, alt_answer = generate_icl_prompt(
    #         alt_src, alt_tgt, num_examples=20, seed=2 * NUM_PAIRS - i, dir = "rtl"
    #     )
    #     alt_prompts.append(alt_prompt)
    #     alt_answers.append(get_alt_answer(DATA, alt_bridge, "capital"))
