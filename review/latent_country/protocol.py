"""Deterministic prompt manifests, independent of models and model outcomes.

Prompt formatting and historical seed replay are adapted from
ICL_prompt_generator.py and rank_diff_many_kv_groups.py. Those files are not
imported: their import-time behavior is deliberately outside this workflow.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIELDS = {"city": "cities", "university": "universities", "landmark": "landmarks"}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def stable_seed(*parts: Any) -> int:
    return int(digest(parts)[:16], 16)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    p = cfg["protocol"]
    if len(p["languages"]) != len(set(p["languages"])) or "en" not in p["languages"]:
        raise ValueError("Languages must be unique and include en.")
    if p["language_pairs"] not in {"english_and_monolingual", "all"}:
        raise ValueError("Unknown language_pairs setting.")
    for key in ("localization_pairs", "evaluation_pairs", "num_examples"):
        if not isinstance(p[key], int) or p[key] < 1:
            raise ValueError(f"protocol.{key} must be a positive integer.")
    if set(p["donor_sources"]) - {"university", "landmark"} or not p["donor_sources"]:
        raise ValueError("donor_sources must contain university and/or landmark.")
    if len(set(p["donor_sources"])) != len(p["donor_sources"]):
        raise ValueError("Duplicate donor source.")
    if cfg["evaluation"]["random_draws"] < 1 or cfg["evaluation"]["bootstrap_resamples"] < 1:
        raise ValueError("Random draws and bootstrap resamples must be positive.")
    if cfg["inference"]["batch_size"] < 1:
        raise ValueError("batch_size must be positive.")
    return cfg


def load_data(directory: Path, languages: list[str]) -> dict:
    data = {lang: json.loads((directory / f"{lang}_data.json").read_text()) for lang in languages}
    reference = data["en"]
    for lang, countries in data.items():
        if set(countries) != set(reference):
            raise ValueError(f"Country keys are not aligned in {lang}.")
        for country, record in countries.items():
            for field in ("capital", "country_code"):
                if not isinstance(record[field], str) or not record[field].strip():
                    raise ValueError(f"Invalid {lang}/{country}/{field}.")
            for field in FIELDS.values():
                if len(record[field]) != len(reference[country][field]):
                    raise ValueError(f"Entity lists are not index-aligned: {lang}/{country}/{field}.")
                if not record[field] or any(not isinstance(v, str) or not v.strip() for v in record[field]):
                    raise ValueError(f"Empty entity in {lang}/{country}/{field}.")
    return data


def render(plan: list, source: str, target: str, data: dict, language: str) -> tuple[str, str, str]:
    sep, suffix = ("، ", "،") if language == "ar" else (", ", ",")
    pairs = [(data[c][FIELDS[source]][i], data[c][target]) for c, i in plan]
    examples = ". ".join(f"{entity}{sep}{answer}" for entity, answer in pairs[:-1])
    prompt = (examples + ". " if examples else "") + pairs[-1][0] + suffix
    return prompt, plan[-1][0], " " + pairs[-1][1]


def historical_prompt(data: dict, language: str, source: str, target: str,
                      seed: int, n_examples: int) -> tuple[str, str, str]:
    # Preserve the old generator's insertion order and duplicate-string handling.
    unique = {}
    for country, record in data.items():
        for index, text in enumerate(record[FIELDS[source]]):
            unique.setdefault(text, (country, index))
    plan = list(unique.values())
    random.Random(seed).shuffle(plan)
    if len(plan) < n_examples + 1:
        raise ValueError("Not enough unique entities for historical prompt replay.")
    return render(plan[:n_examples + 1], source, target, data, language)


def language_pairs(settings: dict) -> list[tuple[str, str]]:
    return [(donor, recipient) for donor in settings["languages"]
            for recipient in settings["languages"]
            if settings["language_pairs"] == "all" or donor == recipient or "en" in (donor, recipient)]


def historical_hashes(data: dict, settings: dict) -> set[str]:
    """Replay the published 100-pair / 20-demonstration seed schedule.

    Replay every ordered language pair, a superset of the published conditions.
    This is a precise exclusion for that schedule, not a claim about arbitrary
    historical experiments elsewhere in the workspace.
    """
    history = settings["historical_exclusion"]
    n, examples = history["num_pairs"], history["num_examples"]
    hashes = set()
    for recipient in data:
        for i in range(n):
            prompt, country, _ = historical_prompt(data[recipient], recipient, "city", "capital", i, examples)
            hashes.add(digest(prompt))
            for donor in data:
                for source in ("university", "landmark"):
                    for attempt in range(10000):
                        alt, alt_country, _ = historical_prompt(data[donor], donor, source, "country_code",
                                                               2 * n - i + attempt, examples)
                        if alt_country != country:
                            hashes.add(digest(alt))
                            break
                    else:
                        raise ValueError("Historical replay could not find distinct countries.")
    return hashes


def make_plan(data: dict, source: str, count: int, seed: int) -> list:
    # The same semantic entity indices are rendered in every language.
    entries = [(c, i) for c, record in data["en"].items() for i in range(len(record[FIELDS[source]]))]
    random.Random(seed).shuffle(entries)
    chosen, used = [], {lang: set() for lang in data}
    for country, index in entries:
        values = {lang: d[country][FIELDS[source]][index] for lang, d in data.items()}
        if any(value in used[lang] for lang, value in values.items()):
            continue
        chosen.append([country, index])
        for lang, value in values.items():
            used[lang].add(value)
        if len(chosen) == count:
            return chosen
    raise ValueError("Not enough unique entities in every language for the requested prompt length.")


def build_protocol(settings: dict, data_dir: Path) -> dict:
    data = load_data(data_dir, settings["languages"])
    forbidden = historical_hashes(data, settings)
    legacy_count = len(forbidden)
    rows = {"localization": [], "evaluation": []}
    plans = {"localization": [], "evaluation": []}
    for split, n in (("localization", settings["localization_pairs"]), ("evaluation", settings["evaluation_pairs"])):
        for pair_index in range(n):
            accepted = {}
            for source in ["city"] + settings["donor_sources"]:
                for attempt in range(10000):
                    seed = stable_seed(settings["seed"], split, pair_index, source, attempt)
                    plan = make_plan(data, source, settings["num_examples"] + 1, seed)
                    if source != "city" and plan[-1][0] == accepted["city"]["plan"][-1][0]:
                        continue
                    target = "capital" if source == "city" else "country_code"
                    prompts = {lang: render(plan, source, target, d, lang) for lang, d in data.items()}
                    hashes = {digest(value[0]) for value in prompts.values()}
                    if not (hashes & forbidden):
                        forbidden.update(hashes)
                        accepted[source] = {"seed": seed, "plan": plan, "prompts": prompts}
                        break
                else:
                    raise ValueError("Could not construct disjoint prompts; increase entity pool or prompt length.")
            plans[split].append({"pair_index": pair_index, "sources": {
                source: {"seed": value["seed"], "entities": value["plan"]} for source, value in accepted.items()}})
            pairs = [("en", "en")] if split == "localization" else language_pairs(settings)
            for source in settings["donor_sources"]:
                for donor, recipient in pairs:
                    rec_prompt, rec_country, rec_answer = accepted["city"]["prompts"][recipient]
                    donor_prompt, donor_country, donor_answer = accepted[source]["prompts"][donor]
                    condition = f"{donor}_to_{recipient}__{source}"
                    rows[split].append({
                        "id": f"{condition}__{pair_index:05d}", "condition": condition,
                        "pair_index": pair_index, "donor_language": donor, "recipient_language": recipient,
                        "donor_source": source, "recipient_prompt": rec_prompt, "donor_prompt": donor_prompt,
                        "recipient_country": rec_country, "donor_country": donor_country,
                        "recipient_answer": rec_answer, "donor_answer": donor_answer,
                        "type_corrected_answer": " " + data[recipient][donor_country]["capital"],
                    })
    payload = {"version": 1, "settings": settings,
               "data_sha256": {lang: hashlib.sha256((data_dir / f"{lang}_data.json").read_bytes()).hexdigest()
                               for lang in data},
               "audit": {"historical_prompt_hashes_excluded": legacy_count,
                         "split_unit": "prompt (including all language renderings)",
                         "countries_held_out": False, "entity_alignment": "country key and source-list index"},
               "plans": plans, **rows}
    payload["sha256"] = digest(payload)
    validate_protocol(payload)
    return payload


def validate_protocol(payload: dict) -> None:
    if digest({k: v for k, v in payload.items() if k != "sha256"}) != payload["sha256"]:
        raise ValueError("Protocol content does not match its fingerprint.")
    split_hashes = []
    for split in ("localization", "evaluation"):
        rows = payload[split]
        if not rows or len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Empty or duplicate records in {split}.")
        hashes = set()
        for row in rows:
            if row["recipient_country"] == row["donor_country"]:
                raise ValueError("Donor and recipient must identify different countries.")
            if split == "localization" and (row["donor_language"], row["recipient_language"]) != ("en", "en"):
                raise ValueError("Localization must use English only.")
            hashes.update(digest(row[key]) for key in ("recipient_prompt", "donor_prompt"))
        split_hashes.append(hashes)
    if split_hashes[0] & split_hashes[1]:
        raise ValueError("Localization/evaluation prompt overlap.")


def load_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text())
    validate_protocol(payload)
    return payload
