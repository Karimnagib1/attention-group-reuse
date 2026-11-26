from __future__ import annotations

from typing import Any, Dict, List


def build_prompt(example: Dict[str, Any], options: List[str], cfg: Dict[str, Any]) -> str:
    question_field = cfg["data"].get("question_field", "question")
    question = str(example[question_field]).strip()
    context_field = cfg["data"].get("context_field")
    context = ""
    if context_field:
        context = str(example.get(context_field, "")).strip()

    labels = cfg["eval"]["option_labels"]
    option_lines = []
    for label, option in zip(labels, options):
        option_lines.append(f"{label}. {option}")

    prompt_template = cfg["eval"]["prompt_template"]
    answer_prefix = cfg["eval"].get("answer_prefix", "Answer:")

    prompt = prompt_template.format(
        question=question,
        context=context,
        passage=context,
        options="\n".join(option_lines),
        answer_prefix=answer_prefix,
    )
    return prompt


def option_continuation(label: str, cfg: Dict[str, Any]) -> str:
    answer_sep = cfg["eval"].get("answer_sep", " ")
    continuation_suffix = cfg["eval"].get("continuation_suffix", "")
    return f"{answer_sep}{label}{continuation_suffix}"


def option_text_continuation(option_text: str, cfg: Dict[str, Any]) -> str:
    answer_sep = cfg["eval"].get("answer_sep", " ")
    continuation_suffix = cfg["eval"].get("continuation_suffix", "")
    return f"{answer_sep}{option_text}{continuation_suffix}"
