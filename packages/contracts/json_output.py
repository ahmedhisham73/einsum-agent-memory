"""Parse model JSON without accepting unrelated surrounding prose."""

import json
import re


def parse_model_json(content):
    if not isinstance(content, str):
        raise ValueError("Model content must be text")

    value = content.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*\n?(.*?)\s*```",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        value = fenced.group(1).strip()

    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a model JSON object")
    return parsed
