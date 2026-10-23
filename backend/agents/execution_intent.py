"""Recognize explicit requests to execute generated code, before adding context."""

import re


def requests_sandbox_execution(text: str) -> bool:
    text = text.lower().replace("’", "'")
    # Code-only requests and explicit prohibitions must not execute anything.
    if re.search(
        r"\b(?:do not|don't|never|without)\s+(?:\w+\s+){0,3}"
        r"(?:run|running|runs|execute|executing|executed|execution)\b"
        r"|\b(?:only|just)\s+(?:write|generate|create|show)\b"
        r"|\b(?:code|script)\s+only\b",
        text,
    ):
        return False
    if re.match(r"\s*(?:what\b|explain\b|define\b|how\s+(?:does|do|to)\b)", text):
        return False
    return bool(re.search(r"\b(?:run|runs|running|execute|executes|executed|executing|execution|docker|sandbox)\b", text))
