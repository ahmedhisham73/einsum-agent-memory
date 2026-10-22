"""Normalize Chat Completions usage into our shared ledger."""

from typing import Optional

from packages.contracts.usage import TokenUsage
from packages.telemetry.token_ledger import TokenLedger


def record_completion_usage(
    response: dict,
    *,
    ledger: TokenLedger,
    call_id: str,
    experiment_id: str,
    approach: str,
    phase: str,
    stage: str,
    provider: str,
    requested_model: str,
    query_id: Optional[str] = None,
    attempt: int = 1,
) -> TokenUsage:
    """Record reported usage without estimating missing token counts."""
    usage = response.get("usage")
    if usage is None:
        usage = {}
    if not isinstance(usage, dict):
        raise ValueError("API usage must be an object")

    details = usage.get("prompt_tokens_details")
    if details is None:
        details = {}
    if not isinstance(details, dict):
        raise ValueError("prompt_tokens_details must be an object")

    record = TokenUsage(
        call_id=call_id,
        experiment_id=experiment_id,
        approach=approach,
        phase=phase,
        stage=stage,
        provider=provider,
        model=response.get("model") or requested_model,
        query_id=query_id,
        attempt=attempt,
        usage_kind="generation",
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        cached_input_tokens=details.get("cached_tokens"),
    )

    # Verify the provider's total when all components are reported.
    reported_total = usage.get("total_tokens")
    if reported_total is not None:
        if type(reported_total) is not int or reported_total < 0:
            raise ValueError("Invalid API total_tokens")
        if (
            record.total_tokens is not None
            and reported_total != record.total_tokens
        ):
            raise ValueError("API token total does not match its components")

    ledger.record(record)
    return record


import json
import time
import uuid
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def complete_chat(
    *,
    config,
    ledger: TokenLedger,
    messages: list,
    experiment_id: str,
    approach: str,
    phase: str,
    stage: str,
    provider: str,
    query_id: Optional[str] = None,
    max_completion_tokens: int = 512,
    attempt: int = 1,
    transport=None,
) -> dict:
    """Execute one non-streaming request. No automatic retries."""
    if not messages:
        raise ValueError("messages must not be empty")
    if type(max_completion_tokens) is not int or max_completion_tokens < 1:
        raise ValueError("max_completion_tokens must be a positive integer")

    call_id = str(uuid.uuid4())

    metadata = dict(
        ledger=ledger,
        call_id=call_id,
        experiment_id=experiment_id,
        approach=approach,
        phase=phase,
        stage=stage,
        provider=provider,
        requested_model=config.model,
        query_id=query_id,
        attempt=attempt,
    )

    # Validate attribution before making a potentially billable request.
    TokenUsage(
        call_id=call_id,
        experiment_id=experiment_id,
        approach=approach,
        phase=phase,
        stage=stage,
        provider=provider,
        model=config.model,
        query_id=query_id,
        attempt=attempt,
    )

    payload = {
        "model": config.model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "stream": False,
    }

    request = urllib.request.Request(
        url=config.base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + config.api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    if transport is None:
        transport = urllib.request.build_opener(NoRedirect()).open

    started = time.perf_counter()

    try:
        with transport(
            request, timeout=config.timeout_seconds
        ) as response:
            body = json.loads(response.read().decode("utf-8"))

        if not isinstance(body, dict):
            raise ValueError("Expected an API response object")
        if body.get("error") is not None:
            raise ValueError("API returned an error object")

        usage = record_completion_usage(body, **metadata)

    except Exception:
        # A failed request may still have consumed tokens.
        # Record unknown usage rather than assuming zero.
        record_completion_usage({}, **metadata)
        raise RuntimeError(
            f"LLM request failed; call_id={call_id}. Usage is unknown."
        ) from None

    return {
        "call_id": call_id,
        "response": body,
        "usage": usage,
        "request_latency_seconds": time.perf_counter() - started,
    }
