from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import requests


OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"


@dataclass(frozen=True)
class OpenRouterCredits:
    total_credits: float
    total_usage: float
    raw_response: dict[str, Any]

    @property
    def remaining(self) -> float:
        return self.total_credits - self.total_usage


def get_openrouter_credits(
    api_key: str | None = None,
    *,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    timeout: float = 15,
) -> OpenRouterCredits:
    """Fetch OpenRouter credit totals for the authenticated management key."""
    resolved_api_key = api_key or os.environ.get(api_key_env)
    if not resolved_api_key:
        raise RuntimeError(f"Missing OpenRouter API key. Set {api_key_env} or pass api_key.")

    response = requests.get(
        OPENROUTER_CREDITS_URL,
        headers={"Authorization": f"Bearer {resolved_api_key}"},
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(
            f"OpenRouter credits request failed: {response.status_code} {response.text}"
        )

    payload = response.json()
    data = _require_mapping(payload.get("data"), "data")
    return OpenRouterCredits(
        total_credits=float(data["total_credits"]),
        total_usage=float(data["total_usage"]),
        raw_response=payload
    )


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"OpenRouter response missing object field: {name}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check remaining OpenRouter credits.")
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help=f"Environment variable containing the OpenRouter management key (default: {DEFAULT_API_KEY_ENV})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15,
        help="Request timeout in seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    credits = get_openrouter_credits(api_key_env=args.api_key_env, timeout=args.timeout)
    print(f"Total credits:     {credits.total_credits:.6f}")
    print(f"Total usage:       {credits.total_usage:.6f}")
    print(f"Remaining credits: {credits.remaining:.6f}")
    print(f"Raw response:      {credits.raw_response}")


if __name__ == "__main__":
    main()
