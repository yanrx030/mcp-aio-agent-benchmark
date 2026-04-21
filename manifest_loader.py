from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RoleModelConfig:
    model_id: str
    openrouter_params: dict[str, Any]


@dataclass(slots=True)
class RunnerManifestConfig:
    source_path: Path
    profile: str
    tool_agent: RoleModelConfig
    judge: RoleModelConfig | None


def resolve_runner_manifest(
    manifest_path: str | Path,
    *,
    profile: str = "default",
    tool_agent_model_override: str | None = None,
    judge_model_override: str | None = None,
) -> RunnerManifestConfig:
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise RuntimeError(f"Manifest file not found: {manifest_file}")

    manifest = _load_manifest(manifest_file)
    openrouter_defaults = _as_mapping(
        _as_mapping(manifest.get("openrouter"), "openrouter").get("defaults", {}),
        "openrouter.defaults",
    )
    model_catalog = _build_model_catalog(
        _as_list(manifest.get("model_catalog"), "model_catalog")
    )

    profiles = _as_mapping(manifest.get("profiles"), "profiles")
    if profile not in profiles:
        available = ", ".join(sorted(profiles.keys()))
        raise RuntimeError(
            f"Profile '{profile}' not found in manifest. Available profiles: {available}"
        )

    profile_cfg = _as_mapping(profiles[profile], f"profiles.{profile}")
    profile_openrouter_override = _as_mapping(
        profile_cfg.get("openrouter_override", {}),
        f"profiles.{profile}.openrouter_override",
    )

    tool_agent_model_id = tool_agent_model_override or _pick_model_from_profile(
        role="tool_agent",
        profile_cfg=profile_cfg,
        profile_name=profile,
        model_catalog=model_catalog,
    )
    judge_model_id = judge_model_override
    if judge_model_id is None:
        judge_model_id = _pick_model_from_profile(
            role="judge",
            profile_cfg=profile_cfg,
            profile_name=profile,
            model_catalog=model_catalog,
            allow_missing=True,
        )

    tool_agent = _build_role_model_config(
        model_id=tool_agent_model_id,
        role="tool_agent",
        model_catalog=model_catalog,
        openrouter_defaults=openrouter_defaults,
        profile_openrouter_override=profile_openrouter_override,
    )
    judge = (
        _build_role_model_config(
            model_id=judge_model_id,
            role="judge",
            model_catalog=model_catalog,
            openrouter_defaults=openrouter_defaults,
            profile_openrouter_override=profile_openrouter_override,
        )
        if judge_model_id
        else None
    )

    return RunnerManifestConfig(
        source_path=manifest_file.resolve(),
        profile=profile,
        tool_agent=tool_agent,
        judge=judge,
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Manifest root must be a mapping in {path}")
    return payload


def _build_model_catalog(catalog_entries: list[Any]) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}

    for index, entry in enumerate(catalog_entries, start=1):
        entry_path = f"model_catalog[{index}]"
        model = _as_mapping(entry, entry_path)
        model_id = model.get("id")
        role = model.get("role")

        if not isinstance(model_id, str) or not model_id.strip():
            raise RuntimeError(f"{entry_path}.id must be a non-empty string")
        if role not in {"tool_agent", "judge"}:
            raise RuntimeError(
                f"{entry_path}.role must be 'tool_agent' or 'judge', got: {role!r}"
            )
        if model_id in catalog:
            raise RuntimeError(f"Duplicate model id in model_catalog: {model_id}")

        openrouter_cfg = _as_mapping(model.get("openrouter", {}), f"{entry_path}.openrouter")
        provider_cfg = _as_mapping(openrouter_cfg.get("provider", {}), f"{entry_path}.openrouter.provider")
        provider_order = provider_cfg.get("order")
        if provider_order is not None:
            if not isinstance(provider_order, list):
                raise RuntimeError(f"{entry_path}.openrouter.provider.order must be a list")
            if len(provider_order) != 1 or not isinstance(provider_order[0], str):
                raise RuntimeError(
                    f"{entry_path}.openrouter.provider.order must contain exactly one provider string"
                )

        catalog[model_id] = {
            "id": model_id,
            "role": role,
            "enabled": bool(model.get("enabled", True)),
            "openrouter": openrouter_cfg,
        }

    return catalog


def _pick_model_from_profile(
    *,
    role: str,
    profile_cfg: dict[str, Any],
    profile_name: str,
    model_catalog: dict[str, dict[str, Any]],
    allow_missing: bool = False,
) -> str | None:
    key = "tool_agent_candidates" if role == "tool_agent" else "judge_candidates"
    candidates = _as_list(profile_cfg.get(key, []), f"profiles.{profile_name}.{key}")

    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            raise RuntimeError(f"profiles.{profile_name}.{key} must contain only non-empty strings")

        model_entry = model_catalog.get(candidate)
        if model_entry is None:
            raise RuntimeError(
                f"profiles.{profile_name}.{key} references unknown model id: {candidate}"
            )
        if model_entry["role"] != role:
            raise RuntimeError(
                f"Model '{candidate}' in profiles.{profile_name}.{key} has role "
                f"'{model_entry['role']}' but expected '{role}'"
            )
        if model_entry["enabled"]:
            return candidate

    if allow_missing:
        return None
    raise RuntimeError(
        f"profiles.{profile_name}.{key} has no enabled model to select"
    )


def _build_role_model_config(
    *,
    model_id: str,
    role: str,
    model_catalog: dict[str, dict[str, Any]],
    openrouter_defaults: dict[str, Any],
    profile_openrouter_override: dict[str, Any],
) -> RoleModelConfig:
    model_entry = model_catalog.get(model_id)
    if model_entry is None:
        raise RuntimeError(f"Model '{model_id}' not found in model_catalog")
    if model_entry["role"] != role:
        raise RuntimeError(
            f"Model '{model_id}' role mismatch: expected '{role}', got '{model_entry['role']}'"
        )
    if not model_entry["enabled"]:
        raise RuntimeError(f"Model '{model_id}' is disabled in model_catalog")

    merged = _deep_merge(openrouter_defaults, _as_mapping(model_entry.get("openrouter", {}), "model.openrouter"))
    merged = _deep_merge(merged, profile_openrouter_override)
    return RoleModelConfig(model_id=model_id, openrouter_params=merged)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


def _as_mapping(value: Any, field_path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError(f"{field_path} must be a mapping/object")
    return value


def _as_list(value: Any, field_path: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RuntimeError(f"{field_path} must be a list/array")
    return value
