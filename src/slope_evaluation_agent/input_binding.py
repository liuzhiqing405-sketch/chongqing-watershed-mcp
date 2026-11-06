"""Bind a displayed plan and a submitted run to their actual task inputs."""
from __future__ import annotations

import hashlib
import json


def input_signature(session) -> str:
    # Exclude conversational state, status polling and rendered products: those
    # may change without changing the physical computation being referenced.
    fields = ("review_id", "assessment_mode", "study_area_coordinates",
              "rainfall_request", "rainfall_package_id", "model_input_package_id",
              "hydro_input_id", "parameter_preset", "parameter_overrides")
    value = {key: getattr(session, key, None) for key in fields}
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
