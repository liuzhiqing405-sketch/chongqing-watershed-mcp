from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .settings import SETTINGS, Settings


ID_PATTERN = re.compile(r"^[0-9A-Za-z_-]{6,80}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_id(value: str) -> str:
    if not ID_PATTERN.fullmatch(value):
        raise ValueError("编号格式无效")
    return value


def review_dir(review_id: str, settings: Settings = SETTINGS) -> Path:
    return settings.runs_root / "watersheds" / safe_id(review_id)


def rainfall_dir(package_id: str, settings: Settings = SETTINGS) -> Path:
    return settings.runs_root / "rainfall" / safe_id(package_id)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"记录不存在：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def require_confirmed(review_id: str, settings: Settings = SETTINGS) -> dict[str, Any]:
    record = read_json(review_dir(review_id, settings) / "review.json")
    if record.get("status") != "confirmed":
        raise ValueError("该流域尚未人工确认，不能构造降雨或准备模型输入")
    require_spatial_quality(record)
    return record


def require_spatial_quality(record: dict[str, Any]) -> None:
    """Historical proximity-only checks cannot certify named-river channels."""
    qa = record.get("qa") or {}
    if qa.get("passed") is False or any(
            (qa.get(field) or {}).get("passed") is False
            for field in ("river_check","channel_check","coverage_check")):
        raise ValueError("空间QA未通过，不能确认或继续计算；请核对空间数据后重新划分")
    if record.get("selection_mode") == "drawn_study_area":
        from watershed_exchange.domain_policy import drawn_modeling_allowed
        if not drawn_modeling_allowed(qa,record.get("assessment_mode")):
            raise ValueError("该手绘范围的空间QA或水文合理性建模门槛未通过，请重新提交边界后再确认或计算")
    elif (qa.get("hydrologic_check") or {}).get("passed") is False:
        raise ValueError("空间QA未通过，不能确认或继续计算")
    if ((qa.get("river_check") or {}).get("checked") is True
            and not (qa.get("channel_check") or {}).get("checked")):
        raise ValueError("该旧划分仅检查了距河距离，尚未核对目标汇流河道；请重新划分后再确认或计算")
