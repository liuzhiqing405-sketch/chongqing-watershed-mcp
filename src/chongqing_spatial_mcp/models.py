from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RainfallRequest(BaseModel):
    review_id: str = Field(description="已人工确认的流域审核编号")
    mode: Literal["historical_max", "explicit_period"]
    station_count: int | None = Field(default=None, ge=1, le=100, strict=True,
                                    description="不指定时按选区面积自动布站；用户指定1到100个")
    lookback_years: int = Field(default=30, ge=1, le=100)
    end_year: int | None = Field(default=None, ge=1900, le=2200)
    date: str | None = Field(default=None, description="全天日期，YYYY-MM-DD")
    start_datetime: str | None = Field(default=None, description="YYYY-MM-DD HH:MM")
    end_datetime: str | None = Field(default=None, description="闭区间，YYYY-MM-DD HH:MM")
    source_timezone: str = "Asia/Shanghai"

    @model_validator(mode="after")
    def validate_mode_fields(self) -> "RainfallRequest":
        if self.mode == "historical_max":
            return self
        if self.date and (self.start_datetime or self.end_datetime):
            raise ValueError("date 与 start_datetime/end_datetime 不能同时提供")
        if not self.date and not (self.start_datetime and self.end_datetime):
            raise ValueError("explicit_period 需要 date，或同时提供开始和结束时间")
        if bool(self.start_datetime) != bool(self.end_datetime):
            raise ValueError("开始和结束时间必须同时提供")
        if self.date:
            datetime.strptime(self.date, "%Y-%m-%d")
        else:
            start = datetime.fromisoformat(str(self.start_datetime))
            end = datetime.fromisoformat(str(self.end_datetime))
            if end < start:
                raise ValueError("结束时间不能早于开始时间")
            if (start.year, start.month) != (end.year, end.month):
                raise ValueError("永川雨站字段格式要求单次事件位于同一自然月")
            hours = int((end - start).total_seconds() // 3600) + 1
            if hours > 247:
                raise ValueError("事件超过永川站点DBF支持的247个时次")
        return self
