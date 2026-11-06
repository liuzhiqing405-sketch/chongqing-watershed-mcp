"""Model geographic knowledge proposes a map centre, never a chosen outlet."""

import json
import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GeographicLocation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    status: Literal["resolved", "ambiguous", "not_found", "unavailable"]
    region: str = Field(default="", max_length=200)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    confidence: float = Field(default=0, ge=0, le=1)
    description: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def consistent(self):
        if self.status == "resolved" and (self.longitude is None or self.latitude is None):
            raise ValueError("resolved needs both coordinates")
        if self.status != "resolved" and (self.longitude is not None or self.latitude is not None):
            raise ValueError("an unresolved place cannot carry a guessed point")
        return self


class GeographicLocator:
    SYSTEM_PROMPT = (
        "你是地理定位助手。本地水系名称检索失败，请根据通用地理知识和用户上下文定位这条河。"
        "项目基础数据位于重庆及附近，但它不证明用户说的河一定在重庆。优先采用用户明确的区县、乡镇或地标；"
        "如果同名河流无法区分或不记得可靠位置，返回 ambiguous 或 not_found，不能随意猜一个点。"
        "resolved 的经纬度是 WGS84 河段附近的粗略地图中心，最多保留四位小数，"
        "不是实测坐标、不是已确定出水口，也不能声称已检索网络、确认流域或执行计算。"
        "只能给一个位置，不要推荐地点列表。用户会在地图上选实际出水口。"
        "只输出 JSON，字段 status（resolved/ambiguous/not_found）、region、longitude、latitude、"
        "confidence（0到1）、description。未定位时经纬度为 null。"
        '格式示例：{"status":"ambiguous","region":"","longitude":null,"latitude":null,'
        '"confidence":0.2,"description":"需要所在区县以区分同名河流"}。'
    )

    def __init__(self, llm=None, logger=None):
        self.llm = llm
        self.logger = logger or logging.getLogger("slope_agent")

    async def resolve(self, river_name, session):
        if self.llm is None:
            return GeographicLocation(status="unavailable", description="模型未配置，无法补充地理定位。")
        try:
            payload = await self.llm.complete_json(
                system=self.SYSTEM_PROMPT,
                user=json.dumps({"river_name": river_name, "user_context": [
                    turn.text for turn in session.recent_conversation(8) if turn.role == "user"
                ]}, ensure_ascii=False))
            result = GeographicLocation.model_validate(payload)
            if result.status == "resolved" and result.confidence < 0.8:
                return GeographicLocation(status="ambiguous", region=result.region,
                                          description="模型对位置把握不足，需要所在区县或附近地标。")
            if result.status == "resolved":
                result = result.model_copy(update={"longitude": round(result.longitude, 4),
                                                   "latitude": round(result.latitude, 4)})
            return result
        except Exception as exc:
            self.logger.info("地理定位未成功：%s", getattr(exc, "reason", type(exc).__name__))
            return GeographicLocation(status="unavailable", description="本轮模型定位未成功。")
