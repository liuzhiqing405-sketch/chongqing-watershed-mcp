"""Extract station preferences; the dialogue model still decides question/task."""
from __future__ import annotations

import re


NOUN = r"(?:虚拟\s*)?(?:雨量站|雨站|降雨站)"
NUMBER = r"(?:-?\d+(?:\.\d+)?|[负零〇一二两三四五六七八九十百点]+)"


def chinese_integer(text: str) -> int:
    if re.fullmatch(r"\d+", text):
        return int(text)
    digits = dict(zip("零〇一二两三四五六七八九", [0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9]))
    if text in {"一百", "百"}:
        return 100
    if "十" in text:
        parts = text.split("十")
        if len(parts) == 2 and (not parts[0] or parts[0] in digits) and (not parts[1] or parts[1] in digits):
            return (digits.get(parts[0], 1) * 10) + digits.get(parts[1], 0)
    if text in digits:
        return digits[text]
    raise ValueError("不是有效整数")


def parse_station_preference(text: str):
    matches = list(re.finditer(rf"(?P<count>{NUMBER})\s*(?:个|座|处)?\s*{NOUN}", text))
    matches += list(re.finditer(rf"{NOUN}(?:数量|数目|数)?\s*(?:改为|改成|设为|设置为|设置成|设立|布设|设置|用|为|是|:|：)?\s*(?P<count>{NUMBER})\s*(?:个|座|处)?", text))
    matches.sort(key=lambda match: match.start())
    auto = re.search(r"(?:按|根据|依据).{0,12}面积.{0,12}(?:自动|自行|判断|确定|决定|设置|布站|布设|选)|(?:恢复|改回|改成|使用|采用|切换到).{0,6}自动.{0,10}(?:布站|雨站|雨量站)|(?:雨站|雨量站).{0,10}(?:自动确定|自动选择)", text)
    if auto and not matches:
        return None, "auto", None
    if not matches:
        return None, None, None
    if len(matches) > 1 and re.search(r"还是|或者|或是", text):
        return None, None, "请明确一个虚拟雨站数量（1～100个），或者使用按面积自动布站。"
    match = matches[-1]
    try:
        value = chinese_integer(match.group("count"))
        if not 1 <= value <= 100:
            raise ValueError("数量越界")
    except ValueError:
        return None, None, "虚拟雨站数量需要是1～100的整数；也可以改用按面积自动布站。"
    # Explicit replacement at the end wins: "原来4个，按面积自动布站".
    if auto and auto.start() > match.end():
        return None, "auto", None
    return value, "manual", None


def adaptive_station_contract_error(rows) -> str | None:
    """Refuse a live server that would silently fall back to four stations."""
    tool = next((row for row in rows if row.name == "build_virtual_rainfall"), None)
    if tool is None:
        return None  # The required-tool check reports this separately.
    schema = tool.input_schema
    request = schema.get("properties", {}).get("request", {})
    if "$ref" in request:
        request = schema.get("$defs", {}).get(request["$ref"].rsplit("/", 1)[-1], {})
    field = request.get("properties", {}).get("station_count", {})
    choices = field.get("anyOf", [field])
    integer = next((item for item in choices if item.get("type") == "integer"), {})
    nullable = any(item.get("type") == "null" for item in choices)
    if (not nullable or field.get("default", "missing") is not None
            or integer.get("minimum") != 1 or integer.get("maximum") != 100):
        return "build_virtual_rainfall requires area-based automatic station counts and manual 1-100; restart the spatial MCP with current source"
    return None
