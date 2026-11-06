"""Turning a table into a workbook a person can open and check.

An ``.xlsx`` written here is a **derived convenience copy** of a CSV that the
run already produced. It is never the primary artifact, it never replaces the
file it was made from, and the first sheet says so — along with the digest of
the source, the timezone its timestamps are in, and what an empty cell means.

Two rules the workbook obeys:

* it is a real ``.xlsx`` written by a spreadsheet library, not a renamed CSV;
* it refuses rather than truncates. A table too large for a worksheet produces
  an error that names the CSV to download instead, because a workbook that
  silently drops the last hundred thousand rows is worse than no workbook.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .state import utc_now


#: The version of this converter, recorded on every workbook it produces so a
#: later change is visible in the metadata rather than silent.
CONVERTER_VERSION = "1.0"

#: Excel's own worksheet limit, and the bound this converter applies. The
#: service's bound is lower because a workbook nobody can open is not a
#: deliverable either.
EXCEL_ROW_LIMIT = 1_048_576
MAXIMUM_ROWS = 200_000
MAXIMUM_COLUMNS = 512
MAXIMUM_CELLS = 4_000_000

#: Datetime shapes our own tables use. Anything else is written as text, which
#: is lossless; guessing at an unfamiliar format would not be.
DATETIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d")

_SHEET_BAD = set('[]:*?/\\')


class WorkbookTooLarge(ValueError):
    """The table does not fit in a worksheet, and was not truncated."""

    def __init__(self, message: str, *, rows: int | None = None) -> None:
        super().__init__(message)
        self.rows = rows


def sheet_title(name: str, fallback: str = "数据") -> str:
    """A worksheet name Excel accepts: 31 characters, none of ``[]:*?/\\``."""
    cleaned = "".join(character for character in str(name) if character not in _SHEET_BAD)
    cleaned = cleaned.strip().strip("'") or fallback
    return cleaned[:31]


def _unique(columns: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for index, column in enumerate(columns):
        name = str(column).strip() or f"column_{index + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        result.append(name)
    return result


def _cell(value: str) -> Any:
    """A CSV cell as a spreadsheet value.

    Numbers become numbers so a reader can sum a column; the datetime shapes
    this service writes become real datetimes so a reader can plot a series.
    Everything else stays text, which loses nothing.
    """
    text = value.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "n/a", "na", "null", "none", "-"}:
        return None
    for shape in DATETIME_FORMATS:
        try:
            return datetime.strptime(text, shape)
        except ValueError:
            continue
    try:
        number = float(text)
    except ValueError:
        return text
    # Keep identifiers as text: a run id that looks like a number is still a name.
    if "." not in text and "e" not in lowered and not (-10**15 < number < 10**15):
        return text
    return int(number) if number.is_integer() and "." not in text else number


def read_table(path: Path) -> tuple[list[str], list[list[Any]], int, bool]:
    """The header and rows of a delimited file, plus how many rows there were.

    Returns ``(columns, rows, total_rows, complete)``. ``complete`` is false when
    reading stopped at the row bound, which is what makes the refusal honest: the
    caller can say "there are more than N rows" instead of guessing a total.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            return [], [], 0, True
        columns = _unique(header)
        rows: list[list[Any]] = []
        total = 0
        for total, raw in enumerate(reader, start=1):
            if len(rows) < MAXIMUM_ROWS:
                rows.append([_cell(item) if index < len(columns) else None
                             for index, item in enumerate(raw[:len(columns)])]
                            + [None] * max(0, len(columns) - len(raw)))
        return columns, rows, total, total <= MAXIMUM_ROWS


def build_workbook(source: Path, destination: Path, *,
                   metadata: dict[str, Any],
                   timezone: str = "Asia/Shanghai") -> dict[str, Any]:
    """Write ``source`` as a real workbook at ``destination``.

    The first sheet is the provenance: what this file was derived from, by what,
    when, with which digest — because a workbook that circulates without that is
    a table of unknown origin.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font

    columns, rows, total, complete = read_table(source)
    if not columns:
        raise WorkbookTooLarge("源表为空，没有可导出的表头")
    if total > MAXIMUM_ROWS:
        raise WorkbookTooLarge(
            f"表格有 {total} 行，超过工作簿上限 {MAXIMUM_ROWS} 行；"
            "请改下载 CSV（xlsx 只是便于查看的派生件）", rows=total)
    if len(columns) > MAXIMUM_COLUMNS:
        raise WorkbookTooLarge(f"表格有 {len(columns)} 列，超过上限 {MAXIMUM_COLUMNS} 列")
    if len(columns) * max(len(rows), 1) > MAXIMUM_CELLS:
        raise WorkbookTooLarge(
            f"表格单元格数超过上限 {MAXIMUM_CELLS}；请改下载 CSV")

    workbook = Workbook(write_only=True)
    meta_sheet = workbook.create_sheet(title="元数据")
    for key, value in _provenance(metadata, source, columns, rows, total,
                                  timezone).items():
        meta_sheet.append([key, value])

    data_sheet = workbook.create_sheet(title=sheet_title(source.stem))
    # Both set *before* the first row: a write-only sheet streams its header as
    # soon as it starts writing, so anything set afterwards is silently dropped.
    data_sheet.freeze_panes = "A2"
    data_sheet.auto_filter.ref = f"A1:{_column_name(len(columns))}1"
    data_sheet.append(list(columns))
    for row in rows:
        data_sheet.append(row)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        workbook.save(temporary)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "columns": len(columns),
        "rows": len(rows),
        "source_rows": total,
        "complete": complete,
    }


def _column_name(index: int) -> str:
    """The spreadsheet column letter for a 1-based index."""
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _provenance(metadata: dict[str, Any], source: Path, columns: list[str],
                rows: list[list[Any]], total: int,
                timezone: str) -> dict[str, Any]:
    """The metadata sheet, in the order a reader would ask the questions."""
    empty_meaning = ("空单元格对应源文件中的空值、NaN 或 NoData；"
                     "数值列可直接参与计算，文本列保持文本")
    return {
        "文件性质": "派生文件：由运行成果转换而来，用于查看与人工复核，"
                    "不是计算结果的原始载体",
        "转换器": f"watershed_hydro_mcp.workbook {CONVERTER_VERSION}",
        "生成时间": utc_now(),
        "运行编号": metadata.get("run_id"),
        "可视化编号": metadata.get("visualization_id") or "—",
        "来源成果编号": metadata.get("artifact_id"),
        "来源文件": metadata.get("source_title"),
        "来源 SHA-256": metadata.get("source_sha256"),
        "QA 状态": metadata.get("qa_status") or "未知（运行尚未完成质量门槛检查）",
        "单位": metadata.get("units") or "见源文件列名（多列多单位）",
        "CRS": metadata.get("crs") or "不适用",
        "时间列时区": timezone,
        "表格规模": f"列 {len(columns)}，行 {len(rows)}"
                    + ("" if total <= len(rows) else f"（源文件共 {total} 行，超出部分未导出）"),
        "空值与 NoData": empty_meaning,
        "适用边界": "本表只反映该次运行的模型输出，"
                    "不能作为工程级边坡安全鉴定或灾害预报的依据",
    }
