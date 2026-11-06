"""Processing limits shared by spatial intake and hydrologic execution."""

DEFAULT_MAXIMUM_PROCESSING_CELLS = 8_000_000


def cell_budget(value=DEFAULT_MAXIMUM_PROCESSING_CELLS):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("DEM 像元预算必须为正整数")
    return value


def require_cells(cells, maximum, label="DEM"):
    maximum = cell_budget(maximum)
    if cells > maximum:
        raise ValueError(f"{label}需要{cells:,}像元，超过{maximum:,}像元预算；"
                         "请缩小选区、分区计算或提高明确配置的预算")
