"""Reading vector data whose attribute encoding is not declared consistently.

The managed layers differ: some carry a ``.cpg`` sidecar naming their encoding,
others rely on the system default. Resolving this in one place keeps every
reader from guessing differently.
"""

from __future__ import annotations

from pathlib import Path

import shapefile


#: Used when a Shapefile has no ``.cpg`` sidecar, which is the case for the
#: managed Chongqing river network and administrative boundary.
FALLBACK_ATTRIBUTE_ENCODING = "gbk"

#: Encodings tried, in order, when reading an archive whose encoding is unknown.
CANDIDATE_ENCODINGS = ("utf-8", "gbk", "latin1")


def attribute_encoding(path: Path) -> str:
    """Encoding to read ``path``'s attributes with, from its ``.cpg`` if present."""
    sidecar = Path(path).with_suffix(".cpg")
    if sidecar.is_file():
        declared = sidecar.read_text(encoding="ascii", errors="ignore").strip().splitlines()
        if declared and declared[0].strip():
            name = declared[0].strip().lower().replace("_", "-")
            aliases = {
                "utf8": "utf-8", "utf-8": "utf-8", "65001": "utf-8",
                "gbk": "gbk", "gb2312": "gbk", "gb18030": "gbk", "936": "gbk",
                "ansi": "gbk",
                "latin1": "latin1", "iso-8859-1": "latin1", "1252": "cp1252",
            }
            return aliases.get(name, name)
    return FALLBACK_ATTRIBUTE_ENCODING


def open_reader(path: Path, encoding: str | None = None) -> shapefile.Reader:
    """Open a Shapefile with the encoding its sidecar declares."""
    return shapefile.Reader(str(path), encoding=encoding or attribute_encoding(path))


def open_reader_bytes(shp_bytes: bytes | None, dbf_bytes: bytes) -> shapefile.Reader:
    """Open an in-memory Shapefile, trying each candidate encoding."""
    import io

    last_error: Exception | None = None
    for encoding in CANDIDATE_ENCODINGS:
        try:
            reader = shapefile.Reader(
                shp=io.BytesIO(shp_bytes) if shp_bytes else None,
                dbf=io.BytesIO(dbf_bytes),
                encoding=encoding,
            )
            reader.fields  # force the header to be parsed
            return reader
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            last_error = exc
    raise ValueError(f"无法解析 Shapefile 属性表：{last_error}")
