"""Path and archive safety for the exchange package.

Everything a client can influence (package ids, relative paths, archive
members) is funnelled through these helpers before it reaches the filesystem.
"""

from __future__ import annotations

import hashlib
import os
import re
import zipfile
from pathlib import Path


#: Identifiers the MCPs accept (review ids, package ids, run ids, artifact ids).
ID_PATTERN = re.compile(r"^[0-9A-Za-z_\-]{6,80}$")

#: Names Windows treats as devices; a path may never contain them.
RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}

#: Rejected wherever they appear in a package-relative path.
FORBIDDEN_DIRECTORY_NAMES = {"__pycache__", ".git", ".env", "node_modules"}

#: Default archive limits: an input package must not be a decompression bomb.
MAXIMUM_ARCHIVE_MEMBERS = 4096
MAXIMUM_SINGLE_MEMBER_BYTES = 512 * 1024 * 1024
MAXIMUM_TOTAL_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
MAXIMUM_COMPRESSION_RATIO = 200.0


class PathSafetyError(ValueError):
    """Raised when a client-supplied path could escape the managed directory."""


def safe_id(value: str, label: str = "编号") -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise PathSafetyError(f"{label}格式无效：只允许字母、数字、下划线和连字符，长度6到80")
    return value


def safe_relative_path(value: str) -> str:
    """Validate a package-relative path and return it normalised with ``/``."""
    if not isinstance(value, str) or not value.strip():
        raise PathSafetyError("相对路径不能为空")
    text = value.strip().replace("\\", "/")
    if text.startswith("/") or text.startswith("//"):
        raise PathSafetyError(f"不允许绝对路径：{value}")
    if re.match(r"^[A-Za-z]:", text):
        raise PathSafetyError(f"不允许盘符路径：{value}")
    if ":" in text:
        raise PathSafetyError(f"不允许包含冒号的路径：{value}")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts:
        raise PathSafetyError(f"相对路径为空：{value}")
    for part in parts:
        if part == "..":
            raise PathSafetyError(f"不允许路径穿越：{value}")
        stem = part.split(".")[0].lower()
        if stem in RESERVED_NAMES:
            raise PathSafetyError(f"不允许使用系统保留名称：{value}")
        if part.lower() in FORBIDDEN_DIRECTORY_NAMES:
            raise PathSafetyError(f"不允许的目录名：{value}")
        if part.rstrip() != part or part.endswith("."):
            raise PathSafetyError(f"路径组件含非法结尾空格或点：{value}")
    return "/".join(parts)


def resolve_within(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root``, refusing to leave ``root``.

    Rejects symlinks and junctions on the resolved path as well as ``..``
    traversal, so a pre-planted link cannot be used to read outside the run.
    """
    root = Path(root).resolve()
    candidate = (root / safe_relative_path(relative)).resolve()
    if candidate != root and root not in candidate.parents:
        raise PathSafetyError(f"路径超出受管目录：{relative}")
    return candidate


def contains_link(root: Path, relative: str) -> bool:
    """True when any component of the path is a symlink or reparse point."""
    root = Path(root).resolve()
    current = root
    for part in Path(safe_relative_path(relative)).parts:
        current = current / part
        if current.is_symlink():
            return True
        try:
            if os.name == "nt" and current.exists():
                attributes = os.lstat(current).st_file_attributes
                if attributes & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
                    return True
        except (OSError, AttributeError):
            pass
    return False


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_zip(
    archive: Path,
    destination: Path,
    *,
    maximum_members: int = MAXIMUM_ARCHIVE_MEMBERS,
    maximum_member_bytes: int = MAXIMUM_SINGLE_MEMBER_BYTES,
    maximum_total_bytes: int = MAXIMUM_TOTAL_UNCOMPRESSED_BYTES,
    maximum_ratio: float = MAXIMUM_COMPRESSION_RATIO,
) -> list[str]:
    """Extract a zip into ``destination`` after checking every member.

    Guards against ZIP Slip, absolute and drive-letter members, oversized
    archives, decompression bombs and link members. Returns the extracted
    relative paths.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        if len(members) > maximum_members:
            raise PathSafetyError(f"压缩包成员过多：{len(members)} > {maximum_members}")
        total = 0
        for member in members:
            name = member.filename
            if name.endswith("/"):
                continue
            relative = safe_relative_path(name)
            if member.file_size > maximum_member_bytes:
                raise PathSafetyError(f"压缩包内单文件过大：{relative}")
            total += member.file_size
            if total > maximum_total_bytes:
                raise PathSafetyError("压缩包解压总量超出上限")
            compressed = max(member.compress_size, 1)
            if member.file_size / compressed > maximum_ratio:
                raise PathSafetyError(f"压缩率异常，疑似压缩炸弹：{relative}")
            if (member.external_attr >> 16) & 0xA000 == 0xA000:
                raise PathSafetyError(f"压缩包内含符号链接：{relative}")
            target = resolve_within(destination, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(member) as source, target.open("wb") as sink:
                while chunk := source.read(1024 * 1024):
                    sink.write(chunk)
            extracted.append(relative)
    return extracted


def find_secrets(payload: dict) -> list[str]:
    """Detect credential-shaped keys that must never travel inside a package."""
    findings: list[str] = []

    def walk(node, trail: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                if lowered in {"api_key", "apikey", "secret", "password", "token", "authorization"}:
                    findings.append(f"{trail}.{key}".lstrip("."))
                walk(value, f"{trail}.{key}".lstrip("."))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{trail}[{index}]")

    walk(payload, "")
    return findings
