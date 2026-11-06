"""Model Input Package v1 — the self-contained hand-off between the two MCPs.

The package is the *only* coupling between the spatial MCP and the hydro MCP:
either side can be produced or consumed independently, and nothing in it may
reference an absolute path, a credential, or a directory outside the package.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .paths import safe_relative_path


SCHEMA_VERSION = "1.0"

#: File roles the hydro MCP understands. ``landuse`` and ``river`` are the
#: member files of a shapefile set and are expanded by the reader.
REQUIRED_ROLES = ("dem", "boundary", "rainfall", "landuse", "river")

SHAPEFILE_COMPANIONS = (".shx", ".dbf", ".prj", ".cpg")


class PackageFile(BaseModel):
    role: str
    relative_path: str
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def _check_relative(cls, value: str) -> str:
        return safe_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def _check_hex(cls, value: str) -> str:
        if any(character not in "0123456789abcdef" for character in value.lower()):
            raise ValueError("sha256 必须是十六进制字符串")
        return value.lower()


class EventSpec(BaseModel):
    start_datetime: str
    end_datetime: str
    timezone: str = "Asia/Shanghai"
    rainfall_source: str

    @field_validator("rainfall_source")
    @classmethod
    def _flag_virtual_stations(cls, value: str) -> str:
        # The v1 contract requires the virtual-station caveat to travel with the data.
        return value


class StudyArea(BaseModel):
    crs: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    area_km2: float = Field(gt=0)


class ModelInputPackage(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    package_id: str
    created_at: str
    event: EventSpec
    study_area: StudyArea
    files: list[PackageFile]
    qa: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)

    def file_for(self, role: str) -> PackageFile | None:
        for entry in self.files:
            if entry.role == role:
                return entry
        return None

    def roles(self) -> set[str]:
        return {entry.role for entry in self.files}

    def missing_roles(self) -> list[str]:
        return sorted(set(REQUIRED_ROLES) - self.roles())
