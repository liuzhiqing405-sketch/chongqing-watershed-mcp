"""The browser side of a download (stage F).

What the page may do with a server field is the subject here: every value that
came from a service is written with ``textContent`` or a DOM property, never
assembled into markup. A download is an ``<a href>`` whose target the server
built, and the format labels are the page's own words.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from slope_evaluation_agent.web import STATIC_ROOT


pytestmark = [pytest.mark.unit]

SCRIPT = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
STYLES = (STATIC_ROOT / "app.css").read_text(encoding="utf-8")

#: The one place markup is built from a literal: the typing indicator, whose
#: content is a fixed string in this file and never a server value.
TYPING_LINE = re.compile(r"bubble\.innerHTML = '<span class=\"typing-label\">.*?';")


class TestNoServerFieldBecomesMarkup:
    def test_the_only_innerhtml_is_the_fixed_typing_indicator(self):
        uses = re.findall(r"innerHTML\s*=", SCRIPT)
        assert len(uses) == 1, f"app.js 里出现了 {len(uses)} 处 innerHTML"
        assert TYPING_LINE.search(SCRIPT), "innerHTML 只应出现在固定的“正在理解”提示里"

    def test_download_labels_are_written_not_interpolated(self):
        assert "link.textContent = `下载原始 ${formatLabel(artifact.format)}" in SCRIPT
        assert "artifact.title" in SCRIPT and "link.title = artifact.title" in SCRIPT

    def test_a_download_is_a_link_to_the_servers_own_url(self):
        assert "link.href = artifact.download_url" in SCRIPT
        assert "link.setAttribute('download', '')" in SCRIPT

    def test_the_page_never_builds_a_download_url_itself(self):
        # URLs come from the server payload; the page only reads them.
        assert "artifact.download_url" in SCRIPT
        assert not re.search(r"`/api/v1/[^`]*\$\{[^}]*\}/download", SCRIPT)


class TestTheDownloadSurfaces:
    def test_every_figure_carries_its_own_files(self):
        assert "function addVisualizations(node, visualizations)" in SCRIPT
        assert "figure.artifacts" in SCRIPT and "addArtifactList" in SCRIPT

    def test_the_older_images_field_is_still_rendered(self):
        """A reply written by the previous release has only ``images``."""
        assert "function addImages(node, images, skipUrls = null)" in SCRIPT
        assert "addImages(node, reply.images, previews)" in SCRIPT

    def test_a_figure_is_not_shown_twice(self):
        assert "skipUrls" in SCRIPT and "previewUrls.add(figure.preview.url)" in SCRIPT

    def test_the_swmm_model_files_get_their_own_section(self):
        assert "function addSwmmSection(node, swmm)" in SCRIPT
        assert "'SWMM 建模成果'" in SCRIPT or "SWMM 建模成果（" in SCRIPT

    def test_all_process_files_have_an_original_format_section(self):
        assert "function addProcessArtifacts(node, artifacts)" in SCRIPT
        assert "全部过程源文件" in SCRIPT
        assert "CSV 直接下载 CSV" in SCRIPT

    def test_incomplete_swmm_packages_are_not_called_complete(self):
        assert "诊断模型包 ZIP" in SCRIPT
        assert "swmm.bundle_complete === false" in SCRIPT

    def test_the_binary_result_is_labelled_as_such(self):
        assert "out: 'OUT（SWMM 二进制）'" in SCRIPT
        assert "不能当文本打开" in SCRIPT

    def test_a_quality_failure_is_visible_before_downloading(self):
        assert "qa-badge" in SCRIPT and "仅供诊断，QA 未通过" in SCRIPT
        assert ".qa-badge" in STYLES

    def test_only_formats_the_service_offers_get_a_button(self):
        assert "artifact.export_formats || []" in SCRIPT
        assert "生成 " in SCRIPT  # the label is built from the offered format

    def test_generating_replaces_the_button_with_a_link(self):
        """A second click must not ask the service to derive the same file again."""
        assert "button.replaceWith(link)" in SCRIPT
        assert "正在生成…" in SCRIPT and "正在打包…" in SCRIPT

    def test_a_failure_is_reported_in_the_message(self):
        assert "artifact-error" in SCRIPT and "生成失败：" in SCRIPT
        assert "打包失败：" in SCRIPT
        assert ".artifact-error" in STYLES


class TestRestoreAndDetail:
    def test_a_reloaded_page_restores_the_downloads(self):
        assert "async function restoreArtifacts()" in SCRIPT
        assert "/artifacts`" in SCRIPT
        assert "await restoreArtifacts();" in SCRIPT

    def test_the_details_show_what_a_download_is(self):
        for label in ("单位", "坐标系", "时间列", "表格规模", "SHA-256", "质量门槛"):
            assert label in SCRIPT, label
        assert "sha256_short" in SCRIPT
        assert "artifact.stage_id || '—'" in SCRIPT

    def test_long_names_are_ellipsised_and_still_available(self):
        assert "text-overflow: ellipsis" in STYLES
        assert "link.title = artifact.title" in SCRIPT

    def test_the_controls_are_keyboard_reachable(self):
        assert ":focus-visible" in STYLES
        assert "aria-label" in SCRIPT and "aria-expanded" in SCRIPT

    def test_the_list_wraps_on_a_narrow_screen(self):
        assert ".artifact-row { display: flex; flex-wrap: wrap;" in STYLES


class TestPathsStayOutOfThePage:
    def test_the_page_has_no_absolute_path_of_its_own(self):
        leaked = re.findall(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", SCRIPT)
        assert not leaked, leaked

    def test_the_page_does_not_read_an_internal_path_field(self):
        assert "internal_path" not in SCRIPT
