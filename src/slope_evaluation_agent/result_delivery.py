"""Complete explicit read-only deliverables after the existing semantic route."""
from dataclasses import dataclass
import re

from .intent import Intent, RuleParser


@dataclass(frozen=True)
class Deliverables:
    result: bool = False
    figure: bool = False
    download: bool = False
    provenance: bool = False
    figure_kind: str | None = None

    @property
    def compound(self):
        return sum((self.result, self.figure, self.download, self.provenance)) > 1


def actionable_clauses(text):
    # A quoted former request and a hypothetical question are not instructions.
    text = re.sub(r'“[^”]*”|「[^」]*」|"[^"\n]*"|‘[^’]*’', '', text)
    return [c for c in re.split(r'[，,；;。\n]|但是|不过|但', text)
            if c.strip() and not re.search(r'如果|假如|假设|如何|怎么|哪里|是否支持|只(?:是)?问', c)]


def _asks(clauses, pattern):
    for clause in clauses:
        for match in re.finditer(pattern, clause, re.IGNORECASE):
            if not re.search(r'不要|不必|无需|先别|暂不|不需要|不想|(?<!分)别|禁止',
                             clause[max(0, match.start()-12):match.end()]):
                return True
    return False


def requested_deliverables(text, parsed):
    clauses = actionable_clauses(text)
    result = _asks(clauses, r'(?:解释|解读|说明|分析|报告|查看|读取).{0,40}'
                  r'(?:结果|数值|指标|安全系数|FS|面积|比例|分母|时间口径|QA)'
                  r'|(?:给(?:出)?|提供|列出|告诉).{0,40}'
                  r'(?:数值|指标|最[小低].{0,4}(?:安全系数|FS)|面积|比例|分母|时间口径|QA)')
    figure = _asks(clauses, r'(?:显示|展示|渲染|绘制|画|看看|看一下|查看|看).{0,28}(?:图|figure|plot)'
                  )
    download = _asks(clauses, r'下载|导出|另存|download')
    if not download:
        figure |= _asks(clauses, r'(?:给|提供).{0,20}(?:图件|图像|图)')
    provenance = _asks(clauses, r'运行(?:编号|ID)|run[_ -]?id|降雨.{0,12}(?:起止|来源)|'
                      r'参数(?:来源|预设|覆盖)|追溯')
    clean = '，'.join(clauses)
    kind = parsed.visualization_kind or RuleParser._visualization(clean)
    # The semantic router still supplies the primary intent. Extra outputs
    # require the user's affirmative words, never a model-generated tool plan.
    if clauses:
        result |= parsed.intent is Intent.SHOW_RESULT
        figure |= parsed.intent is Intent.SHOW_VISUALIZATION
        download |= parsed.intent is Intent.DOWNLOAD_ARTIFACT
    return Deliverables(result=result, figure=figure, download=download,
                        provenance=provenance, figure_kind=kind)


def permits_any_file(text):
    return _asks(actionable_clauses(text), r'任一|任意|(?:一[个份种]|随便).{0,12}(?:文件|成果|数据|表|栅格)')
