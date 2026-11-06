"""General engineering knowledge is distinct from evidence about a run."""

from __future__ import annotations

import re

ENGINEERING_TERMS = (
    "水文", "地质", "地理", "灾害", "岩土", "土力学", "地下水", "含水层", "渗流",
    "坡面", "水流", "地表水",
    "达西", "水力", "孔压", "孔隙水", "边坡", "滑坡", "崩塌", "泥石流", "地震",
    "降雨", "雨量", "雨站", "径流", "汇流", "流域", "河流", "河道", "侵蚀", "支护",
    "抗剪", "固结", "排水", "安全系数", "摩尔", "库仑", "饱和", "非饱和",
)
QUESTION_CUES = ("什么", "为什么", "为何", "如何", "怎么", "哪些", "区别", "关系",
                 "原理", "机理", "解释", "介绍", "讲讲", "科普", "吗", "？", "?")
CURRENT_RESULT_CUES = ("本次", "这次", "当前结果", "这个结果", "刚才的结果", "本流域",
                       "本次流域", "这里的结果", "最低值", "最小值", "为什么这么低",
                       '评价区域', '评测区域', '研究区', '计算区域', '分布情况', '计算产物')


def asks_general_concept(question: str) -> bool:
    return any(cue in question for cue in ("一般", "通用", "理论", "概念", "定义", "原理"))


def asks_hypothetical_mechanism(question: str) -> bool:
    """A counterfactual discussion is not a request for this run's summary."""
    return bool(re.search(r'如果|假如|假设|要是', question) and
                re.search(r'怎样|怎么样|机理|原理|影响|变化|会不会', question) and
                re.search(r'雨量|降雨|降水|参数|黏聚|摩擦|饱和|水深|流速|孔压|坡度|土层|厚度|密度|渗透', question))


def is_system_question(question: str, topic: str | None) -> bool:
    if topic in {"greeting", "capability", "no_date", "qa"}:
        return True
    if topic == "boundary":
        return any(cue in question for cue in ("确认", "边界", "你", "系统", "智能体"))
    if topic in {"risk_grade", "composite_score", "debris_flow", "engineering",
                 "limitations", "workflow"}:
        return any(cue in question for cue in (
            "系统", "智能体", "你能", "你为什么", "不给", "不提供", "不产生", "这里的",
            "这个结果", "本次结果", "当前结果", "下一步", "如何开始", "怎么开始",
        ))
    return False


def asks_current_result(question: str) -> bool:
    return any(cue in question for cue in CURRENT_RESULT_CUES) or bool(re.search(
        r"(?:安全系数|FS|失稳).*(?:这么|那么|如此)(?:低|高|大|小)", question, re.IGNORECASE))


def is_engineering_question(question: str) -> bool:
    """Recognise a knowledge question without treating a scenario as a concept."""
    if not any(term in question for term in ENGINEERING_TERMS):
        return False
    if not any(cue in question for cue in QUESTION_CUES):
        return False
    # A request to judge stability needs actual location/event inputs. Naming
    # a slope in a question must not turn that judgement into general theory.
    if not asks_general_concept(question) and re.search(
            r"是否稳定|稳不稳定|稳定吗|会不会(?:失稳|滑坡|发生滑坡)", question):
        return False
    # These ask for a concrete event assessment, even when phrased as a question.
    scenario = any(cue in question for cue in ("最近这场", "最近一次", "这场雨", "这条河",
                                               "这个流域", "这个地方", "这里会", "这座边坡"))
    if scenario and any(cue in question for cue in ("稳定", "失稳", "滑坡", "评估", "分析")):
        return False
    if re.search(r"\d{4}[-年/]\d{1,2}", question) and any(
            cue in question for cue in ("稳定", "失稳", "评估", "分析")):
        return False
    if any(cue in question for cue in ("帮我评估", "帮我分析", "帮我计算", "开始计算", "开始运行")):
        return False
    return True


def knowledge_answer_problems(text: str, *, task_background: dict | None = None) -> list[str]:
    """Reject fabricated session evidence or performed actions, not theory numbers."""
    problems = []
    checks = (
        (r"(?:本次|当前|该|这个)流域.{0,35}(?:安全系数|失稳面积|风险等级|降雨量|计算结果)",
         "知识回答声称了当前流域的计算结论"),
        (r"(?:本次|当前|这次)(?:的)?(?:计算结果|模拟结果|安全系数|失稳比例).{0,15}(?:为|是|等于|=)",
         "知识回答声称了本次计算指标"),
        (r"(?:已|已经|刚刚|本轮).{0,12}(?:调用|运行|执行|完成|确认|取消).{0,20}(?:MCP|SWMM|计算|流域|边界|任务)",
         "知识回答声称执行了工作流操作"),
        (r"(?:计算|SWMM|任务|边界|流域).{0,8}(?:已经|已|刚刚|本轮)(?:被)?(?:完成|运行|执行|确认|取消)",
         "知识回答声称执行了工作流操作"),
        (r"(?:hydro-run|basin|artifact)-[0-9a-f]{12}", "知识回答包含项目成果编号"),
    )
    background = task_background or {}
    if background.get('discussion_mode') == 'hypothetical_mechanism':
        if re.search(r'(?:安全系数|\bFs\b|失稳面积|孔压).{0,18}'
                     r'(?:为|至|到|降低|减少|增加|提高|变化).{0,8}\d', text, re.IGNORECASE):
            problems.append('假设讨论没有新计算，不支持预测场景数值')
        if re.search(r'(?:安全系数|\bFs\b|失稳面积|孔压).{0,12}(?:必然|一定)|'
                     r'(?:必然|一定).{0,12}(?:安全系数|\bFs\b|失稳面积|孔压)', text, re.IGNORECASE):
            problems.append('假设讨论不支持确定性的物理响应结论')
    run = background.get('run') or {}
    detail = run.get('detail') or {}
    stages = detail.get('stages') or []
    completed = set(detail.get('completed_stages') or [])
    for pattern, reason in checks:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            fragment = match.group()
            suffix = text[match.end():match.end()+50]
            if reason.startswith('知识回答声称了当前流域') and re.match(
                    r'\s*(?:尚未|未|没有|无法|不能|需要|须|不应)', suffix):
                continue  # Describing missing/unverified results is not a metric claim.
            if reason == '知识回答包含项目成果编号' and fragment in {
                    run.get('id'), (background.get('boundary') or {}).get('review_id')}:
                continue
            if reason == '知识回答声称执行了工作流操作':
                prefix = text[max(0, match.start()-8):match.start()]
                new_action = any(cue in prefix+fragment for cue in ('我', '帮你', '刚刚', '本轮'))
                if not new_action:
                    if ('确认' in fragment and '边界' in fragment
                            and (background.get('boundary') or {}).get('confirmed')):
                        continue
                    if ('取消' in fragment and run.get('status') == 'cancelled'):
                        continue
                    if run.get('submitted') and '取消' not in fragment and '确认' not in fragment:
                        if ('SWMM' in fragment.upper() and 'swmm' in completed):
                            continue
                        if (stages and set(stages) <= completed
                                and any(word in fragment for word in ('计算', '任务'))):
                            continue
            problems.append(reason)
            break
    for phrase in ("绝对安全", "保证不会失稳", "保证不会滑坡", "保证安全", "保证不会发生"):
        position = text.find(phrase)
        if position >= 0 and not any(word in text[max(0, position - 8):position]
                                     for word in ("不", "无法", "不能", "并非")):
            problems.append("知识回答作出了安全保证")
    return problems
