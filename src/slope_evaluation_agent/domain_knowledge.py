"""What the agent knows that is not a computation.

Everything in this module is static text a person wrote and approved. Nothing
here is generated, and nothing here is a number: the numeric side of the system
lives in the MCP servers and in the evidence ledger, and a conversational answer
is not allowed to invent any part of it.

The term explanations are taken from :mod:`slope_evaluation_agent.evaluation`
rather than copied, so the wording a user sees in a conversation and the wording
they see in a result explanation cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from .evaluation import TOPICS as TERM_TOPICS


#: What this agent does, in the order a user would meet it.
CAPABILITIES: tuple[str, ...] = (
    "使用配置的通用大模型回答水文地质、地质灾害、地理、岩土等工程知识问题，无需启动计算。",
    "按河名加载选址地图，由你点选出水口；也可以直接给经纬度，划分对应的流域边界。",
    "把边界交给你确认——只有你确认过的流域才会进入后续计算。",
    "把区域尺度的 ERA5 网格降雨插值成流域内的虚拟雨站，生成降雨事件。",
    "裁切并导出模型输入包，交给水文服务做 SWMM 建模与逐像元的稳定性计算。",
    "跑完质量门槛后给出区域尺度的初步筛查结果，并解释其中每个数值的含义与来源。",
)

#: What it does not do. These are the boundaries, said plainly.
NOT_CAPABILITIES: tuple[str, ...] = (
    "不做工程级边坡安全鉴定：这是区域尺度、降雨事件驱动的初步筛查与诊断。",
    "不做灾害预报，也不给泥石流易发性评价或任何风险等级。",
    "不产生综合风险分数，也不替工程判断选址、支护或治理方案。",
    "目前不联网检索；知识问答来自通用模型或内置知识，项目计算结论仅依据当前 MCP 成果。",
)

#: The two services, and what each is for.
SERVICES: dict[str, str] = {
    "spatial": "空间处理服务：检查数据、生成选址地图、划分流域、生成虚拟雨站降雨、"
               "裁切数据并导出模型输入包。它不做任何水文或稳定性计算。",
    "hydro": "水文与稳定性服务：导入模型输入包、跑 SWMM 和稳定性计算、做敏感性与"
             "质量门槛检查、注册成果并出图。它不生成流域边界。",
}

#: Explanations that are not about a particular number.
CONCEPTS: dict[str, str] = {
    "greeting": "你好。我可以回答水文地质、地质灾害与岩土等工程知识问题，也可以做区域尺度的坡面稳定性初步筛查与诊断："
                "给一条河的名字或一个经纬度，我会划分流域、请你确认边界、准备降雨、"
                "建模计算，最后给出带来源的初步结果。也可以直接问我某个概念。",
    "capability": "我可以利用通用大模型回答水文地质、地质灾害、地理与岩土等工程知识问题，知识问答无需启动计算。"
                  "需要分析具体流域时，我也可以带你走完一次区域尺度的坡面稳定性初步筛查：给一条河或一个坐标，"
                  "我会划分流域、请你确认边界、准备降雨、建模计算，最后给出带来源的初步结果，"
                  "并解释其中每个数值。",
    "boundary": "边界必须由你确认，是这套系统的设计前提。流域边界决定后面所有计算的范围："
                "换一个出口点，面积、降雨分布和每个像元的结果都会变。所以先由模型给出一个"
                "候选边界和图件，再由你决定它是否可信——我不代替你做这个决定。",
    "no_date": "没有具体日期也可以开始。降雨有两种取法：指定某一天或某个时段，"
               "或者取资料期内的样本最大日。后者不需要你挑日期，但它是一个样本最大值，"
               "不是重现期设计暴雨。",
    "engineering": "不能直接用于工程设计。这是区域尺度、降雨事件驱动的初步筛查与诊断，"
                   "安全系数由简化力学模型计算、未做场地勘察标定，降雨来自 ERA5 网格虚拟雨站。"
                   "它没有岩土勘察参数、没有场地边界条件，也没有经过工程实践的校验，"
                   "所以做不了工程级安全鉴定。它可以帮你判断哪些位置值得进一步勘察，"
                   "但工程上的结论需要专门的勘察与计算。",
    "risk_grade": "系统只报指标，不给等级：它没有风险等级，也不分低、中、高。原因不是措辞保守，"
                  "而是没有依据——分级需要一套经过验证的标准：多大的失稳可能算「高」、"
                  "对什么对象而言算「高」。本系统只做区域尺度、降雨事件驱动的初步筛查，"
                  "没有做过把像元级结果标定为区域风险的工作。给出等级，等于把未经验证的"
                  "判断包装成结论。能提供的是各项指标本身、它们的来源和适用边界，"
                  "由你判断它们有多要紧。",
    "composite_score": "系统不产生综合风险分数，也不把多个指标加权成一个总分。原因有两条："
                       "一是权重没有依据——把安全系数、失稳面积和持续时间合成一个数，"
                       "需要一套经过验证的权重，而目前没有；二是总分会把不确定性藏起来——"
                       "一个数字看不出它来自局部像元还是整个流域，也看不出哪一项接近临界。"
                       "所以这里把各项指标分开列出，每项都带着单位和来源。",
    "debris_flow": "这里的结论不是泥石流易发性评价，也没有易发性等级。判断泥石流要看松散物源、"
                   "沟道形态、汇流与堆积条件，而这里算的是坡面在降雨下的稳定性——"
                   "用的数据、方法和结论对象都不是一回事。把一个模型的输出"
                   "贴到另一个问题上，比不回答更容易误导人。",
    "workflow": None,  # filled in below from WORKFLOW_STEPS
}

#: The steps a run goes through, from the user's point of view. Numbered in
#: Chinese numerals on purpose: the reply audit reads an Arabic digit as a
#: quantity, and a step number is not one.
WORKFLOW_STEPS: tuple[str, ...] = (
    "确定位置：给一条河或一个经纬度。",
    "划分流域：得到一个边界和一张流域图，请你确认或拒绝。",
    "准备降雨：指定日期、时段，或使用资料期内的样本最大日。",
    "准备输入：裁切数据并导出模型输入包，导入水文服务后校验。",
    "生成计划：列出将要执行的阶段和参数，你确认后才会开始。",
    "后台计算：SWMM 建模、逐像元稳定性计算、敏感性分析与质量门槛检查。",
    "给出结果：门槛全过时是初步评价，未过时只给诊断，不构成稳定性结论。",
)

_CN_ORDINALS = ("其一", "其二", "其三", "其四", "其五", "其六", "其七", "其八", "其九")

CONCEPTS["workflow"] = "一次完整的评价是这样走的：" + "；".join(
    f"{_CN_ORDINALS[index]}，{step}" for index, step in enumerate(WORKFLOW_STEPS))

#: The scientific caveats, reused verbatim from the policy module so the
#: conversation cannot soften them.
LIMITATIONS: tuple[str, ...] = (
    "本结果为区域尺度、降雨事件驱动的坡面稳定性初步筛查与诊断，不是工程级边坡安全鉴定。",
    "降雨来自 ERA5 网格虚拟雨站，不是地面实测雨量站。",
    "历史最大日是资料期内样本最大值，不是重现期设计暴雨。",
    "当前模型为独立子汇水区出水口配置：各子汇水区径流之和不是选定流域出口的路由流量过程。",
    "最小安全系数只说明存在局部像元进入 FS<1，不等于整个流域失稳。",
)

#: Keywords that pick a topic out of a question. Checked in order; the term
#: topics from ``evaluation`` come last so a specific question wins.
TOPIC_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("greeting", ("你好", "您好", "在吗", "hello", "hi", "早上好", "下午好")),
    # Boundary questions come first: "为什么不给风险等级" also contains 给 and 等级,
    # and answering it as a general capability question would miss the point.
    ("risk_grade", ("风险等级", "风险分级", "安全等级", "稳定等级", "定级", "分级",
                    "低风险", "中风险", "高风险", "风险评级")),
    ("composite_score", ("综合风险分数", "风险分数", "综合分数", "总分", "打分", "评分",
                         "一个分数", "一个数")),
    ("debris_flow", ("泥石流", "易发性")),
    ("engineering", ("工程设计", "设计", "施工", "能不能用", "可以直接用", "工程上",
                     "安全鉴定", "工程鉴定")),
    ("boundary", ("确认", "边界为什么要", "为什么要我", "人工", "审核", "门")),
    ("no_date", ("没有日期", "不知道日期", "没有具体", "随便哪场", "不确定日期",
                 "没有降雨", "没日期", "不知道哪天", "哪场雨都行")),
    ("capability", ("你能做", "能做什么", "会做什么", "能分析哪", "能分析什么",
                    "有什么功能", "支持什么", "你是谁", "你是什么")),
    ("workflow", ("怎么走", "什么流程", "步骤", "下一步", "怎么做", "如何开始", "怎么开始")),
    ("limitations", ("局限", "限制", "边界在哪", "有什么问题", "准确吗", "可靠吗")),
) + tuple((name, keywords) for name, keywords in (
    ("safety_factor", ("安全系数", "fs")),
    ("pore_pressure", ("孔压", "孔隙水压力", "水压")),
    ("unstable", ("失稳", "不稳定", "持续时间", "比例")),
    ("qa", ("质量门槛", "qa", "门槛")),
    ("rainfall", ("降雨", "雨量", "era5", "虚拟站", "雨站")),
    ("model", ("模型", "汇流", "子汇水区", "出口")),
))

DEFAULT_TOPIC = "capability"


def answers_from_knowledge(topic: str | None) -> bool:
    """Whether this topic is answered from here rather than from the explainer.

    ``evaluation.explain`` answers *terms* and the current result, from the
    evidence ledger. A question about why the system does not produce a kind of
    conclusion has no evidence to quote — explaining it is what this module is
    for, and sending it to the explainer produced a generic answer, or worse,
    the generic answer plus the user's own question echoed back.
    """
    return bool(topic) and topic not in TERM_TOPICS


def match_topic(question: str) -> str | None:
    """The topic a question is about, or ``None`` when nothing matches."""
    lowered = str(question or "").lower()
    for topic, keywords in TOPIC_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return topic
    return None


def snippets(topic: str) -> list[str]:
    """The approved text for a topic. An unknown topic yields nothing."""
    if topic in CONCEPTS and CONCEPTS[topic]:
        return [CONCEPTS[topic]]
    if topic in TERM_TOPICS:
        return [TERM_TOPICS[topic]]
    if topic == "limitations":
        return list(LIMITATIONS)
    if topic == "capabilities_list":
        return list(CAPABILITIES)
    return []


def catalogue() -> dict[str, Any]:
    """A machine-readable view, for the tests and for the Web health endpoint."""
    return {
        "topics": sorted({*CONCEPTS, *TERM_TOPICS, "limitations", "capabilities_list"}),
        "capabilities": list(CAPABILITIES),
        "not_capabilities": list(NOT_CAPABILITIES),
        "services": dict(SERVICES),
        "workflow_steps": list(WORKFLOW_STEPS),
        "limitations": list(LIMITATIONS),
    }
