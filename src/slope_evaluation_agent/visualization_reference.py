"""Public figure identifiers for understanding requests, not plotting code."""
TARGETS={
    'watershed':'已保存的人工评价边界（空间审核图）',
    'delineate_subcatchments':'本次建模的子汇水区划分图',
    'delineate_area_histogram':'子汇水区面积分布',
    'delineate_coverage_gap':'子汇水区覆盖缺口',
    'rainfall_station_series':'逐时降雨过程',
    'rainfall_cumulative':'累积降雨',
    'rainfall_station_map':'虚拟雨站位置',
    'build_model_structure':'SWMM 模型结构',
    'swmm_runoff_hydrograph':'径流过程',
    'safety_minimum_factor':'最小安全系数分布',
    'safety_time_series':'安全系数时序',
    'analysis_sensitivity_ranking':'参数敏感性排序',
    'analysis_sobol_contribution':'参数贡献',
    'qa_dashboard':'质量检查图',
}
ALIASES={'subcatchments':'delineate_subcatchments','rainfall_series':'rainfall_station_series',
         'sobol_contribution':'analysis_sobol_contribution'}


def normalize_kind(kind):
    return ALIASES.get(kind,kind)
