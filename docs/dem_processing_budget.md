# DEM 处理预算

空间端与水文端共用默认 800 万像元的处理预算。`config/spatial.default.yaml` 的 `delineation.maximum_processing_cells` 控制边界提交与模型输入准备；`model_inputs.drainage_context_maximum_cells` 可设置更严格的外围窗口上限。水文端可在 `hydro.default.yaml` 的 `terrain.maximum_processing_cells` 设置独立上限，与输入包声明同时存在时取更严格值。

提交边界、准备输入、规划与创建任务均在读取地形数组前检查原生网格窗口。模型输入包含选区外围 1 km 的冻结 DEM，用于辅助排水划分；计算降雨面积与稳定性范围仍采用确认选区，框外来水不纳入计算。外围地形按不超过 512×512 像元分块复制，保持高程、NoData、原生分辨率及网格对齐。超预算时明确拒绝，不自动降采样或改变评价边界。

输入清单的 `provenance.dem_processing` 保存冻结窗口和预算，工作进程在分配数组前再次检查。旧包没有声明时使用统一默认。水文合理性诊断的 `drawn_area_hydrology.maximum_cells` 是独立预算；资源预检通过不代表自然流域完整性检查通过。区域评价和严格检查继续采用各自的确认门槛。
