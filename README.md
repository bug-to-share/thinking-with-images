# Thinking with Images for Medical VQA

独立研究项目。使用视觉语言模型做医学图像问答，并探索何时调用视觉工具、如何在后续推理中使用裁图或分割结果、怎样通过 SFT 与 GRPO 改善决策。

[HTML 面试与项目文档](docs/interview.html) 包含实验图表、3 版简历描述、26 道面试问答、论文调研与研究计划。克隆仓库或下载 ZIP 后，在浏览器打开 `docs/interview.html` 即可离线使用。

[运行与复现说明](docs/reproduction.md) · [技术复盘](docs/project_review.md) · [汇总指标](docs/metrics.json) · [参考文献](docs/references.md)

| 模块 | 入口 |
| --- | --- |
| 多轮视觉推理 | [`qwen35_agent/`](qwen35_agent/) 中的 Agent 循环与 trace，配合 [`tool/`](tool/) 中的检索、分割和外部沙箱接口 |
| 教师轨迹与 SFT 数据 | [`gemini_agent_native_bbox/`](gemini_agent_native_bbox/) 中的坐标适配与轨迹校验，配合 [`script/`](script/) 中的轨迹分桶及导出 |
| 数据准备 | [`data/med_twi/scripts/`](data/med_twi/scripts/) 与 [`data_curation/`](data_curation/) 中的转换、采样与难度筛选 |
| 后训练 | [`recipe/med_twi/`](recipe/med_twi/) 中的无工具 CoT GRPO 和可选 zoom/retrieve 的多轮方案 |
| 评测与环境 | [`infra/`](infra/) 中的 MedEvalKit、VERL 与 SGLang 补丁 |

历史实验中，Qwen3-VL-4B 的无工具 CoT GRPO 在 SLAKE English 官方测试 1,061 题上的非 judge 总正确率由 46.65% 变为 67.30%。该口径合并 OPEN 精确匹配与 CLOSE 规则评分。历史推理记录存在后端与解码差异，复核范围见技术复盘。

在 9,997 条项目内部样本上，固定先用工具的 Qwen3.5-9B 流程相较 Direct 有 199 题改善、450 题退化。这个对照描述流程差异；下一步是在干净数据、同一模型与固定预算下验证选择性调用。

仓库提供源码、配置、补丁和文档。原始医学图像、问答数据、权重、逐条轨迹及日志由各自来源和使用条件管理。项目用于离线研究，未做临床部署验证。上游依赖与补丁归属见 [第三方组件说明](infra/README.md)。
