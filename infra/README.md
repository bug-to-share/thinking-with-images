# 上游组件与补丁

- [`medevalkit.patch`](medevalkit.patch) 基于 [MedEvalKit](https://github.com/alibaba-damo-academy/MedEvalKit) commit `9b12e3b34232a8779cc1275417bf4f72f03d28e0`。包含 Qwen3-VL 注册、提示模板、SLAKE English 和评分辅助改动。
- [`verl_qwen3_vl_dedup.patch`](verl_qwen3_vl_dedup.patch) 基于 [VERL v0.7.1](https://github.com/verl-project/verl/releases/tag/v0.7.1)，增加 Qwen3-VL 重复视觉输入去重。
- [`sglang/`](sglang/) 保留历史部署中使用的两份 [SGLang](https://github.com/sgl-project/sglang) 补丁，分别针对视觉权重前缀与多次序列化时共享内存段复用。

上述上游组件及衍生补丁保留各自的 Apache 2.0 授权条款，参见 [Apache 2.0 文本](APACHE-2.0.txt)。本项目未附带这些框架的完整源码，也未附带 Qwen、BioMedParse、MedSAM3 权重或 SLAKE 数据；使用外部资产时遵守原发布方条款。
