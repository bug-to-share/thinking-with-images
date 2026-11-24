# 运行与复现说明

下面的命令从仓库根目录执行。Python 3.10 或更新版本可用于 Agent 与数据工具。训练依赖 Linux、CUDA、VERL 及与 GPU 环境兼容的 SGLang。训练与评测使用独立环境可以避免推理后端依赖冲突。

## 先检查轻量入口

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python qwen35_agent/agent_loop.py --help
python gemini_agent_native_bbox/agent_loop.py --help
python script/export_epoch1_llamafactory.py --help
```

Qwen Agent 读取一个 JSON 数组，每行包含本地图片 `image_path` 以及 `vqa_data`。后者包括 `question`、带 `A.` 等选项标记的 `options`、一个字母 `answer`。`answer` 用于保存结果和判分。图像路径相对当前运行目录解析。

```json
[
  {
    "image_path": "external/example.png",
    "vqa_data": {
      "question": "What is the shape in this synthetic image?",
      "options": ["A. Square", "B. Circle"],
      "answer": "A"
    }
  }
]
```

配置一个支持图像的 OpenAI 兼容模型端点和兼容 Jupyter 沙箱。Agent 开始时会在沙箱注册原图。分割和检索服务按需调用，可用 `--biomedparse-url`、`--medsam3-url`、`--retrieve-url` 指定；受保护服务的凭据由本地环境变量提供。沙箱运行模型生成的 Python，应使用隔离的实例。

```bash
python qwen35_agent/agent_loop.py \
  --vqa-path external/examples.json \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.5-9B --reasoning-backend vllm \
  --max-items 1 --allow-answer-without-server-tool \
  --output-dir outputs/agent-smoke
```

历史固定工具对照使用 `--require-server-tool-before-answer`。轨迹和汇总写入 `--output-dir`，含本地样本路径与工具结果，建议留在实验环境内。Gemini 教师使用 `GEMINI_API_KEY`；`agent_loop_with_answer.py` 使用答案特权，只适用于训练教师轨迹。

`tool/` 是 HTTP 工具栈，`tools/medical_tools.py` 是 VERL 原生工具适配器。检索服务需要自行准备知识库和与编码器一致的 FAISS 索引。BioMedParse 服务通过 `BIOMEDPARSE_ROOT` 读取外部模型代码及推理配置。MedSAM3 服务通过 `MEDSAM3_SERVER_SCRIPT` 加载兼容入口。可用工具及请求字段见 `tool/bridge.py` 和原生 recipe 的 `config/tools.yaml`。

## 准备 SLAKE English

按数据集条款获取 SLAKE，放入 `eval/benchmarks/SLAKE/`，保留 `train.json`、`validation.json`、`test.json` 和 `imgs/` 的原始布局。请确认数据来源与项目所用版本一致。训练转换仅使用 train 和 validation。

```bash
python -m pip install datasets pyarrow
R1=recipe/med_twi/text_cot_infra_fixed/r1_text_cot_em
python data/med_twi/scripts/build_text_cot_dataset.py --languages en \
  --train-output data/med_twi/text_cot_infra_fixed/r1_text_cot_em/train.parquet \
  --val-output eval/benchmarks/SLAKE/views/slake_validation_en/validation.parquet
```

## 后训练

历史实验使用 VERL 0.7.1 及 SGLang 0.5.9 系列环境。将兼容的 VERL 工作树放在 `verl/`，按上游安装说明配置 CUDA 依赖。`infra/verl_qwen3_vl_dedup.patch` 适用于 v0.7.1，用于去重重复视觉输入。SGLang 补丁仅用于仍存在视觉权重前缀或共享内存复用问题的对应版本，应用前用 `git apply --check` 验证。

```bash
bash "$R1/scripts/train.sh"
```

启动脚本保留内存保护。按设备和数据设置 `CUDA_VISIBLE_DEVICES`、`N_GPUS_PER_NODE`、`TRAIN_DATA`、`VAL_DATA`、`MODEL_PATH`。其他 CoT 方案按各目录的 `prepare_data.sh` 或 `TRAIN_DATA` 参数准备输入。其中 judge 方案需配置 `LLM_AS_A_JUDGE_BASE`、`LLM_AS_A_JUDGE_MODEL` 和服务要求的密钥。

多轮工具方案在 `recipe/med_twi/agentic_zoomin_retrieve/qwen3_vl_4b_instruct/`。其脚本按需要生成 train parquet，读取上述 validation parquet，默认通过 SLAKE `KG/` 检索知识，并使用同一 judge 服务配置。参考这个 recipe 的 `train.sh`、`config/tools.yaml` 和 `reward.py` 配置。

SFT 数据转换入口为 `script/export_epoch1_llamafactory.py`，需本地 VQA、已筛选轨迹以及 LLaMA-Factory。历史 SFT 实验有样本重叠，重新生成数据时应按图像与题目检查训练、开发和测试集的重合。

## 评测

使用 MedEvalKit 基线 `9b12e3b34232a8779cc1275417bf4f72f03d28e0` 及仓库中随附的 `infra/medevalkit.patch`。补丁加入 Qwen3-VL vLLM adapter、通用 prompt 配置、SLAKE English 筛选与历史评测修正。

```bash
mkdir -p eval
git clone https://github.com/alibaba-damo-academy/MedEvalKit.git eval/MedEvalKit
git -C eval/MedEvalKit checkout 9b12e3b34232a8779cc1275417bf4f72f03d28e0
git -C eval/MedEvalKit apply "$PWD/infra/medevalkit.patch"
```

在独立评测环境按上游要求安装依赖，再安装与 Qwen3-VL 兼容的 `transformers`、`vllm`、`qwen-vl-utils` 和 `rouge-score`。将 FSDP checkpoint 用 VERL model merger 导出成推理模型目录后，设置 `MODEL_PATH` 和独立的 `OUTPUT_PATH`。

```bash
MODEL_PATH="$PWD/external/merged-model" OUTPUT_PATH="$PWD/outputs/slake-r1" \
  bash "$R1/scripts/eval_medevalkit.sh"
```

若模型路径是相对路径，请传入仓库根目录展开后的绝对路径，例如 `MODEL_PATH="$PWD/external/merged-model"`，因为启动器会进入 MedEvalKit 目录。需要 judge 结果时另设 `RUN_LLM_JUDGE=True`、`JUDGE_MODEL`、`BASE_URL` 和新的 `OUTPUT_PATH`。非 judge 总正确率采用 OPEN EM 加 CLOSE 规则判分。使用新输出目录可以避免加载旧缓存。

发布整理验证了语法、部分无需模型的入口、HTML 交互与补丁适用性。历史正式分数没有在整理过程中重新训练或评测。汇总指标保存在 `metrics.json`，训练与测试的详细证据边界见 [技术复盘](project_review.md)。
