# alfworld-verl 统一运行环境

环境名称：`alfworld-verl`

解释器：`/home/LXJ/anaconda3/envs/alfworld-verl/bin/python`

创建方式：从项目现有 `verl` 环境克隆，再安装：

```text
alfworld==0.4.2
gymnasium==0.29.1
stable-baselines3==2.6.0
```

已验证的关键包：

```text
Python 3.12.0
verl 0.8.0.dev0（项目内 verl_backend）
torch 2.9.1
transformers 4.57.1
vllm 0.11.0
sglang 0.5.8
ray 2.54.0
alfworld 0.4.2
gymnasium 0.29.1
stable-baselines3 2.6.0
pandas 3.0.1
pyarrow 23.0.1
omegaconf 2.3.0
```

验证过的链路：

```text
项目内 verl 导入
→ ALFWorld/TextWorld 导入
→ Qwen2.5 tokenizer 原生 template 加载
→ old-VERL hermes parser 与 alfworld_action 工具配置加载
→ ALFWorldTool.create
→ 合法动作 execute
→ tool.release
→ alfworld_tool_agent 注册
```

注意：`pip check` 仍会报告克隆前 `verl` 环境中已有的 vLLM/PyTorch/xformers 版本约束提示；这些提示在原 `verl` 环境也存在，不能据此宣称 vLLM 训练已完成验证。正式 pilot 前仍需运行统一环境 preflight、Hydra composition 和真实模型加载检查。old-VERL 图像修复配置实际使用 SGLang rollout，ALFWorld 配置也固定为 `rollout.name: sglang`；vLLM 仅作为可选依赖保留。
