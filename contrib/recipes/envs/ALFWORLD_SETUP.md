# ALFWorld 文本环境

本目录的 ALFWorld 集成使用独立 Conda 环境 alfworld，不启动 Ray、vLLM 或 GRPO。上游适配位于 agl_envs（记录其 git commit），数据位于 agl_envs/alfworld/alfworld_source，数据集为 agl_envs/task_data/alfworld/{train,test}.parquet。

## 安装与数据

    cd /home/LXJ/Python_Projects/Agent_Lightning/contrib/recipes/envs
    export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897
    conda create -n alfworld python=3.10 -y
    /home/LXJ/anaconda3/envs/alfworld/bin/pip install -i https://pypi.org/simple --proxy http://127.0.0.1:7897 \
      gymnasium==0.29.1 alfworld==0.4.2 pandas pyarrow omegaconf
    export ALFWORLD_DATA=$PWD/agl_envs/alfworld/alfworld_source
    conda run -n alfworld python agl_envs/alfworld/download_alfworld_source.py
    conda run -n alfworld python agl_envs/task_data/alfworld/make_alfworld_dataset.py

stable-baselines3 不是文本 smoke test 的必要依赖；若后续需要其 API，应在确认 PyTorch/CUDA 版本后单独安装 stable-baselines3==2.6.0，避免意外拉取大型 CUDA wheel。

## 验证

    cd /home/LXJ/Python_Projects/Agent_Lightning/contrib/recipes/envs
    export ALFWORLD_DATA=$PWD/agl_envs/alfworld/alfworld_source
    conda run -n alfworld python alfworld_smoke_test.py --steps 5
    conda run -n alfworld python alfworld_smoke_test.py --steps 1 --max-steps 1 --agl

脚本会检查 parquet、任务文件、reset、合法文本动作、observation、admissible actions、reward、done 和 close；--agl 额外检查 make_env_manager、single prompt 和八元组 step 合约。LLM 验证需另行提供 OpenAI-compatible endpoint，不属于本脚本硬性验收。

## 当前验证边界

环境验证通过只代表文本环境及 AGL 适配可运行，不代表模型质量、Ray 分布式链路或 RL 训练已验证。若下载或运行失败，优先检查 ALFWORLD_DATA、base_config.yaml、任务文件路径和 ALFWorld/TextWorld 版本。
