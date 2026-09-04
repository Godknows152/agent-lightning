"""Dependency, model, data, template, and component preflight."""
from __future__ import annotations
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ENV_ROOT = ROOT.parents[1] / "contrib" / "recipes" / "envs"
sys.path.insert(0, str(SRC))

def main() -> int:
    import importlib.util
    required = ["alfworld", "gymnasium", "stable_baselines3", "pandas", "pyarrow", "omegaconf", "transformers", "swanlab"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"VERL/ALFWorld runtime preflight missing dependencies: {missing}")
    cxx = Path(os.environ.get("ALFWORLD_CXX", "/usr/bin/g++-10"))
    nvcc = Path(os.environ.get("ALFWORLD_NVCC", "/usr/local/cuda/bin/nvcc"))
    if not cxx.is_file():
        raise RuntimeError(f"C++20 compiler not found: {cxx}. Install g++-10 or set ALFWORLD_CXX.")
    if not nvcc.is_file():
        raise RuntimeError(f"CUDA compiler not found: {nvcc}. Set ALFWORLD_NVCC.")
    with tempfile.TemporaryDirectory(prefix="alfworld-cxx20-") as tmp:
        source = Path(tmp) / "concepts.cpp"
        output = Path(tmp) / "concepts"
        source.write_text("#include <concepts>\ntemplate<class T> concept C = true; int main(){return C<int>?0:1;}\n")
        subprocess.run([str(cxx), "-std=c++20", str(source), "-o", str(output)], check=True, capture_output=True, text=True)
        subprocess.run([str(output)], check=True)
        cuda_source = Path(tmp) / "concepts.cu"
        cuda_object = Path(tmp) / "concepts.o"
        cuda_source.write_text("#include <concepts>\n__global__ void kernel() {}\n")
        subprocess.run(
            [str(nvcc), "-ccbin", str(cxx), "-std=c++20", "-arch=sm_80", "-c", str(cuda_source), "-o", str(cuda_object)],
            check=True,
            capture_output=True,
            text=True,
        )
    print(f"cxx20_compiler={cxx} nvcc={nvcc}")
    from transformers import AutoTokenizer
    from alfworld_baseline.parser import parse_tool_call
    from alfworld_baseline.tool_registry import ALFWorldToolRegistry
    from alfworld_baseline.validator import ValidationStatus, validate_tool_call
    model = Path(os.environ.get("ALFWORLD_MODEL", "/home/LXJ/Python_Projects/Models/Qwen3.5-2B"))
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    if not tokenizer.chat_template:
        raise RuntimeError(f"Qwen tokenizer has no native chat_template: {model}")
    data_dir = Path(os.environ.get("ALFWORLD_DATASET_DIR", ROOT / "data" / "qwen35_2b"))
    missing_data = [str(data_dir / name) for name in ("train.parquet", "test.parquet") if not (data_dir / name).is_file()]
    if missing_data:
        raise RuntimeError("missing prepared VERL parquet files: " + ", ".join(missing_data))
    registry = ALFWorldToolRegistry(["look", "go to cabinet 1"])
    rendered = tokenizer.apply_chat_template([{"role": "system", "content": "Use one tool."}, {"role": "user", "content": "Choose."}], tools=[registry.build_tool_schema()], tokenize=False, add_generation_prompt=True)
    print(f"tokenizer={tokenizer.__class__.__name__} eos={tokenizer.eos_token_id} pad={tokenizer.pad_token_id}")
    print(f"native_template_sha256={hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()} rendered_tokens={len(tokenizer(rendered, add_special_tokens=False)["input_ids"])}")
    valid = parse_tool_call('<tool_call>\n<function=alfworld_action>\n<parameter=action>\nlook\n</parameter>\n</function>\n</tool_call>')
    if validate_tool_call(valid, registry).status is not ValidationStatus.VALID:
        raise AssertionError("component contract failed")
    print("status=preflight_ok")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
