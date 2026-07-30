"""
gradient_attribution_probe.py

验证梯度归因能否在 Qwen3.5 thinking chain 中定位工具选择决策点。

原理：
  计算 log P(action_token | context) 对每个 input embedding 的梯度。
  梯度范数大的 thinking token 位置 = 对最终动作选择因果影响最大的位置。

用法：
  # 使用内置示例
  python gradient_attribution_probe.py

  # 指定模型路径和动作
  python gradient_attribution_probe.py --model_path /path/to/Qwen3.5-9B --action focalnet_dehaze

  # 从文件读取生成的文本（粘贴一条rollout的thinking+tool_call内容）
  python gradient_attribution_probe.py --text_file my_rollout.txt --action focalnet_dehaze

  # 使用 LoRA adapter（训练过的版本）
  python gradient_attribution_probe.py --lora_path /path/to/lora --action nafnet_denoise

  # 保存归因热图为 PNG
  python gradient_attribution_probe.py --save_plot attribution.png
"""

import argparse
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 默认路径 ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "/home/LXJ/Python_Projects/Models/Qwen3.5-9B"
DEFAULT_LORA_PATH = (
    "/home/LXJ/Python_Projects/Agent_Lightning/LlamaFactory/"
    "image_restoration_experts/outputs/qwen3_5_0721/format_cold_start/fog"
)

# ── 16 个合法动作 ─────────────────────────────────────────────────────────────
VALID_ACTIONS = [
    "real_esrgan", "scunet", "retinexformer_fivek", "hvicidnet", "lightdiff",
    "turbo_rain", "s2former", "idt", "ridcp", "kanet", "turbo_snow",
    "snowmaster", "nafnet_denoise", "focalnet_dehaze", "focalnet_desnow",
    "mb_taylorformer_dehaze",
]

# ── 内置示例文本（可替换为真实 rollout 内容）────────────────────────────────
EXAMPLE_TEXT = """\
<think>
Let me analyze this image carefully. The image shows significant atmospheric haze \
with reduced visibility and a milky white overlay across the entire scene. \
Distant objects are barely visible and the contrast is very low.

For fog and haze degradation, I need to choose the right tool. The available \
dehaze tools are focalnet_dehaze and mb_taylorformer_dehaze. FocalNet architecture \
has attention mechanisms specifically designed for non-uniform haze patterns, \
while MB-TaylorFormer uses a multi-scale approach.

Given the uniform haze distribution in this image, I will start with focalnet_dehaze \
as it tends to perform well on standard fog conditions.
</think>
<tool_call>{"action": "focalnet_dehaze"}</tool_call>\
"""


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def find_thinking_span(token_ids: list[int], tokenizer) -> tuple[int, int]:
    """
    在 token 序列中定位 <think>...</think> 内容区间（不含标签本身）。
    返回 (start, end)，end 是 exclusive index。
    若找不到则返回 (-1, -1)。
    """
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    start_tag = "<think>"
    end_tag = "</think>"

    char_start = text.find(start_tag)
    char_end = text.find(end_tag)
    if char_start == -1 or char_end == -1:
        return -1, -1

    # 内容区间（标签之后 / 标签之前）
    content_char_start = char_start + len(start_tag)
    content_char_end = char_end

    # 用 batch_encoding 把 char offset 转成 token index
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]  # list of (char_start, char_end) per token

    tok_start, tok_end = -1, -1
    for i, (cs, ce) in enumerate(offsets):
        if tok_start == -1 and ce > content_char_start:
            tok_start = i
        if cs < content_char_end:
            tok_end = i + 1  # exclusive

    return tok_start, tok_end


def find_action_token_pos(token_ids: list[int], tokenizer, action_name: str) -> int:
    """
    找到 action_name 在 token 序列中最后一次出现时的起始 token 位置。
    （选"最后一次"是因为 tool_call JSON 在序列末尾。）
    """
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    # 在 tool_call JSON 里找 action_name
    search_str = f'"{action_name}"'
    char_pos = text.rfind(search_str)
    if char_pos == -1:
        # 也尝试不带引号
        char_pos = text.rfind(action_name)
    if char_pos == -1:
        return -1

    # +1 跳过开头的引号，定位到 action_name 本身
    action_char_pos = char_pos + 1 if text[char_pos] == '"' else char_pos

    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    for i, (cs, ce) in enumerate(enc["offset_mapping"]):
        if cs <= action_char_pos < ce or cs == action_char_pos:
            return i
    return -1


def compute_attribution(
    model,
    tokenizer,
    text: str,
    action_name: str,
    device: str,
) -> dict:
    """
    核心计算：
      1. 把文本转成 input embeddings（requires_grad=True）
      2. forward → 取 action token 位置的 log prob
      3. backward → 读每个位置的 embedding 梯度范数
    返回包含 tokens/importance/region indices 的 dict。
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]  # [seq_len]
    token_list = input_ids.tolist()
    seq_len = len(token_list)

    think_start, think_end = find_thinking_span(token_list, tokenizer)
    action_pos = find_action_token_pos(token_list, tokenizer, action_name)

    print(f"  序列长度        : {seq_len} tokens")
    if think_start != -1:
        print(f"  thinking 区间   : [{think_start}, {think_end})  "
              f"({think_end - think_start} tokens)")
    else:
        print("  ⚠️  未找到 <think>...</think> 标签，将在全序列上计算归因")
        think_start, think_end = 0, seq_len

    if action_pos == -1:
        raise ValueError(f"在文本中未找到动作名 '{action_name}'，请检查 --action 参数")

    print(f"  动作 token 位置 : {action_pos}  "
          f"(token: '{tokenizer.decode([token_list[action_pos]])}', "
          f"log prob 预测位置 = {action_pos - 1})")

    # ── 获取 embeddings，设置梯度追踪 ─────────────────────────────────────────
    embed_layer = model.get_input_embeddings()
    with torch.no_grad():
        embeddings_raw = embed_layer(input_ids.unsqueeze(0))  # [1, seq, d]

    embeddings = embeddings_raw.detach().clone().requires_grad_(True)

    # ── Forward ───────────────────────────────────────────────────────────────
    with torch.enable_grad():
        out = model(inputs_embeds=embeddings, use_cache=False)
        logits = out.logits  # [1, seq, vocab]

        # logits[0, action_pos-1, :] 预测 action_pos 处的 token
        pred_pos = action_pos - 1
        log_probs = torch.nn.functional.log_softmax(logits[0, pred_pos], dim=-1)
        target_lp = log_probs[token_list[action_pos]]

        print(f"  log P(action token) = {target_lp.item():.4f}  "
              f"  P = {target_lp.exp().item():.4f}")

        # ── Backward ─────────────────────────────────────────────────────────
        target_lp.backward()

    grad = embeddings.grad[0]               # [seq, d]
    importance = grad.norm(dim=-1).cpu().float().numpy()  # [seq]

    return {
        "tokens": token_list,
        "importance": importance,
        "think_start": think_start,
        "think_end": think_end,
        "action_pos": action_pos,
        "action_log_prob": target_lp.item(),
    }


# ── 可视化 ────────────────────────────────────────────────────────────────────

def visualize_text(result: dict, tokenizer, top_k: int = 20) -> None:
    """终端文本可视化：打印 thinking 区域 Top-K 高重要性位置及热图。"""
    tokens = result["tokens"]
    importance = result["importance"]
    ts, te = result["think_start"], result["think_end"]
    action_pos = result["action_pos"]

    think_imp = importance[ts:te].copy()
    imp_min, imp_max = think_imp.min(), think_imp.max()
    think_imp_norm = (think_imp - imp_min) / (imp_max - imp_min + 1e-8)

    # ── Top-K ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  梯度归因结果  |  Top-{top_k} 高重要性 thinking token 位置")
    print(f"  动作: '{tokenizer.decode([tokens[action_pos]])}'  "
          f"log P = {result['action_log_prob']:.4f}")
    print("=" * 72)

    top_rel = np.argsort(think_imp_norm)[::-1][:top_k]
    for rank, rel_i in enumerate(top_rel):
        abs_i = ts + rel_i
        tok_str = tokenizer.decode([tokens[abs_i]]).replace("\n", "↵")
        score = think_imp_norm[rel_i]
        bar = "█" * int(score * 28) + "░" * (28 - int(score * 28))

        # 前后各 3 个 token 的上下文
        ctx = tokenizer.decode(
            tokens[max(ts, abs_i - 3): min(te, abs_i + 4)]
        ).replace("\n", " ")
        print(f"  #{rank + 1:2d} [tok {abs_i:4d}] |{bar}| {score:.3f}"
              f"  '{tok_str}'  «…{ctx}…»")

    # ── 全 thinking 区域热图（字符版）────────────────────────────────────────
    print("\n" + "-" * 72)
    print("  全 thinking 区域重要性热图  (░▒▓█ = 低→高):\n")
    chars = " ░▒▓█"
    heat_line = ""
    text_line = ""
    line_width = 80
    for rel_i in range(te - ts):
        score = think_imp_norm[rel_i]
        heat_ch = chars[min(4, int(score * 5))]
        tok_ch = tokenizer.decode([tokens[ts + rel_i]])
        # 换行处理
        if "\n" in tok_ch:
            print(heat_line)
            print(text_line)
            heat_line = ""
            text_line = ""
            continue
        heat_line += heat_ch * max(1, len(tok_ch))
        text_line += tok_ch
        if len(text_line) >= line_width:
            print(heat_line[:line_width])
            print(text_line[:line_width])
            heat_line = heat_line[line_width:]
            text_line = text_line[line_width:]
    if text_line:
        print(heat_line)
        print(text_line)
    print()


def visualize_plot(result: dict, tokenizer, save_path: str) -> None:
    """用 matplotlib 生成归因热图并保存为图片。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    tokens = result["tokens"]
    importance = result["importance"]
    ts, te = result["think_start"], result["think_end"]

    think_imp = importance[ts:te].copy()
    imp_norm = (think_imp - think_imp.min()) / (think_imp.max() - think_imp.min() + 1e-8)

    # 把 token 文本整理成可显示的形式
    labels = [tokenizer.decode([tokens[ts + i]]).replace("\n", "↵")
              for i in range(te - ts)]

    n = len(labels)
    cols = 60
    rows = (n + cols - 1) // cols

    fig, ax = plt.subplots(figsize=(cols * 0.22, rows * 0.6 + 1.5))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis("off")
    ax.set_title(
        f"Gradient Attribution — action: '{tokenizer.decode([tokens[result['action_pos']]])}'"
        f"  log P = {result['action_log_prob']:.3f}",
        fontsize=10, pad=8,
    )

    cmap = plt.cm.YlOrRd
    for idx in range(n):
        row = rows - 1 - idx // cols
        col = idx % cols
        score = imp_norm[idx]
        color = cmap(score)
        ax.add_patch(plt.Rectangle((col, row), 1, 0.8, color=color))
        text_color = "white" if score > 0.6 else "black"
        ax.text(col + 0.5, row + 0.4, labels[idx],
                ha="center", va="center", fontsize=5, color=text_color)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.02,
                 fraction=0.02, label="Normalized gradient norm")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  热图已保存到: {save_path}")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="用梯度归因验证 Qwen3.5 thinking chain 中的工具选择决策点"
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH,
                        help="HuggingFace 模型路径")
    parser.add_argument("--lora_path", default=None,
                        help="LoRA adapter 路径（可选，不加载则用 base model）")
    parser.add_argument("--text", default=None,
                        help="直接传入生成的文本（thinking + tool_call）")
    parser.add_argument("--text_file", default=None,
                        help="从文件读取生成的文本")
    parser.add_argument("--action", default="focalnet_dehaze",
                        choices=VALID_ACTIONS,
                        help="该文本中选择的动作名")
    parser.add_argument("--device", default="cuda:0",
                        help="运行设备，例如 cuda:0 或 cuda:1")
    parser.add_argument("--top_k", type=int, default=20,
                        help="显示前 K 个高重要性 token")
    parser.add_argument("--save_plot", default=None,
                        help="若指定，将归因热图保存为此路径的 PNG 文件")
    args = parser.parse_args()

    # ── 确定输入文本 ──────────────────────────────────────────────────────────
    text = args.text
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()
    if text is None:
        print("未指定 --text 或 --text_file，使用内置示例文本。\n")
        text = EXAMPLE_TEXT

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    print(f"加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    print(f"加载模型 (bfloat16, device={args.device}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    if args.lora_path:
        try:
            from peft import PeftModel
            print(f"加载 LoRA adapter: {args.lora_path}")
            model = PeftModel.from_pretrained(model, args.lora_path)
        except ImportError:
            print("⚠️  peft 未安装，跳过 LoRA 加载。pip install peft")

    # ── 计算归因 ──────────────────────────────────────────────────────────────
    print(f"\n计算梯度归因 (action='{args.action}') ...")
    result = compute_attribution(model, tokenizer, text, args.action, args.device)

    # ── 输出结果 ──────────────────────────────────────────────────────────────
    visualize_text(result, tokenizer, top_k=args.top_k)

    if args.save_plot:
        visualize_plot(result, tokenizer, args.save_plot)

    # ── 打印简要结论 ──────────────────────────────────────────────────────────
    ts, te = result["think_start"], result["think_end"]
    think_imp = result["importance"][ts:te]
    imp_norm = (think_imp - think_imp.min()) / (think_imp.max() - think_imp.min() + 1e-8)

    # 最高重要性位置在 thinking chain 里的相对位置（0=开头, 1=结尾）
    peak_rel = int(np.argmax(imp_norm))
    peak_ratio = peak_rel / max(1, te - ts - 1)
    peak_token = tokenizer.decode([result["tokens"][ts + peak_rel]]).replace("\n", "↵")
    top10_tokens = [
        tokenizer.decode([result["tokens"][ts + i]]).replace("\n", "↵")
        for i in np.argsort(imp_norm)[::-1][:10]
    ]

    print("\n" + "=" * 72)
    print("  结论摘要")
    print("=" * 72)
    print(f"  最高重要性 token : '{peak_token}'  "
          f"(位于 thinking 的 {peak_ratio * 100:.0f}% 处)")
    print(f"  Top-10 token 内容: {top10_tokens}")

    if peak_ratio > 0.8:
        print("\n  ⚠  决策峰值集中在 thinking 末尾 → 模型在结论阶段才定向，"
              "thinking 早期信息贡献少")
    elif peak_ratio < 0.3:
        print("\n  ⚠  决策峰值集中在 thinking 前段 → 模型很早就确定了动作")
    else:
        print("\n  ✓  决策峰值分布在 thinking 中间段，与分析过程对应")

    print()


if __name__ == "__main__":
    main()
