"""Render the audited new-SFT experiment as a compact Chinese research report."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
s = json.loads((root/'summary.json').read_text())
assert s['audit_complete']
a = s['attribution']; own = a['sft_self_768']; f = own['objectives']['full']
font = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                     'axes.unicode_minus': False, 'font.size': 10, 'savefig.dpi': 180})
pdfmetrics.registerFont(TTFont('CJK', font))
pdfmetrics.registerFontFamily('CJK', normal='CJK', bold='CJK', italic='CJK', boldItalic='CJK')
styles = {name: ParagraphStyle(name, fontName='CJK', fontSize=size, leading=lead,
            textColor=colors.HexColor('#18354b'), wordWrap='CJK', spaceAfter=8)
          for name, size, lead in [('title', 23, 32), ('h1', 16, 24), ('h2', 12, 18),
                                  ('body', 10, 16), ('small', 8, 12), ('cell', 8.5, 13)]}
story = []


def p(text, style='body'):
    story.append(Paragraph(text, styles[style]))


def table(rows, widths):
    tab = Table([[Paragraph(html.escape(str(c)).replace('\n', '<br/>'), styles['cell']) for c in row]
                 for row in rows], colWidths=widths, hAlign='LEFT')
    tab.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5edf5')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f4f7fa')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7), ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    story.extend([tab, Spacer(1, 10)])


def page(title):
    story.append(PageBreak()); p(title, 'h1')


def rate(n, d):
    return f'{n}/{d} ({n/d:.1%})' if d else '-'


def hit(key, obj='full'):
    r = a[key]['objectives'][obj]
    return rate(r.get('peak_first_mention', 0), r['with_literal'])


def effect(r):
    return f"{r['mean']*100:+.2f} [{r['ci95'][0]*100:+.2f}, {r['ci95'][1]*100:+.2f}]"


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#ccd9e5')); canvas.line(44, 43, 551, 43)
    canvas.setFont('CJK', 8); canvas.setFillColor(colors.HexColor('#617487'))
    canvas.drawString(44, 28, 'ALFWorld | action_explicit_v1 SFT step 100 | 2026-10-09')
    canvas.drawRightString(551, 28, str(doc.page))


p('最新 SFT 模型<br/>动作选择梯度归因实验', 'title')
p('Qwen3.5-2B · action_explicit_v1 · checkpoint-100', 'h2')
p('本次 SFT 从原始发布权重开始，监督思考句为 I will choose the action “具体动作”。使用最终 step 100 的 LoRA adapter 合并完整模型，冻结权重复跑；该模型未经过本项目 RL。')
p('主要结果', 'h2')
p(f"模型学会了简短且显式包含动作的回答。在相同 360 个上下文、每个上下文两次采样和 768-token 预算下，合法动作率为 {rate(s['fresh_sft']['legal'], 720)}，原始模型为 {rate(s['fresh_raw']['legal'], 720)}。SFT 的 720 次回答均结束思考，无长度截断。这里的合法性只指 XML 可解析且动作在允许列表内。")
p(f"首抽中 {own['n']} 个回答可归因，全部在思考中恰好提及一次最终动作。完整动作 logP 的梯度峰值落入该动作的比例是 {hit('sft_self_768')}，落在动作首 token 的比例是 {rate(f['peak_first_mention_first_token'], f['with_literal'])}；峰值落在思考开头三个 token 的比例为 {rate(f['peak_first_3_tokens'], f['n'])}。")
p(f"对完全相同的新回答，原始权重的动作范围命中率仅 {hit('raw_on_sft_self')}。因此本次显式动作 SFT 确实改变了这批回答上的梯度位置，而不是仅由回答格式或长度变化造成。")
top = '、'.join(f'{token.strip()!r}：{count}' for token, count in f['top_peak_tokens'][:4])
p('最常见峰值 token：'+html.escape(top)+'。')
e = s['self_interventions']['effects']['first_replacement']['p_original']
p(f"文本替换产生强烈效果：对 24 个自身回答，将思考中的完整动作替换成另一合法动作后，原动作概率变化为 {effect(e)} 个百分点（任务聚类 95% 区间）。这证明该动作文本在这些条件续写中影响最终选择，但不等价于它具有最大的输入梯度。")
table([['分析部分', '数量'], ['相同旧文本的三种权重对照', '357 主样本 + 60 多次提及富集样本'],
       ['新 SFT 自身回答梯度 / 原始权重交叉评分', f'{own["n"]} / {a["raw_on_sft_self"]["n"]} 个相同文本'],
       ['自身回答干预 / 旧回答干预', f'{s["self_interventions"]["draws"]} / {s["controlled_interventions"]["draws"]} 次条件续写'],
       ['固定后缀评分 / 数值复核', '407 个条件 / 两组各 12 个样本']], [318, 189])
p('结论边界：没有执行完整 ALFWorld 环境轨迹，不能据此报告任务成功率或熵正则的 RL 收益。', 'small')

page('1. 梯度峰值是否移到了动作上？')
p('目标为 XML 中实际动作字符串的 logP 之和；对思考段每个输入 embedding 求梯度并取 L2 范数，在思考段内寻找峰值。对象 / 位置、动作首 token、动作间 logP 差和 |gradient × input| 作为辅助目标。')
rows = [['固定文本组', '旧 SFT + RL666', '原始权重', '新 SFT']]
for label, suffix in [('主样本（含动作提及 19 个）', 'main'), ('多次提及（60 个）', 'enriched')]:
    rows.append([label, hit('grpo_'+suffix), hit('raw_controlled_'+suffix), hit('sft_controlled_'+suffix)])
table(rows, [177, 110, 110, 110])
p('表中为“峰值落入首次完整动作提及”的次数 / 含动作提及的次数。固定旧文本可隔离权重变化；主样本与富集样本分开统计。旧 SFT + RL666 的 SFT 数据与本次不同，因此两者不是纯 RL 消融。', 'small')
rows = [['同一批新 SFT 回答', '原始权重评分', '新 SFT 权重评分']]
for label, obj in [('完整动作 logP', 'full'), ('动作首 token logP', 'first'), ('同动词动作差', 'contrast_same_verb'),
                   ('异动词动作差', 'contrast_different_verb'), ('|gradient × input|', 'abs_grad_input_full')]:
    rows.append([label, hit('raw_on_sft_self', obj), hit('sft_self_768', obj)])
table(rows, [209, 149, 149])
p('这一变化依赖归因目标：同动词候选的 logP 差目标并未更集中于动作提及，异动词差目标也基本不变。因此完整命令 logP 的 82.0% 不能推广为所有动作区分目标的结论。', 'small')
delta = s['paired_weight_changes']['sft_self']['peak_first_mention']
p(f"同一文本上，新 SFT 相对原始权重的首次动作提及命中率变化为 {effect(delta)} 个百分点。各区间以任务为聚类单位，10,000 次 bootstrap；样本内 token 不当作独立实验。")
multi = own['multiword_actions']['full']
p(f"首 token 结论需特别谨慎：去掉 look、inventory 等单词动作后，剩余 {multi['n']} 个多词动作。完整命令目标的峰值落入动作范围为 {rate(multi['peak_first_mention'], multi['with_literal'])}，恰好落在动作首 token 则仅 {rate(multi['peak_first_mention_first_token'], multi['with_literal'])}。最大位置经常是对象或位置词，而不是动作开头。")
p(f"新 SFT 自身思考长度中位数为 {own['thinking_length']['median']:.0f} token，原始模型 768-token 下可归因的 {a['raw_self_768']['n']} 个回答为 {a['raw_self_768']['thinking_length']['median']:.0f} token。两组自然回答的长度和有效样本筛选不同，不能把它们之间的差异全部解释为权重变化。")
nonoverlap = a['sft_self_nonoverlap_prompts']
p(f"排除与 SFT 训练集或验证集完全相同的提示词后，保留 {nonoverlap['n']} 个自身回答；完整动作峰值命中率为 {hit('sft_self_nonoverlap_prompts')}。主分析中的 360 个上下文来自 96 个 valid_seen 任务，有 29 个提示词匹配训练集、1 个匹配 SFT 验证集；按任务 / trial ID 比较未发现相同任务。", 'small')

page('2. 改写思考中的动作会改变最终输出吗？')
conds = s['self_interventions']['conditions_summary']
labels = [('写入动作前', 'prefixes/before_first_mention'), ('写入原动作后', 'prefixes/after_first_mention'),
          ('完整思考后', 'prefixes/after_thinking'), ('改写成替代动作后', 'continuations/replace_first_mention')]
rows = [['续写起点 / 条件', '原动作率', '替代动作率', '合法率']]
for label, key in labels:
    rows.append([label]+[f"{conds[key][metric]['mean']:.2%}" for metric in ['p_original', 'p_alternative', 'valid_rate']])
table(rows, [207, 100, 100, 100])
p('24 个样本按六类任务分层、每类最多四个不同 session，以固定种子预先选取。每种条件采样 16 次，替换条件与原样前缀使用相同采样种子；分母包含所有非法输出。替代动作优先保持同一动词。')
assets = root/'report_figures'; assets.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8.4, 2.3))
vals = [conds[k]['p_original']['mean']*100 for _, k in labels]
bars = ax.barh(range(4), vals, color=['#819db8', '#3276a5', '#3276a5', '#d38744'])
ax.set(yticks=range(4), yticklabels=[x for x, _ in labels], xlim=(0, 112), xlabel='原动作选择率 (%)')
ax.invert_yaxis(); ax.grid(axis='x', alpha=.15)
for b, value in zip(bars, vals):
    ax.text(value+1, b.get_y()+b.get_height()/2, f'{value:.2f}%', va='center', fontsize=9)
fig.tight_layout(); fig.savefig(assets/'self_interventions.png', bbox_inches='tight'); plt.close(fig)
story.append(Image(str(assets/'self_interventions.png'), width=507, height=148))
p('这些自身回答都只提及一次完整动作，因此“首次”和“末次”是同一个位置，本次不对它们作虚假的排序比较。替换的是完整动作字符串，并非仅替换首 token；结果不能证明首 token 单独决定动作或应单独施加熵正则。')
table([['旧文本的配对干预', '原动作率变化 / 95% 区间（百分点）'],
       ['替换首次动作提及', effect(s['controlled_interventions']['effects']['first_replacement']['p_original'])],
       ['替换末次动作提及', effect(s['controlled_interventions']['effects']['last_replacement']['p_original'])]], [208, 299])
p('旧文本干预保留了旧模型较长的自然思考，其条件分布不同于新 SFT 的短句。区间按任务聚类，但不单独估计 16 次采样的 Monte Carlo 噪声。高梯度代表局部 embedding 敏感性，文本替换测量离散扰动下的输出变化，二者不可互相替代。', 'small')

page('3. 可靠性、限制与复现')
rows = [['数值检查', '固定旧文本', 'SFT 自身文本']]
for label, fun in [('BF16 / FP32 峰值一致', lambda r: f"{r['same_peak']}/{r['n']}"),
                   ('梯度范数序列最小余弦相似度', lambda r: f"{r['min_bf16_fp32_cosine']:.5f}"),
                   ('有限差分相对误差中位数 ε=.005', lambda r: f"{r['finite_difference_error_by_epsilon']['0.005']['median']:.2%}"),
                   ('ε=.005 最大误差', lambda r: f"{r['finite_difference_error_by_epsilon']['0.005']['max']:.2%}"),
                   ('目标之后的最大梯度', lambda r: f"{r['max_future_gradient']:.2e}")]:
    rows.append([label, fun(s['numerical_controlled']), fun(s['numerical_self'])])
table(rows, [257, 125, 125])
p(f"HF 归因与 SGLang 生成后端核对了 {s['backend']['n']} 个样本的目标 token ID，全部一致；每个样本的最大 token logP 差的中位数为 {s['backend']['median_case_max_token_logp_difference']:.4f} nats，整体最大值为 {s['backend']['max_token_logp_difference']:.4f} nats。BF16 内核并非逐位一致。")
precision = s['precision']; pf = precision['attribution']['objectives']['full']
p(f"新 SFT 的动作概率接近 1，原 FP32 log-softmax 有消减误差。为此额外复算全部 {own['n']} 个自身回答：FP32 模型 + FP64 log-softmax，关闭 TF32。动作范围峰值仍为 {rate(pf['peak_first_mention'], pf['with_literal'])}，与 BF16 的逐样本峰值一致 {precision['bf16_fp32_full_peak_agreement']}/{own['n']}；同一 12 例在 ε=.005 下的有限差分误差中位数降为 {precision['finite_difference_error_by_epsilon']['0.005']['median']:.2%}，最大为 {precision['finite_difference_error_by_epsilon']['0.005']['max']:.2%}。")
p('ε=.001、.005、.02 的全部检查和不利结果均保留，不按结果筛选步长。本次未复跑积分梯度，因此不作积分梯度结论。', 'small')
p('模型与复现', 'h2')
p('SFT run：'+html.escape(Path(s['protocol']['sft_run']).name), 'small')
p('Adapter SHA256：'+s['protocol']['adapter_sha256'], 'small')
p(f"训练完成于 step 100，约 {s['protocol']['training_results']['epoch']:.3f} epoch；训练 loss {s['protocol']['training_results']['train_loss']:.5f}，验证 loss {s['protocol']['training_results']['eval_loss']:.5f}。这些是 teacher-forcing 损失，不是任务成功率。", 'small')
p('模型来源及导出校验和见 model_manifest.json；汇总见 summary.json；逐 token 梯度与目标得分见 controlled_*.jsonl、attribution_self.jsonl 和 raw_on_sft_self.jsonl。全部采样结果、非法输出、固定干预设计、数值检查及运行日志均保留。', 'small')
p('实验目录：'+html.escape(str(root)), 'small')
p('复现命令见同目录 REPRODUCE.txt，执行过的精确命令见 execution_logs/commands.jsonl。脚本位于 examples/alfworld_grpo_baseline/scripts/thinking_attribution/，本次版本及哈希保存在 source_snapshot/ 和 source_manifest.json。', 'small')
p('本次仅使用物理 GPU 0 和 3。实验服务已停止，模型无训练更新。对照旧实验不会区分“旧 SFT 的影响”与“RL 的影响”；对照原始权重也不表示原始发布模型未经厂商后训练。', 'small')
args.output.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(str(args.output), pagesize=(595.28, 841.89), leftMargin=44, rightMargin=44,
    topMargin=40, bottomMargin=57, title='最新 SFT 动作选择梯度归因实验', author='Codex').build(
    story, onFirstPage=footer, onLaterPages=footer)
print(args.output)
