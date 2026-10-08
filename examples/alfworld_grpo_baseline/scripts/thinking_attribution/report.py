"""Render the completed local attribution experiment as a Chinese PDF report."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
summary = json.loads((root / 'summary.json').read_text())
assert summary['audit']['all_checks_passed']
main = summary['cohorts']['main']
extra = summary['cohorts']['enriched']
assets = root / 'report_figures'
assets.mkdir(exist_ok=True)
font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
fp = FontProperties(fname=font_path)
plt.rcParams.update({'font.family': fp.get_name(), 'axes.unicode_minus': False,
                     'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 180})
from matplotlib import font_manager
font_manager.fontManager.addfont(font_path)
blue, orange, dark = '#2463AA', '#D98232', '#18334F'

# Figure 1: coverage is a census, distinct from the selected intervention cases.
fig, ax = plt.subplots(figsize=(9, 1.45))
ax.barh([0], [100], color='#E5EDF5', height=.42)
ax.barh([0], [1143 / 43586 * 100], color=blue, height=.42)
ax.text(5, 0, 'thinking 中完整动作的字面提及：2.62%', va='center', color=dark)
ax.set(xlim=(0, 100), yticks=[], xlabel='全部验证决策占比（%）')
ax.spines['left'].set_visible(False)
fig.tight_layout()
fig.savefig(assets / 'coverage.png', bbox_inches='tight')
plt.close(fig)

# Figure 2: causal text interventions, each compared to its matched intact prefix.
fig, ax = plt.subplots(figsize=(9, 3.5))
keys = ['first_replacement_matched', 'last_replacement_matched', 'high_deletion_matched', 'low_deletion_matched']
labels = ['替换首次动作提及', '替换末次动作提及', '删除梯度最高词', '删除匹配低梯度词']
for index, key in enumerate(keys):
    value = extra['paired_effects'][key]['p_original']
    mean = value['mean'] * 100
    lo, hi = np.array(value['ci95']) * 100
    ax.errorbar(mean, 3-index, xerr=[[mean-lo], [hi-mean]], fmt='o', capsize=5,
                color=blue if index < 2 else orange, markersize=7)
    ax.text(34, 3-index, f'{mean:+.1f} [{lo:+.1f}, {hi:+.1f}]', va='center', fontsize=10)
ax.axvline(0, color='#8B98A5', linestyle='--', linewidth=1)
ax.set(yticks=range(4), yticklabels=labels[::-1], xlim=(-40, 70),
       xlabel='原动作选择率变化（百分点）；横线为任务聚类 95% 区间')
ax.grid(axis='x', alpha=.15)
fig.tight_layout()
fig.savefig(assets / 'causal_effects.png', bbox_inches='tight')
plt.close(fig)

# Figure 3: progressive commitment, full legal commands rather than wording diversity.
prefix_keys = ['before_first_mention', 'after_first_mention', 'after_last_mention', 'after_thinking', 'before_schema_action']
labels = ['首次提及前', '首次提及后', '末次提及后', 'thinking 结束', 'XML 动作前']
fig, axes = plt.subplots(1, 2, figsize=(10, 3.45))
for ax, metric, label, scale in zip(axes, ['p_original', 'entropy'], ['原动作选择率（%）', '合法完整动作的经验熵（nats）'], [100, 1]):
    values = [extra['condition_intervals']['prefixes/'+k][metric] for k in prefix_keys]
    means = np.array([v['mean'] for v in values]) * scale
    lower = np.array([v['ci95'][0] for v in values]) * scale
    upper = np.array([v['ci95'][1] for v in values]) * scale
    ax.plot(range(5), means, marker='o', color=blue)
    ax.fill_between(range(5), lower, upper, alpha=.14, color=blue)
    ax.set(xticks=range(5), xticklabels=labels, ylabel=label)
    ax.tick_params(axis='x', rotation=25, labelsize=9)
    ax.grid(alpha=.15)
axes[0].set_ylim(0, 105)
axes[1].set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(assets / 'commitment.png', bbox_inches='tight')
plt.close(fig)

# Figure 4: a concrete short example, retaining all tokens in original order.
attrs = [json.loads(line) for name in ['attribution_0.jsonl', 'attribution_1.jsonl', 'attribution_enriched_0.jsonl'] for line in (root/name).open()]
example = next(r for r in attrs if r['sample_id'] == '8a0a4213-1b98-44a7-81f9-0bb4036f3a2c_5_2')
fig, ax = plt.subplots(figsize=(9.2, 2.45))
values = np.array(example['grad_norms']['full'])
ax.plot(range(len(values)), values, color=blue, linewidth=1.4)
for index, (start, end) in enumerate(example['exact_mentions']):
    token_indices = [i for i, (s, e) in enumerate(example['thinking_offsets']) if e > start and s < end]
    ax.axvspan(min(token_indices)-.5, max(token_indices)+.5, color=orange, alpha=.18,
               label='最终动作的字面提及' if index == 0 else None)
peak = int(np.argmax(values))
ax.annotate('峰值 token: '+example['thinking_tokens'][peak].strip(), (peak, values[peak]),
            xytext=(max(peak-30, 0), values[peak]*1.12), fontsize=10,
            arrowprops={'arrowstyle': '->', 'color': dark})
ax.set(xlabel='thinking token 位置（从 0 开始）', ylabel='梯度 L2 范数', ylim=(0, values.max()*1.4))
ax.legend(frameon=False, loc='upper left', fontsize=9)
fig.tight_layout()
fig.savefig(assets / 'example_attribution.png', bbox_inches='tight')
plt.close(fig)

pdfmetrics.registerFont(TTFont('CJK', font_path))
pdfmetrics.registerFontFamily('CJK', normal='CJK', bold='CJK', italic='CJK', boldItalic='CJK')
styles = {
    'title': ParagraphStyle('title', fontName='CJK', fontSize=25, leading=34, textColor=colors.HexColor(dark), spaceAfter=12),
    'subtitle': ParagraphStyle('subtitle', fontName='CJK', fontSize=11, leading=18, textColor=colors.HexColor('#586A7B'), spaceAfter=16),
    'h1': ParagraphStyle('h1', fontName='CJK', fontSize=18, leading=26, textColor=colors.HexColor(dark), spaceAfter=13),
    'h2': ParagraphStyle('h2', fontName='CJK', fontSize=12, leading=18, textColor=colors.HexColor(blue), spaceBefore=8, spaceAfter=6),
    'body': ParagraphStyle('body', fontName='CJK', fontSize=10, leading=16, textColor=colors.HexColor('#243649'), wordWrap='CJK', spaceAfter=8),
    'small': ParagraphStyle('small', fontName='CJK', fontSize=8.2, leading=12.8, textColor=colors.HexColor('#526475'), wordWrap='CJK', spaceAfter=7),
    'cell': ParagraphStyle('cell', fontName='CJK', fontSize=9, leading=13, wordWrap='CJK', textColor=colors.HexColor('#243649')),
    'code': ParagraphStyle('code', fontName='CJK', fontSize=8.6, leading=14, wordWrap='CJK', textColor=colors.HexColor('#243649'), backColor=colors.HexColor('#EEF3F8'), borderPadding=7, spaceAfter=10),
}
story = []
width = 507


def p(text, style='body'):
    story.append(Paragraph(text, styles[style]))


def table(rows, widths):
    wrapped = [[Paragraph(html.escape(str(cell)).replace('\n', '<br/>'), styles['cell']) for cell in row] for row in rows]
    t = Table(wrapped, colWidths=widths, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E6EEF7')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F6F8FB')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 9),
        ('RIGHTPADDING', (0, 0), (-1, -1), 9), ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor('#C4D5E7'))]))
    story.append(t)
    story.append(Spacer(1, 10))


def figure(name, max_height=210):
    path = assets/name
    w, h = ImageReader(str(path)).getSize()
    scale = min(width/w, max_height/h)
    story.append(Image(str(path), width=w*scale, height=h*scale, hAlign='CENTER'))
    story.append(Spacer(1, 6))


def page(title):
    if story:
        story.append(PageBreak())
    p(title, 'h1')


def effect(value):
    lo, hi = value['ci95']
    return f"{value['mean']*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}]"


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#D6E1EC'))
    canvas.line(44, 43, 551, 43)
    canvas.setFont('CJK', 8)
    canvas.setFillColor(colors.HexColor('#617487'))
    canvas.drawString(44, 28, 'ALFWorld · checkpoint 666 · thinking 动作选择归因验证 · 2026-10-08')
    canvas.drawRightString(551, 28, str(doc.page))


p('ALFWorld thinking<br/>动作选择归因验证', 'title')
p('实验结果报告 · Qwen3.5-2B · GRPO step 666<br/>实验资源：物理 GPU 0、GPU 3；冻结权重，未进行训练更新。', 'subtitle')
table([['验证决策', '梯度分析', '干预条件', '动作续写'],
       ['43,586 个', '417 个决策', '407 个固定后缀条件', '14,384 次']], [127, 127, 127, 126])
p('核心判断', 'h2')
p('结果支持把研究重点放在 thinking 中动作意图形成的位置。对于明确写出动作的样本，首次和末次动作提及都会影响最终选择；但本次实验没有证据证明“首次提及比末次更强”，也没有支持“首次提及是普遍梯度峰值”。')
p('1. 在 24 个多次提及样本中，替换首次提及使原动作选择率平均下降 18.23 个百分点；替换末次提及下降 21.88 个百分点。两者差异的区间跨过 0。')
p('2. 在这一条件样本中，thinking 结束后原动作选择率达到 99.48%，完整动作的经验熵降至 0.0157 nats，支持“动作在 XML 输出前已高度确定”的判断。该结论不能直接推广到所有决策。')
p('3. 首次完整动作的字面匹配只有 2.62% 的全量覆盖率。多数普通样本的梯度峰值落在 will、\'ll 等通用词上；不能把这些峰值自动当作适合施加探索熵的动作位置。')
figure('coverage.png', 108)
p('适用边界：这是单 checkpoint、离线单步的文本干预实验。未运行环境反事实轨迹，也未训练首 token 熵正则模型，因此不涉及任务成功率提升的结论。', 'small')

page('01  数据、目标与实验设计')
table([['层级', '样本与用途'],
       ['全量覆盖率', 'step 666 验证日志：1,120 条轨迹、43,586 个决策。完整动作字面提及 1,143 次；其中 43,571 个决策具有可提取动作和闭合 thinking。'],
       ['主样本', '6 类任务各选 20 条轨迹，尽量覆盖成功/失败和不同任务；每条选早、中、末 3 步。360 个候选中 357 个有效，来源于 120 条轨迹、96 个任务。'],
       ['条件样本', '另选 60 个多词动作且至少提及两次的决策，每类 10 个。专门研究首次/末次差异，不用于估计总体覆盖率。'],
       ['干预子集', '主样本 36 个决策 / 32 个任务；条件样本 24 个决策 / 20 个任务。每个生成条件采样 16 次，共 899 个条件。'],
       ['可靠性子集', '12 个决策做 FP32 / 有限差分 / 积分梯度；20 个决策、135 个动作 token 对照 HF 与 SGLang。']], [83, 424])
p('评分与归因目标', 'h2')
p('主目标是完整动作字符串的条件对数概率之和。对每个 thinking token 的输入词向量求梯度，记录 L2 范数及 gradient × input。辅助目标包含 schema 首 token，以及原动作相对同动词、不同动词合法替代动作的对数概率差。')
p('S(a) = sum_j log p(a_j | 完整提示词, thinking, XML 前缀, a_&lt;j)<br/>g_t = || dS(a) / de_t ||_2', 'code')
p('干预同时采用两种方式：保留原后缀并重新评分，或在改动位置后重新生成剩余 thinking 与 XML。首次/末次替换各自对照同一截断位置的未改动前缀；高/低梯度词删除也保留相同的后续锚点。')
p('生成设置：temperature=1、top_p=1、top_k=-1（不截断），总决策预算 768 tokens；已强制给定的前缀计入预算。不做动作约束解码，沿用原 XML 解析器并校验当前 admissible actions。', 'small')
p('统计：条件内先计算选择率，再做样本内配对差；按 game_file 聚类 bootstrap 10,000 次得到 95% 区间。区间未作多重比较校正，属于探索性证据。抽样及干预种子、全部设计修订均保存在 protocol.json。', 'small')

page('02  梯度峰值并不普遍落在首次提及')
table([['目标', '主样本：首次提及内峰值', '条件样本：首次提及内峰值'],
       ['完整动作 log P', '1 / 19（有提及的样本）', '2 / 60'],
       ['schema 首 token log P', '1 / 19', '1 / 60'],
       ['同动词替代动作的分数差', '0 / 4', '5 / 34'],
       ['不同动词替代动作的分数差', '1 / 19', '0 / 60']], [155, 176, 176])
p('上表的“首次提及内”指整段完整动作文字的 token 范围，不仅是其第一个 token。不同目标的有效分母不同；同动词比较只在存在合法同动词替代动作时计算。', 'small')
p('主样本 357 个决策中，完整动作梯度峰值有 281 个位于 will、27 个位于 \'ll；条件样本 60 个中有 56 个位于 Okay。这说明梯度 L2 会突出通用上下文位置，不能直接解释为该词决定了具体动作。')
p('改变评分目标可以改变归因位置：条件样本的同动词分数差有 5/34 个峰值落在首次动作提及范围，仍不足以确立统一位置。完整输出概率还会受语言形式、置信度和词向量尺度影响。')
figure('example_attribution.png', 190)
p('示例（为便于阅读选取较短样本，并非总体代表）：最终动作为 go to drawer 2，thinking 中出现两次。其最高梯度 token 是 drawer，而非动作的首 token go。橙色区域标出两次完整动作提及。', 'small')
p('该示例原文片段', 'h2')
p('“The current action I can take is `go to drawer 2` or similar. … Otherwise, I will use `go to drawer 2`.”', 'code')
p('因此，本次梯度证据不支持直接复用“找到首次完整动作，再固定正则它的第一个 token”的规则。梯度敏感性必须结合实际替换/续写检验；也不能把通用词峰值直接变成新的正则位置。')

page('03  首次与末次提及都有可测量的干预效果')
figure('causal_effects.png', 218)
rows = [['条件样本中的配对变化', '原动作选择率', '替代动作选择率', '合法输出率']]
for label, key in [('首次提及替换', 'first_replacement_matched'), ('末次提及替换', 'last_replacement_matched')]:
    values = extra['paired_effects'][key]
    rows.append([label, f"{values['p_original']['mean']*100:+.2f} pp",
                 f"{values['p_alternative']['mean']*100:+.2f} pp", f"{values['valid_rate']['mean']*100:+.2f} pp"])
table(rows, [150, 119, 119, 119])
comparison = extra['first_minus_last_effect']
p('首次效应减去末次效应：'+effect(comparison)+' 个百分点。区间包含 0，因此无法判断哪一个更强；末次的平均下降更大只是点估计。')
p('替代动作由当前合法动作集合选取，优先选择同动词、文字前缀最接近的候选。它不一定有助于完成任务。这一实验测量的是动作选择是否被改变，不是改后动作是否更优。')
p('替换后，原动作选择率下降约 18–22 个百分点，合法输出率只下降约 2–3 个百分点，同时指定替代动作的选择率上升。因此变化不只是 XML 失效，但仍需把格式变化单独计入结果。')
p('高梯度词与低梯度词的删除对照', 'h2')
p('主样本删除高梯度词的原动作选择率变化为 '+effect(main['paired_effects']['high_deletion_matched']['p_original'])+'；删除低梯度词为 '+effect(main['paired_effects']['low_deletion_matched']['p_original'])+' 个百分点。')
p('高减低的配对差为 '+effect(main['high_minus_low_deletion_effect'])+' 个百分点；条件样本的高梯度删除效应区间跨 0。最高梯度词的排序不能稳定替代具体动作语义的干预效果排序。', 'small')

page('04  动作在 thinking 中逐步确定；固定后缀会遮蔽影响')
figure('commitment.png', 205)
table([['给定前缀终点（条件样本）', '原动作选择率', '完整动作经验熵'],
       ['首次动作提及前', '58.07%', '0.8812 nats'],
       ['首次动作提及后', '67.71%', '0.6390 nats'],
       ['末次动作提及后', '93.49%', '0.0954 nats'],
       ['thinking 结束后', '99.48%', '0.0157 nats'],
       ['XML 中 action 参数开始前', '99.74%', '0.0097 nats']], [259, 124, 124])
p('这支持用户原先的动机：在明确陈述动作的 thinking 中，等到 schema 阶段再推动探索，往往已经较晚。但主样本在 thinking 结束后的原动作选择率仅 68.92%、经验熵 0.5661 nats；“schema 总是高度确定”并不成立。')
p('为何必须重新生成后缀', 'h2')
p('固定原后缀时，条件样本替换首次提及后的完整动作 log P 平均仅变化 +0.00084 nats；替换末次则为 -0.9140 nats。前者仍保留后面未改动的动作陈述，可能把影响覆盖掉。')
p('让改动后的剩余 thinking 自由续写，并匹配相同前缀终点后，首次替换也产生 -18.23 个百分点的选择率变化。固定后缀梯度/打分和自由续写回答的是不同问题，应同时保留。')
p('图中经验熵按合法完整命令的 16 次采样频率计算，非法输出单列。它不同于下一 token 全词表熵，存在小样本低估；浅色区间按任务聚类计算。不同截断点仅展示条件分布变化，不能单独证明自然思考过程的全部内部机制。', 'small')

page('05  数值可靠性与未通过的检查')
numeric = summary['numerical_checks']
backend = summary['backend_check']
table([['检查', '结果'],
       ['checkpoint 导出', '合并后的 HF 权重逐张量通过原 FSDP checkpoint 对照测试。'],
       ['BF16 / FP32 普通梯度', f"12/12 个决策峰值位置一致；梯度范数向量余弦相似度最小 {numeric['minimum_gradient_cosine']:.6f}。"],
       ['因果遮罩检查', f"目标之后位置的最大梯度 {numeric['maximum_causal_future_gradient']:.2e}，接近数值零。"],
       ['HF / SGLang 动作 log P', f"20 个决策、135 个 token；绝对差中位数 {backend['median_absolute_logprob_difference']:.2e}、均值 {backend['mean_absolute_logprob_difference']:.4f}、最大 {backend['maximum_absolute_logprob_difference']:.4f} nats。"]], [145, 362])
p('FP32 中心有限差分（沿最高梯度位置的归一化梯度方向）', 'h2')
fd_rows = [['epsilon', '12 个样本的相对误差中位数', '最大相对误差']]
for epsilon, values in numeric['finite_difference_errors_by_epsilon'].items():
    fd_rows.append([epsilon, f"{values['median']*100:.2f}%", f"{values['maximum']*100:.2f}%"])
table(fd_rows, [90, 230, 187])
p('积分梯度：保留失败结果，不作为普遍位置结论的依据', 'h2')
ig_rows = [['基线', '初始 BF16：误差 ≤10%', 'FP32：误差 ≤10%', 'FP32：综合验收']]
for baseline, label in [('zero', '全零词向量'), ('space', '空格词向量')]:
    first = summary['ig'][baseline]
    refined = summary['ig_refined'][baseline]
    ig_rows.append([label, f"{first['pass_relative_error_10pct']}/12", f"{refined['pass_relative_error_10pct']}/12", f"{refined['accepted']}/12"])
table(ig_rows, [105, 140, 135, 127])
p('初始积分采用 32 / 128 点；复核使用 FP32、Gauss–Legendre 求积、alpha=u^4，并在 64 点不满足阈值时增加到 256 点。综合验收要求完整性相对误差 ≤10%，或绝对误差 ≤0.02 nats。', 'small')
p('基线替换整段 thinking 的词向量，保留提示词与 schema；它会构造远离真实文本的输入，完整性失败说明当前数值积分不足以支持该路径的归因解释。普通梯度复核与文本干预是独立证据，不能用它们掩盖 IG 失败。')
p('空格基线复核中，6 个条件样本只有 1 个的绝对贡献峰值落在首次动作提及范围。该结果也未支持首次提及普遍占优，但它只对应这一基线及小样本。', 'small')
p('推理后端：HF 使用 SDPA；SGLang 使用 Triton attention / PyTorch sampling。未修改共享依赖，启动时显式跳过 sgl-kernel 版本门禁；以上实际 log P 对照用于量化后端差异，不宣称逐 bit 一致。', 'small')

page('06  对迁移熵正则的建议与复现边界')
p('建议保留研究方向，调整定位假设', 'h2')
p('第一步应把目标写清楚：增加“合法完整动作”的探索性。ALFWorld 很多动作共享首 token，例如 go to drawer 1 与 go to drawer 2；即使 go 的熵增加，也未必增加目标位置的多样性。')
p('对于 thinking 中存在明确动作表述的样本，可以把首次提及作为候选位置；同时保留末次提及以及区分物体/位置的 token 作为比较组。本次更改的是整段动作文字，尚未证明只对第一个 token 施加熵正则就足够。')
p('对于没有动作字面提及的大多数样本，需要单独处理。优先评估更稳定的动作意图表达或语义跨度定位，再检查是否损害原模型格式与成功率；不要把 unmatched 样本静默丢掉，也不要直接把 will / Okay 的梯度峰值当作动作探索目标。')
p('后续训练应做独立消融，而不是把归因实验当作训练结论', 'h2')
table([['实验组', '验证重点'],
       ['原训练配置', '同 checkpoint、同采样与环境预算，作为成功率与格式基线。'],
       ['首次提及首 token 熵', '检验原迁移方案；报告命中率、有效样本数、动作分布与成功率。'],
       ['区分动作的语义 token 熵', '区分动词、物体、位置；检验完整动作多样性是否增加。'],
       ['末次 / 多位置对照', '检验首次位置是否比其他候选更有效，按命中 token 数归一化。']], [150, 357])
p('实现时若选中输出 token 位置 t，应取预测它的 logits（t-1）上的分布熵，不能误用 token t 的下一个位置。候选位置、熵系数和归一化方式需要同时记录；本次没有改动训练配置或奖励。', 'small')
p('解释范围', 'h2')
p('仅 step 666、valid_seen 日志；主样本按任务类别及轨迹阶段分层，不是总体随机抽样。条件样本刻意富集多次字面提及。每条件仅 16 次采样，bootstrap 未包含重复训练随机性，也未作多重比较校正。原日志没有原始 token IDs：提示词按生产模板恢复并验证解码一致，回答使用规范重分词。')
p('复现资料', 'h2')
p('数据目录：diagnostics/thinking_attribution_20261008（位于本次训练输出目录）。主要文件：summary.json、protocol.json、artifact_manifest.json、generations.jsonl、attribution_*.jsonl、numerical_checks.json、integrated_gradients*.jsonl。运行步骤见同目录 REPRODUCE.txt；脚本位于 examples/alfworld_grpo_baseline/scripts/thinking_attribution/。', 'small')
p('积分梯度方法参考：Sundararajan et al., Axiomatic Attribution for Deep Networks, ICML 2017. <link href="https://proceedings.mlr.press/v70/sundararajan17a.html" color="#2463AA">原论文</link>。本报告对真实模型推理的解释限于上述实验范围。', 'small')

args.output.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(str(args.output), pagesize=(595, 842), leftMargin=44, rightMargin=44,
                        topMargin=42, bottomMargin=58, title='ALFWorld thinking 动作选择归因验证', author='Codex')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(args.output.resolve())
