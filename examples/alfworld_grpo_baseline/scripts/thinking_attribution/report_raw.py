"""Render the audited raw-weight comparison, keeping cohort boundaries explicit."""
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
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
s = json.loads((root/'summary.json').read_text())
assert s['audit_complete']
a = s['attribution']
font = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                     'axes.unicode_minus': False, 'font.size': 10, 'savefig.dpi': 180})
assets = root/'report_figures'
assets.mkdir(exist_ok=True)
blue, orange, dark = '#2866A2', '#CB792A', '#19344D'

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.1))
for ax, key, title in zip(axes, ['controlled_interventions', 'self_interventions'], ['固定旧回答：24 个样本', '原始模型自身回答：11 个样本']):
    for i, label in enumerate(['first_replacement', 'last_replacement']):
        r = s[key]['effects'][label]['p_original']
        mean = r['mean']*100
        lo, hi = np.array(r['ci95'])*100
        ax.errorbar(mean, 1-i, xerr=[[mean-lo], [hi-mean]], fmt='o', color=blue if i == 0 else orange, capsize=4)
    ax.axvline(0, ls='--', color='#9AA7B2')
    ax.set(yticks=[0, 1], yticklabels=['替换末次', '替换首次'], title=title,
           xlabel='原动作选择率变化（百分点）', ylim=(-.6, 1.6))
    ax.grid(axis='x', alpha=.2)
fig.tight_layout()
fig.savefig(assets/'replacement_effects.png', bbox_inches='tight')
plt.close(fig)

pdfmetrics.registerFont(TTFont('CJK', font))
pdfmetrics.registerFontFamily('CJK', normal='CJK', bold='CJK', italic='CJK', boldItalic='CJK')
styles = {name: ParagraphStyle(name, fontName='CJK', fontSize=size, leading=lead, textColor=colors.HexColor(dark),
                               wordWrap='CJK', spaceAfter=8)
          for name, size, lead in [('title', 24, 34), ('h1', 17, 25), ('h2', 12, 18), ('body', 10, 16),
                                   ('small', 8, 12), ('cell', 8.8, 13)]}
story = []


def p(text, style='body'):
    story.append(Paragraph(text, styles[style]))


def table(rows, widths):
    t = Table([[Paragraph(html.escape(str(c)).replace('\n', '<br/>'), styles['cell']) for c in row] for row in rows],
              colWidths=widths, hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E6EEF7')),
                           ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F8FB')]),
                           ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                           ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                           ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7)]))
    story.extend([t, Spacer(1, 10)])


def page(title):
    story.append(PageBreak())
    p(title, 'h1')


def fx(r):
    return f"{r['mean']*100:+.2f} [{r['ci95'][0]*100:+.2f}, {r['ci95'][1]*100:+.2f}]"


def mention(r):
    return f"{r.get('peak_first_mention', 0)}/{r['with_literal']}"


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#CCD9E5'))
    canvas.line(44, 43, 551, 43)
    canvas.setFont('CJK', 8)
    canvas.setFillColor(colors.HexColor('#617487'))
    canvas.drawString(44, 28, 'ALFWorld · 原始 Qwen3.5-2B · thinking 归因对照 · 2026-10-08')
    canvas.drawRightString(551, 28, str(doc.page))


p('原始 Qwen3.5-2B<br/>动作选择梯度归因实验', 'title')
p('对照原实验的 SFT + GRPO step 666 权重；本次使用未经过本项目 ALFWorld SFT/GRPO 的原始发布权重。冻结模型，仅分析，不更新训练配置。')
table([['实验部分', '样本 / 续写数量'], ['固定相同提示词和回答，只更换权重', '357 个主样本 + 60 个多次提及样本'],
       ['原始模型重新回答相同上下文', '768-token：720 次；1,536-token：360 次'],
       ['对原始模型自身回答归因', f"768-token：{a['raw_self_768']['n']} 个；1,536-token：{a['raw_self_1536']['n']} 个"],
       ['配对动作提及替换与前缀续写', '固定文本 2,304 次；自身文本 1,056 次']], [303, 204])
p('结论', 'h2')
p('本次没有支持“thinking 中首次出现最终动作的位置，普遍是最关键的梯度位置”。即使更换为原始发布权重，固定旧回答上的峰值仍常落在 will、Okay 等通用词；因此不能将这一现象简单归因于本项目 SFT。')
p(f"原始模型自身回答中，完整动作 logP 的梯度峰值落入首次动作提及的比例为 {mention(a['raw_self_768']['objectives']['full'])}（768-token）和 {mention(a['raw_self_1536']['objectives']['full'])}（1,536-token）。分母仅含有效且明确提及最终动作的回答。")
p('原始模型在当前 RL 提示词下大量不闭合 thinking。这既限制了自身回答的归因覆盖面，也使“输出非法”和“换了动作”必须分开解释。实验没有执行反事实环境轨迹，不能据此断言哪种熵正则能提升任务成功率。')

page('1 / 控制回答文本，隔离权重差异')
p('沿用上次冻结的同一批提示词、环境观察、历史与回答，只将计算梯度的权重替换为原始模型。目标为实际 XML 调用中完整动作 token 的 log 概率之和；对 thinking 各 token 输入嵌入求梯度 L2 范数。')
rows = [['文本样本', 'SFT + GRPO666', '原始发布权重']]
for suffix, label in [('main', '主样本 357 个'), ('enriched', '多次提及样本 60 个')]:
    x, y = a['grpo_'+suffix]['objectives']['full'], a['raw_controlled_'+suffix]['objectives']['full']
    rows.append([label+'\n峰值落入首次提及', mention(x), mention(y)])
    rows.append(['主要峰值 token', ', '.join(f'{t.strip() or "空白"}:{n}' for t, n in x['top_peak_tokens'][:3]),
                 ', '.join(f'{t.strip() or "空白"}:{n}' for t, n in y['top_peak_tokens'][:3])])
table(rows, [171, 168, 168])
p('如何解读', 'h2')
p('原始权重与训练后权重都存在通用词峰值。一个位置对最终动作 logP 敏感，并不等于这个词的词面语义决定了动作，也不等于对它施加 token 熵就能增加动作探索。开头位置的表示与后续生成存在多条依赖路径。')
p('本次同时保存 schema 首 token、对象/位置片段、原动作与候选动作 logP 差，以及梯度乘输入的结果。它们是不同归因定义；不能只挑符合“首次提及”假设的一种指标作为结论。', 'small')
p('这组数据仍是训练后模型产生的回答，不是原始模型的自然回答分布。两种权重的相似现象只能排除“必然完全由 SFT 造成”这一过强解释，不能证明 SFT 完全没有影响。')

page('2 / 原始模型自己生成回答')
p('使用相同 360 个环境上下文，temperature=1、top_p=1、top_k=-1，沿用实际 RL chat template。每个上下文首个随机样本若合法且闭合 thinking，才进入归因；不从多次采样中挑选最方便分析的回答。')
x, y = s['fresh_768'], s['fresh_1536']
table([['指标', '768 tokens', '1,536 tokens'], ['生成次数', x['generations'], y['generations']],
       ['合法 XML + 当前可执行动作', f"{x['legal']}/{x['generations']}（{x['legal']/x['generations']:.1%}）", f"{y['legal']}/{y['generations']}（{y['legal']/y['generations']:.1%}）"],
       ['thinking 未闭合', x['reason/unclosed_thinking'], y['reason/unclosed_thinking']],
       ['触及长度上限', x['length_exhausted'], y['length_exhausted']],
       ['归因首样本数', a['raw_self_768']['n'], a['raw_self_1536']['n']],
       ['峰值位于首次完整动作提及', mention(a['raw_self_768']['objectives']['full']), mention(a['raw_self_1536']['objectives']['full'])]], [219, 144, 144])
p('典型失败形式是长篇分析角色、输出格式和示例，持续生成 thinking，直到长度耗尽；因此不是简单地“动作多样性不足”。扩大预算只做独立敏感性对照，没有替换或隐去原来的失败样本。')
p('注意：768-token 的合法率使用每上下文两次采样，而 1,536-token 使用一次；两者不能当作严格配对的成功率比较。环境历史来自 GRPO 访问过的状态，也不是原始模型自行执行的完整轨迹。', 'small')
p('自身回答归因仍受明显筛选影响：它仅覆盖能在指定预算内输出合法动作的一小部分回答。完整命令可能出现在候选列举、格式示例或明确决定中，本次“提及”按字面匹配定义，不自动等同于决策承诺。')

page('3 / 替换动作提及，观察实际续写')
p('将首次或末次提及改成另一个 admissible action，立即从替换位置续写；对照为同一位置的原动作前缀。每条件 16 次，保留非法输出。原动作选择率以全部 16 次为分母；区间按游戏任务聚类自助抽样 10,000 次。')
path = assets/'replacement_effects.png'
w, h = ImageReader(str(path)).getSize()
story.append(Image(str(path), width=507, height=507*h/w))
table([['原动作选择率变化（百分点）', '固定旧文本', '原始模型自身文本'],
       ['首次替换', fx(s['controlled_interventions']['effects']['first_replacement']['p_original']), fx(s['self_interventions']['effects']['first_replacement']['p_original'])],
       ['末次替换', fx(s['controlled_interventions']['effects']['last_replacement']['p_original']), fx(s['self_interventions']['effects']['last_replacement']['p_original'])],
       ['首次效应 - 末次效应', fx(s['controlled_interventions']['first_minus_last']), fx(s['self_interventions']['first_minus_last'])]], [195, 156, 156])
for key, label in [('controlled_interventions', '固定旧文本'), ('self_interventions', '自身文本')]:
    r = s[key]
    p(f"{label}：首次替换同时令合法率变化 {fx(r['effects']['first_replacement']['valid_rate'])}；末次替换为 {fx(r['effects']['last_replacement']['valid_rate'])} 个百分点。合法率下降不能直接解释成探索增强。", 'small')
p('括号中为 95% 区间。自身文本干预仅有 11 个具备多次完整动作提及的合法回答，估计不稳定；本实验没有证明首次位置普遍优于末次位置。')

page('4 / 动作对比目标出现了不同信号')
p('前述归因目标只问：“哪些位置影响当前动作的 log 概率？”这可能混合格式、语言组织和具体选择的敏感性。另一个目标是 logP(当前动作) - logP(另一合法动作)，直接比较两个选择。')
p('为尽量比较具体对象或位置，优先选取相同动词、最长公共文本前缀的另一个 admissible action；在两条候选动作的 teacher-forcing 计算中固定同一段 thinking。下表严格在同一批存在候选且明确提及最终动作的样本上比较。')
rows = [['峰值落入首次完整动作提及', '单动作 logP', '动作 logP 差']]
for key, label in [('raw_self_768', '自身回答 / 768 tokens'), ('raw_self_1536', '自身回答 / 1,536 tokens')]:
    r = a[key]['matched_same_verb_literal']
    rows.append([label, f"{r['full_peak_first_mention']}/{r['n']}", f"{r['contrast_peak_first_mention']}/{r['n']}"])
table(rows, [251, 128, 128])
p('这个结果值得保留：它提示归因目标的定义，会影响能否找到与具体选择相关的位置。因此，换原始模型并没有简单恢复“首次提及必然最重要”的结论，但动作对比归因提供了一个更值得进一步验证的方向。')
p('仍然不能直接据此施加首 token 熵：表中命中的是完整命令的提及区间，不保证命中该区间第一个 token；仅比较一个替代动作，也没有构造全部合法动作的归一化分布。')
p('样本量分别只有 7 和 14，且来自有效输出这一条件子集。这是探索性结果，没有独立测试集或针对该对比目标的专门因果验证。训练收益仍需通过单独的熵正则消融验证。')

page('5 / 数值复核、结论边界与复现')
rows = [['数值检查', '固定文本', '自身文本']]
for label, key in [('FP32/BF16 最大梯度位置一致', 'same_peak'), ('梯度范数向量最低余弦相似度', 'min_bf16_fp32_cosine')]:
    vals = [s[k] for k in ['numerical_controlled', 'numerical_self']]
    rows.append([label]+[f"{v[key]}/{v['n']}" if key == 'same_peak' else f"{v[key]:.6f}" for v in vals])
rows.append(['有限差分最大相对误差（ε=0.005）']+[f"{s[k]['finite_difference_error_by_epsilon']['0.005']['max']:.2%}" for k in ['numerical_controlled', 'numerical_self']])
table(rows, [251, 128, 128])
p(f"HF 梯度后端与 SGLang 续写后端复核了 {s['backend']['n']} 个案例，目标 token ID 全部对齐；单 token logP 最大差异为 {s['backend']['max_token_logp_difference']:.3f}。BF16 与 FP32 存在个别峰值换位，因此不将相邻、近似等高 token 的顺序解释为确定事实。", 'small')
p('自身文本的一个样本存在明显步长敏感性：ε=0.005 时有限差分误差约 76.9%；缩小到 0.0005 时为 4.3%，但不同小步长的误差也不单调。未将其标记为稳定通过；原始检查和追加检查均保留。该样本的单动作梯度峰值不在首次动作提及中。', 'small')
p('实际调用目标严格定位于闭合 thinking 之后，避免把 thinking 中引用的 XML 示例当作最终动作。旧 417 个固定文本经审计没有该歧义，因此这项适配不会改变旧文本的比较对象。', 'small')
p('对于熵正则设计的含义', 'h2')
p('本实验不支持仅凭“最终动作首次字面出现”构建通用首 token 掩码。若继续沿用损失层面的路线，应先区分候选列举与真正的选择承诺，并用条件动作分布或动作对比目标验证；最终仍需训练消融，以真实合法动作多样性和任务成功率评估效果。')
p('复现入口', 'h2')
p('全部原始输出、归因数组、配对设计、权重文件哈希、协议修订、数值复核及 summary.json 已保存在本次实验目录。REPRODUCE.txt 提供命令和方法说明；scripts/thinking_attribution/ 中保留可重复执行的分析脚本。', 'small')
p(html.escape(str(root.parent))+'/<br/>'+html.escape(root.name), 'small')
p('本次没有隔离“只做 SFT”的中间检查点；不能从原始权重与 GRPO666 的两端对照中单独估计 SFT 和 RL 各自的贡献。原始发布模型也不等同于未经厂商后训练的纯预训练模型。', 'small')
args.output.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(str(args.output), pagesize=(595, 842), leftMargin=44, rightMargin=44,
                  topMargin=45, bottomMargin=58, title='原始 Qwen3.5-2B thinking 动作归因对照', author='ALFWorld experiment').build(story, onFirstPage=footer, onLaterPages=footer)
print(args.output)
