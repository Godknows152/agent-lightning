"""Read-only analysis of the current trajectory-GRPO run; no training changes."""
from pathlib import Path
import collections, csv, datetime, json, math, re, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path('/home/LXJ/Python_Projects/Agent_Lightning')
BASE = REPO / 'examples/alfworld_grpo_baseline'
OUT = Path(__file__).resolve().parent
RUN = BASE / 'outputs/qwen3.5_2B/full_sft_grpo_tuned'
LOG = BASE / 'log/alfworld/qwen35_2b/full_sft/v1/2gpu/seed0/alfworld_v1_seed0_20260927_010659.log'
rows = []
for line in LOG.read_text(errors='replace').splitlines():
    line = re.sub(r'\x1b\[[0-9;]*m', '', line)
    match = re.search(r'\bstep:\s*(\d+)\s+-\s+', line)
    if not match:
        continue
    row = {'step': int(match.group(1))}
    for part in line[match.end():].split(' - '):
        key, sep, value = part.partition(':')
        if sep:
            try: row[key.strip()] = float(value.strip())
            except ValueError: pass
    n = row['alfworld_termination/total_trajectories']
    invalid = row['alfworld_penalty/invalid_action_count']
    missing = row['alfworld_penalty/no_action_count']
    decisions = n * row['alfworld/valid_action_count/mean'] + invalid + missing
    row['analysis/success_pct'] = row['alfworld_termination/success_count'] / n * 100
    row['analysis/invalid_pct'] = invalid / decisions * 100
    row['analysis/repeat_pct'] = row['alfworld_penalty/repeated_action_count'] / decisions * 100
    row['analysis/episode_reward'] = (10 * row['alfworld_termination/success_count'] - .2 * (invalid + missing)) / n
    rows.append(row)
latest = rows[-1]['step']

def action(row):
    m = re.search(r'<parameter=action>(.*?)</parameter>', row['output'], re.S)
    return m.group(1).strip() if m else '<unparseable>'

def read_episodes(path):
    episodes = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if not r['output'].strip():
            continue  # Native rectangular export includes artificial padding rows.
        episodes[r['uid'].rsplit('_', 1)[0]].append(r)
    for rs in episodes.values(): rs.sort(key=lambda r: int(r['uid'].rsplit('_', 1)[1]))
    return episodes

validation = []
example = None
validation_games = []
for path in sorted((RUN/'validation').glob('*.jsonl'), key=lambda p:int(p.stem)):
    if int(path.stem) > latest: continue
    episodes = read_episodes(path)
    aa = [action(r) for rs in episodes.values() for r in rs]
    validation.append({'step':int(path.stem),'episodes':len(episodes),'decision_rows':len(aa),
        'successes':sum(rs[-1].get('reward',rs[-1]['score']) > 0 for rs in episodes.values()),
        'look_pct':100*aa.count('look')/len(aa),
        'only_look_episodes':sum(all(action(r)=='look' for r in rs) for rs in episodes.values())})
    validation_games.append({rs[0]['gts']['game_file'] for rs in episodes.values()})
    for uid, rs in episodes.items():
        if all(action(r)=='look' for r in rs):
            example={'validation_step':int(path.stem),'trajectory_id':uid,'decisions':len(rs),
                'task':re.findall(r'Your task is to:[^\n]*',rs[0]['input']),
                'game_file':rs[0]['gts']['game_file'],'last_output':rs[-1]['output']}
            break

window_keys = ['alfworld_termination/success_count','analysis/success_pct','critic/score/mean',
    'analysis/episode_reward','alfworld_penalty/invalid_action_count','analysis/invalid_pct',
    'alfworld_penalty/no_action_count','alfworld_penalty/repeated_action_count','analysis/repeat_pct',
    'alfworld_termination/environment_timeout_count','alfworld_termination/decision_limit_count',
    'actor/entropy_loss','actor/kl_loss','actor/grad_norm','actor/pg_clipfrac','timing_s/step']
windows = {}
for name, subset in [('first10',rows[:10]),('last10',rows[-10:])]:
    windows[name] = {'steps':[subset[0]['step'],subset[-1]['step']],
        **{k:statistics.mean(r[k] for r in subset) for k in window_keys}}
    groups_total = groups_no_success = groups_zero_variance = 0
    dump_rewards = []
    for r in subset:
        episodes = read_episodes(RUN/'rollouts'/f"{r['step']}.jsonl")
        groups = collections.defaultdict(list)
        for uid, rs in episodes.items():
            scores = {round(x['score'],5) for x in rs}
            assert len(scores)==1
            score = next(iter(scores))
            groups[uid.rsplit('_',1)[0]].append(score)
            dump_rewards.append(score)
        assert len(episodes)==128 and len(groups)==16
        assert all(len(v)==8 for v in groups.values())
        groups_total += len(groups)
        groups_no_success += sum(max(v)<=0 for v in groups.values())
        groups_zero_variance += sum(len(set(v))==1 for v in groups.values())
    windows[name]['task_groups'] = groups_total
    windows[name]['groups_without_success'] = groups_no_success
    windows[name]['groups_with_equal_rewards'] = groups_zero_variance
    windows[name]['dump_episode_reward_mean'] = statistics.mean(dump_rewards)
    assert abs(windows[name]['analysis/episode_reward']-statistics.mean(dump_rewards))<1e-5

summary = {'generated_at':datetime.datetime.now().astimezone().isoformat(),'run_id':'5z7jo51x',
    'log':str(LOG),'last_step':latest,'windows':windows,'validation':validation,
    'same_validation_games':all(x==validation_games[0] for x in validation_games),
    'loop_example':example,'latest_metrics':rows[-1],
    'nonfinite_metrics':[(r['step'],k) for r in rows for k,v in r.items() if not math.isfinite(v)],
    'limits':['No pre-training validation: cannot establish change versus the initial SFT model.',
              'Training batches contain different tasks; their success-rate trend is not a paired evaluation.',
              'Entropy is token entropy and includes reasoning/XML, not task-conditioned action diversity.']}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
with (OUT/'training_metrics.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))),lineterminator='\n')
    writer.writeheader();writer.writerows(rows)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
x=[r['step'] for r in rows]
def curve(ax,key,label,color):
    y=[r[key] for r in rows]
    ma=[statistics.mean(y[max(0,i-9):i+1]) for i in range(len(y))]
    ax.plot(x,y,color=color,alpha=.2,lw=1)
    ax.plot(x,ma,color=color,lw=2,label=label)
    ax.grid(alpha=.2);ax.set_xlabel('Training step')
curve(axes[0,0],'analysis/success_pct','Train success (10-step mean)','#1b6ca8')
axes[0,0].scatter([r['step'] for r in validation],[0]*len(validation),color='#c12c38',label='Greedy validation: 0/140',zorder=3)
axes[0,0].set(title='Task success',ylabel='% of episodes');axes[0,0].legend(fontsize=8)
curve(axes[0,1],'analysis/episode_reward','Episode mean reward','#15806c')
curve(axes[0,1],'critic/score/mean','Logged decision-row mean','#d47b16')
axes[0,1].set(title='Reward rises as penalties fall',ylabel='Reward');axes[0,1].legend(fontsize=8)
curve(axes[1,0],'analysis/invalid_pct','Invalid actions','#c12c38')
axes[1,0].set(title='Action validity improves',ylabel='% of decisions')
curve(axes[1,1],'analysis/repeat_pct','Repeated valid actions','#9c5fa5')
axes[1,1].set(title='Repetition stays high',ylabel='% of decisions')
curve(axes[2,0],'actor/entropy_loss','Token entropy','#1b6ca8')
curve(axes[2,0],'actor/kl_loss','KL to frozen SFT reference','#d47b16')
axes[2,0].set(title='Entropy and reference drift increase',ylabel='Logged value');axes[2,0].legend(fontsize=8)
axes[2,1].plot([r['step'] for r in validation],[r['look_pct'] for r in validation],'o-',color='#c12c38')
axes[2,1].set(title='Greedy validation is dominated by look',xlabel='Training step',ylabel='% of actions',ylim=(0,105));axes[2,1].grid(alpha=.2)
fig.suptitle(f'ALFWorld trajectory GRPO | 5z7jo51x | steps 1-{latest}\nThin lines: raw batches; thick lines: trailing 10-step means',fontsize=14)
fig.savefig(OUT/'training_effectiveness.png',dpi=160)
print(json.dumps({k:v for k,v in summary.items() if k not in ['latest_metrics','loop_example']},ensure_ascii=False,indent=2))
print('CHART',OUT/'training_effectiveness.png')
