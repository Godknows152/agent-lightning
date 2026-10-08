"""Run independent frozen-model analysis and generation jobs, cleaning only our server."""
import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
args = parser.parse_args()
root = args.root.resolve()
scripts = Path(__file__).resolve().parent
model = root/'model_merged'
logs = root/'execution_logs'
env = dict(os.environ, OMP_NUM_THREADS='4', SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK='1')
server_url = 'http://127.0.0.1:30408'


def run(name, script, arguments, gpu='0'):
    command = [sys.executable, '-u', str(scripts/script), *map(str, arguments)]
    with (logs/'commands.jsonl').open('a') as handle:
        handle.write(json.dumps({'name': name, 'cuda': gpu, 'command': command, 'time': time.time()})+'\n')
    print('START', name, flush=True)
    with (logs/(name+'.log')).open('a') as output:
        subprocess.run(command, env=dict(env, CUDA_VISIBLE_DEVICES=gpu), stdout=output,
                       stderr=subprocess.STDOUT, check=True)
    print('DONE', name, flush=True)


def controlled():
    common = ['--root', root, '--model', model]
    run('controlled_main', 'attribute.py', common+['--output-name', 'controlled_main.jsonl'])
    run('controlled_enriched', 'attribute.py', common+['--cohort', 'enriched', '--output-name', 'controlled_enriched.jsonl'])
    run('fixed', 'fixed_and_ig.py', common+['--mode', 'fixed'])
    run('numerical_controlled', 'raw_checks.py', [root, '--model', model, '--mode', 'numerical'])


try:
    requests.get(server_url+'/health', timeout=2)
except requests.ConnectionError:
    pass
else:
    raise RuntimeError('Port 30408 already responds; refusing to reuse another service')

server_log = (logs/'server.log').open('a')
command = [sys.executable, '-m', 'sglang.launch_server', '--model-path', str(model),
           '--host', '127.0.0.1', '--port', '30408', '--tp-size', '1', '--mem-fraction-static', '0.45',
           '--max-running-requests', '32', '--context-length', '4096', '--trust-remote-code',
           '--dtype', 'bfloat16', '--attention-backend', 'triton', '--sampling-backend', 'pytorch']
server = subprocess.Popen(command, env=dict(env, CUDA_VISIBLE_DEVICES='3'), stdout=server_log,
                          stderr=subprocess.STDOUT, start_new_session=True)
(logs/'server_process.json').write_text(json.dumps({'pid': server.pid, 'command': command, 'cuda': '3'}, indent=2))
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(controlled)
        deadline = time.monotonic()+600
        while True:
            if server.poll() is not None:
                raise RuntimeError('Serving backend exited; see server.log')
            try:
                if requests.get(server_url+'/health', timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Serving backend did not become ready')
            time.sleep(3)
        common = ['--root', root, '--model', model, '--server', server_url]
        run('fresh_generation', 'generate.py', common+['--design', root/'fresh_design.json',
            '--output-name', 'fresh_generations.jsonl', '--repeats', 2, '--concurrency', 8], '')
        run('prepare_self', 'prepare_sft_self.py', [root], '')
        run('self_generation', 'generate.py', common+['--samples', root/'samples_self.jsonl',
            '--design', root/'self_generation_design.json', '--output-name', 'self_intervention_generations.jsonl',
            '--repeats', 16, '--concurrency', 2], '')
        run('controlled_generation', 'generate.py', common+['--design', root/'controlled_generation_design.json',
            '--output-name', 'controlled_generations.jsonl', '--repeats', 16, '--concurrency', 2], '')
        future.result()
    run('backend', 'raw_checks.py', [root, '--model', model, '--mode', 'backend'], '')
finally:
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
    server_log.close()
run('attribution_self', 'attribute.py', ['--root', root, '--model', model, '--samples',
    root/'samples_self.jsonl', '--output-name', 'attribution_self.jsonl'])
run('numerical_self', 'raw_checks.py', [root, '--model', model, '--mode', 'numerical',
    '--samples', root/'samples_self.jsonl', '--attribution', root/'attribution_self.jsonl',
    '--output-name', 'numerical_checks_self.json'])
(root/'suite_complete.json').write_text(json.dumps({'completed_at_unix': time.time(), 'server_stopped': True})+'\n')
print('SUITE_COMPLETE', flush=True)
