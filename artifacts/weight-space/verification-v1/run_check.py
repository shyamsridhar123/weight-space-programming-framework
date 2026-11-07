from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
STAGE = sys.argv[1]
PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'
NODE = shutil.which('node')
if not NODE:
    raise SystemExit('Local node executable unavailable')
STAGES = {
    'python-tests': [str(PYTHON), '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--basetemp', str(OUT / 'pytest-tmp'), 'tests'],
    'typescript-tests': [NODE, str(ROOT / 'node_modules/tsx/dist/cli.mjs'), '--test', '--test-concurrency=2', *[str(p) for p in sorted((ROOT / 'tests').glob('*.test.ts'))]],
    'typescript-check': [NODE, str(ROOT / 'node_modules/typescript/bin/tsc'), '--noEmit'],
    'vite-build': [NODE, str(ROOT / 'node_modules/vite/bin/vite.js'), 'build', 'game', '--outDir', str(OUT / 'web-build')],
}
args = STAGES[STAGE]
if STAGE == 'vite-build' and (OUT / 'web-build').exists():
    raise SystemExit('Build target already exists; refusing overwrite')
if STAGE == 'python-tests' and (OUT / 'pytest-tmp').exists():
    raise SystemExit('Pytest target already exists; refusing overwrite')
env_delta = {
    'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
    'TSX_DISABLE_CACHE': '1',
}
env = os.environ.copy()
env.update(env_delta)
def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
started = datetime.now(timezone.utc).isoformat()
write_json(OUT / f'{STAGE}.command.json', {'stage': STAGE, 'arguments': args, 'cwd': str(ROOT), 'environment_overrides': env_delta, 'started_at': started, 'runner_sha256': sha(Path(__file__))})
status = {'stage': STAGE, 'started_at': started, 'exit_code': None, 'launch_or_capture_error': None}
t0 = time.perf_counter()
stdout_path, stderr_path = OUT / f'{STAGE}.stdout.log', OUT / f'{STAGE}.stderr.log'
try:
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        result = subprocess.run(args, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=600, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    status['exit_code'] = result.returncode
except Exception as error:
    status['launch_or_capture_error'] = f'{type(error).__name__}: {error}'
finally:
    status.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-t0,
                  stdout_sha256=sha(stdout_path) if stdout_path.exists() else None,
                  stderr_sha256=sha(stderr_path) if stderr_path.exists() else None)
    write_json(OUT / f'{STAGE}.status.json', status)
print(json.dumps(status))
sys.exit(status['exit_code'] if status['exit_code'] is not None else 1)
