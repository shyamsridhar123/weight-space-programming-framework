"""One bounded post-hoc derivative; preserves the failed unmodified replay."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import traceback

EXPERIMENT = 'experiments/research/trained-symmetry-pilot'
ARTIFACT = 'artifacts/research/trained-symmetry-pilot'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')

def now():
    return datetime.now(timezone.utc).isoformat()

def worker(public, raw, anchors):
    artifact = public/ARTIFACT
    script = public/EXPERIMENT/'source-freeze/scripts/run_trained_symmetry_pilot.py'
    f = runpy.run_path(str(script), run_name='posthoc_normalized_name_derivative')
    globals_ = f['evaluate'].__globals__
    original_load, original_admission, original_write = globals_['load_model'], globals_['admission'], globals_['write']
    progress = {'schema':'wsp-trained-symmetry-pilot-normalized-name-progress/1',
        'completed_admissions':0, 'normalized_model_loads':0, 'model_checks':[],
        'native_replay_verified':False, 'scope':'Post-hoc derivative progress, not an unmodified frozen replay'}
    progress_path = artifact/'posthoc-normalized-interface-progress.json'
    def load_normalized(path):
        model = original_load(path)
        normalized = f['Network'](**model.tensors,
            input_names=tuple(map(str, model.input_names)), output_names=tuple(map(str, model.output_names)))
        assert normalized.sha256 == model.sha256
        for key,array in model.tensors.items():
            other = normalized.tensors[key]
            assert array.dtype == other.dtype and array.shape == other.shape
            assert array.tobytes(order='C') == other.tobytes(order='C')
        for attr in ('input_names','output_names'):
            original_names, names = getattr(model,attr), getattr(normalized,attr)
            assert len(original_names) == len(names)
            assert all(str(before) == after and type(after) is str for before,after in zip(original_names,names))
            assert tuple(original_names) == names
        progress['normalized_model_loads'] += 1
        progress['model_checks'].append({'file':Path(path).relative_to(artifact).as_posix(),
            'endpoint_sha256':model.sha256, 'tensor_bytes_dtype_shape_identical':True,
            'ordered_textual_names_identical':True,
            'original_name_types':sorted({type(v).__name__ for v in model.input_names+model.output_names}),
            'normalized_name_type':'str'})
        write(progress_path, progress)
        return normalized
    def admission_counted(*args):
        result = original_admission(*args)
        progress['completed_admissions'] += 1
        write(progress_path, progress)
        return result
    def derivative_write(path, value):
        if Path(path) == artifact/'independent-replay.json':
            path = artifact/'posthoc-normalized-interface-replay.json'
            assert not path.exists()
            value = dict(value, schema='wsp-trained-symmetry-pilot-normalized-name-replay/1',
                unmodified_frozen_replay=False, native_replay_verified=False,
                wrapper_source_sha256=sha(Path(__file__)),
                scope='Post-hoc frozen evaluate(replay=True) with interface-name type normalization; tensor bytes, hashes, textual names and ordering asserted identical. Original frozen replay remains failed.')
        return original_write(path, value)
    globals_['load_model'], globals_['admission'], globals_['write'] = load_normalized, admission_counted, derivative_write
    try:
        f['evaluate'](public,raw,anchors,True)
    except BaseException:
        progress['worker_error'] = traceback.format_exc()
        write(progress_path, progress)
        raise

def supervise(public, raw, anchors):
    artifact = public/ARTIFACT
    receipt_path = artifact/'posthoc-normalized-interface-verification.json'
    assert not receipt_path.exists()
    assert not (artifact/'independent-replay.json').exists()
    failure = artifact/'independent-replay-failure.json'
    assert read(failure)['retries'] == 0
    evidence_names = ('evaluation.json','global-selection-seal.json','independent-replay-failure.json')
    before = {name:sha(artifact/name) for name in evidence_names}
    frozen_before = sha(public/EXPERIMENT/'source-freeze/scripts/run_trained_symmetry_pilot.py')
    log = anchors/'posthoc-normalized-interface-verification.log'
    assert not log.exists()
    receipt = {'schema':'wsp-trained-symmetry-pilot-normalized-name-verification/1',
        'status':'running', 'pass':False, 'attempts':1, 'retries':0, 'timeout_seconds':1800,
        'cap_authorization_history':[{'timeout_seconds':300,'invocation_started':False},
            {'timeout_seconds':1800,'invocation_started':False,
             'reason':'Parent raised cap before launch to cover recorded 1299.8s selection cost; same single derivative invocation'}],
        'started_at_utc':now(), 'wrapper_source_sha256':sha(Path(__file__)),
        'frozen_source_sha256':frozen_before, 'original_evidence_sha256':before,
        'native_replay_verified':False, 'unmodified_frozen_replay':False,
        'normalization_delta':'Frozen load_model result -> Network with identical tensor arrays; input_names=tuple(map(str,model.input_names)); output_names likewise.',
        'bookkeeping_delta':'Count completed derivative admissions, retain per-load identity assertions, redirect frozen successful replay receipt to posthoc-normalized-interface-replay.json.',
        'original_loader_source':EXPERIMENT+'/source-freeze/scripts/run_trained_symmetry_pilot.py:81',
        'original_interface_tuple_source':EXPERIMENT+'/source-freeze/scripts/run_trained_symmetry_pilot.py:84',
        'scope':'One separately authorized post-hoc normalized-interface verification. No retraining, reselection, threshold/metric/numeric changes, provider calls, or frozen source writes.'}
    write(receipt_path,receipt)
    with log.open('w',encoding='utf-8') as stream:
        try:
            completed = subprocess.run([sys.executable,str(Path(__file__).resolve()),
                '--public-root',str(public),'--raw-root',str(raw),'--anchors',str(anchors),'--worker'],
                stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
            receipt['exit_code'] = completed.returncode
            receipt['status'] = 'completed' if completed.returncode == 0 else 'failed'
        except subprocess.TimeoutExpired:
            receipt['exit_code'] = 124
            receipt['status'] = 'timeout'
    receipt['finished_at_utc'] = now()
    receipt['external_log'] = {'name':log.name,'bytes':log.stat().st_size,'sha256':sha(log)}
    progress_path = artifact/'posthoc-normalized-interface-progress.json'
    if progress_path.exists():
        progress = read(progress_path)
        receipt['progress_sha256'] = sha(progress_path)
        receipt['completed_derivative_admissions'] = progress['completed_admissions']
        receipt['normalized_model_loads'] = progress['normalized_model_loads']
    result_path = artifact/'posthoc-normalized-interface-replay.json'
    receipt['derivative_success_receipt_present'] = result_path.exists()
    receipt['pass'] = receipt['status'] == 'completed' and result_path.exists() and read(result_path)['pass']
    if result_path.exists():
        receipt['derivative_success_receipt_sha256'] = sha(result_path)
    assert before == {name:sha(artifact/name) for name in evidence_names}
    assert frozen_before == sha(public/EXPERIMENT/'source-freeze/scripts/run_trained_symmetry_pilot.py')
    assert not (artifact/'independent-replay.json').exists()
    receipt['original_evidence_unchanged'] = True
    receipt['verification_gap'] = 'Unmodified frozen native replay remains failed; a timeout or failed derivative does not establish full independent admission/metric replay.'
    write(receipt_path,receipt)
    print(json.dumps(receipt,sort_keys=True),flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-root',type=Path,required=True)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--anchors',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    args = parser.parse_args()
    operation = worker if args.worker else supervise
    operation(args.public_root.resolve(),args.raw_root.resolve(),args.anchors.resolve())
