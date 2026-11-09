"""Separately frozen #16 estimation pilot; no provider and no adaptive retries.

Run freeze, select, evaluate, and replay in separate processes. The executable
snapshot imports its own helper bytes. Only evaluate/replay parse official tests.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import zipfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rlcd.weight_space.artifact import StructuralPolicy
from rlcd.weight_space.contracts import DecisionRule, admit, bind_admission
from rlcd.weight_space.model import InputBox, Network, digest
from rlcd.weight_space.program import Program, apply_program, difference, rollback
from rlcd.weight_space.research_tasks import (TextPreprocessing, compile_arm, encode_inputs,
    evaluate_head, grouped_roles, make_head, normalized_text, public_record, train_encoder)
from rlcd.weight_space.semantics import RELU_ARCHITECTURE, to_tensor_artifact
from rlcd.weight_space.structured import ExecutionSpec
from rlcd.weight_space.symmetry import Gauge, align_hidden, transfer_edit, transform, transport
from rlcd.weight_space.workflow import TaskSpec

EXPERIMENT = 'experiments/research/trained-symmetry-pilot'
ARTIFACT = 'artifacts/research/trained-symmetry-pilot'
GATE_HASH = '601897dc212592ae76cb2eed4b56e120de6ed9d66fa292b7e42ba5b924a13d0f'
RAW = {
 'banking77/banking_data/train.csv': 'b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b',
 'banking77/banking_data/test.csv': 'd12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d',
 'banking77/banking_data/categories.json': '53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63',
 'banking77/LICENSE': '7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661',
 'optical-digits/optical-digits.zip': '0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0',
 'optical-digits/source-page.html': '891f996da7c22b59fd71e2e14fbaa94ec2d955b724f3acf79a7df993ae6d7b33',
 'optical-digits/metadata.json': '3cd30d08daa235f503dee650d25ff4a8ae0088ffaaa01fa4712364ab872dc656',
}
ARMS = ('none', 'raw', 'heuristic', 'wrong', 'matched_random', 'fixed_target_fit')
GAUGES = ('identity', 'permutation', 'dyadic', 'combined')
LEDGER = {'native_full_calls': 0, 'native_full_rows': 0, 'native_hidden_calls': 0,
          'native_hidden_rows': 0, 'binary64_calls': 0, 'binary64_rows': 0}

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    with (gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz') else
          Path(path).open(encoding='utf-8')) as stream:
        return json.load(stream)

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(public_record(value), indent=2, sort_keys=True, allow_nan=False)+'\n'
    if str(path).endswith('.gz'):
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            stream.write(payload)
    else:
        path.write_text(payload, encoding='utf-8')

def save_model(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **model.tensors, input_names=np.array(model.input_names),
                        output_names=np.array(model.output_names))

def load_model(path):
    with np.load(path, allow_pickle=False) as arrays:
        return Network(**{k: arrays[k] for k in ('w1','b1','w2','b2')},
            input_names=tuple(arrays['input_names']), output_names=tuple(arrays['output_names']))

def native(model, inputs):
    import torch
    LEDGER['native_full_calls'] += 1
    LEDGER['native_full_rows'] += len(inputs)
    with torch.no_grad():
        h = torch.relu(torch.nn.functional.linear(torch.tensor(inputs, dtype=torch.float32),
            torch.tensor(model.w1, dtype=torch.float32), torch.tensor(model.b1, dtype=torch.float32)))
        return torch.nn.functional.linear(h, torch.tensor(model.w2, dtype=torch.float32),
                                          torch.tensor(model.b2, dtype=torch.float32)).numpy()

def binary(model, inputs):
    LEDGER['binary64_calls'] += 1
    LEDGER['binary64_rows'] += len(inputs)
    return model(inputs)

def hidden(model, inputs):
    LEDGER['native_hidden_calls'] += 1
    LEDGER['native_hidden_rows'] += len(inputs)
    return encode_inputs({'weight': model.w1.astype(np.float32), 'bias': model.b1.astype(np.float32)}, inputs)

def agreement(a, b):
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    maximum = float(np.max(np.abs(a.astype(float)-b.astype(float))))
    decision_mismatches = int(np.count_nonzero(a.argmax(1) != b.argmax(1)))
    ordered = np.sort(a, axis=1)
    return {'pass': finite and bool(np.allclose(a,b,atol=1e-5,rtol=1e-5)) and decision_mismatches == 0,
            'maximum_absolute_score_discrepancy': maximum, 'decision_mismatches': decision_mismatches,
            'nonfinite_values': int((~np.isfinite(a)).sum()+(~np.isfinite(b)).sum()),
            'near_ties_at_1e-5': int(np.count_nonzero(ordered[:,-1]-ordered[:,-2] <= 1e-5))}

def gauge(kind, seed):
    rng = np.random.default_rng(seed)
    p = np.arange(64) if kind in ('identity','dyadic') else rng.permutation(64)
    s = np.ones(64) if kind in ('identity','permutation') else rng.choice([.5,1.,2.],64)
    return Gauge(tuple(map(int,p)), tuple(map(float,s)))

def realize(parent, program):
    candidate = apply_program(parent, program)
    candidate = candidate.replace(**{k: v.astype(np.float32) for k,v in candidate.tensors.items()})
    return candidate, difference(parent, candidate, program.intent+'; recipient float32 realization')

def random_edit(parent, reference, seed):
    rng, values, diagnostics = np.random.default_rng(seed), {}, {}
    for key, old in parent.tensors.items():
        delta = reference.tensors[key]-old
        count, norm = int(np.count_nonzero(delta)), float(np.linalg.norm(delta))
        candidate = old.copy().reshape(-1)
        if count:
            locations = rng.choice(len(candidate), count, replace=False)
            draws = rng.normal(size=count)
            draws *= norm/np.linalg.norm(draws)
            candidate[locations] += draws
        values[key] = candidate.reshape(old.shape).astype(np.float32)
        actual = values[key].astype(float)-old
        actual_count, actual_norm = int(np.count_nonzero(actual)), float(np.linalg.norm(actual))
        diagnostics[key] = {'reference_count': count, 'realized_count': actual_count,
            'reference_l2': norm, 'realized_l2': actual_norm, 'count_matched': count == actual_count,
            'relative_l2_mismatch': abs(actual_norm-norm)/max(norm,np.finfo(float).tiny)}
    candidate = parent.replace(**values)
    return candidate, difference(parent,candidate,'Assignment-seeded support/L2 matched random control'), diagnostics

def bridge(trained, preprocessing, labels, seed, domain):
    head = make_head(trained, digest(preprocessing), labels, seed, domain)
    model = Network(trained['encoder']['weight'], trained['encoder']['bias'],
        head.tensors['head.weight'], head.tensors['head.bias'],
        tuple(f'{domain}/preprocessing/{digest(preprocessing)}/input/{i}' for i in range(trained['encoder']['weight'].shape[1])),
        tuple(labels+['specialist_routing_code']))
    return model, head

def fit(model, head, target_x, retain_x):
    target, retain = hidden(model,target_x), hidden(model,retain_x)
    result = compile_arm(head,target,retain,model.shape[2]-1,'soft',ridge=.01,retention_weight=8.)
    candidate = model.replace(w2=result['artifact'].tensors['head.weight'], b2=result['artifact'].tensors['head.bias'])
    return candidate, difference(model,candidate,'Fixed donor/recipient soft target fit'), {
        'fit_seconds': result['fit_seconds'], 'target_rows': len(target), 'retain_rows': len(retain),
        'diagnostics': result['diagnostics'], 'head_forward_calls': 1, 'head_forward_rows': len(target)}

def raw_data(raw_root, domain, heldout=False, preprocessing=None):
    if domain == 'banking77':
        labels = read(raw_root/'banking77/banking_data/categories.json')
        with (raw_root/f'banking77/banking_data/{"test" if heldout else "train"}.csv').open(encoding='utf-8',newline='') as stream:
            rows = list(csv.DictReader(stream))
        texts = [r['text'] for r in rows]
        y = np.array([labels.index(r['category']) for r in rows],dtype=np.int64)
        groups = [hashlib.sha256(normalized_text(t).encode()).hexdigest() for t in texts]
        roles = None if heldout else grouped_roles(y,groups,'trained-symmetry-pilot-role/1')
        if preprocessing is None:
            prep = TextPreprocessing.fit([texts[i] for i in roles['train']])
        else:
            prep = TextPreprocessing(tuple(preprocessing['vocabulary']),tuple(preprocessing['idf']))
        x, preprocessing = prep.transform(texts), prep.to_dict()
    else:
        with zipfile.ZipFile(raw_root/'optical-digits/optical-digits.zip') as archive:
            rows = np.loadtxt(io.BytesIO(archive.read('optdigits.tes' if heldout else 'optdigits.tra')),delimiter=',',dtype=np.int64)
        x, y = (rows[:,:-1]/16.).astype(np.float32), rows[:,-1].astype(np.int64)
        labels = [str(i) for i in range(10)]
        groups = [hashlib.sha256(r.tobytes()).hexdigest() for r in rows[:,:-1]]
        roles = None if heldout else grouped_roles(y,groups,'trained-symmetry-pilot-role/1')
        preprocessing = {'schema':'wsp-optical-digits-preprocessing/1','order':'row-major 8x8','scale':'divide by16','fit_scope':'none'}
    return x,y,labels,groups,roles,preprocessing

def freeze(public, raw_root, anchors):
    experiment = public/EXPERIMENT
    if (experiment/'freeze.json').exists():
        raise ValueError('Freeze already exists; no mutation or refreeze')
    gate = read(public/'artifacts/research/live-studies/global-selection-seal.json')
    release = read(public/'artifacts/research/live-studies/evidence-release.json')
    if digest({k:v for k,v in gate.items() if k != 'sha256'}) != GATE_HASH or not release['both_validations_passed']:
        raise ValueError('Authenticated release mismatch')
    raw_hashes = {}
    for name, expected in RAW.items():
        if sha(raw_root/name) != expected:
            raise ValueError('Public acquisition mismatch: '+name)
        raw_hashes[name] = {'sha256':expected,'bytes':(raw_root/name).stat().st_size}
    labels = read(raw_root/'banking77/banking_data/categories.json')
    excluded = ['card_arrival','pending_cash_withdrawal']
    variants = {'banking77': sorted(set(labels)-set(excluded))[:3], 'optical-digits':['0','1','2']}
    assignments = []
    for domain,start in (('banking77',160101),('optical-digits',160201)):
        for i in range(6):
            assignments.append({'pair_id':f'{domain}-pair-{i+1:02d}','domain':domain,
                'variant':variants[domain][i//2], 'donor_seed':start+2*i,'recipient_seed':start+2*i+1,
                'gauge_seed':start+1000+10*i, 'control_seed':start+2000+i})
    protocol = {'schema':'wsp-trained-symmetry-pilot-protocol/1','frozen_at_utc':now(),
        'issue':16,'scope':'Resource-capped estimation pilot; 24-pair confirmation proposal unchanged',
        'pair_count':12,'lineage_count':24,'independent_unit':'Disjoint initialization pairs; shared corpus/preprocessing within domain',
        'variants':variants,'support_variant_selection':'First three alphabetically, excluding historical fixed ownership labels',
        'support_variant_exclusions':excluded,'task':'Route the target original class to the appended specialist; retain every other original class',
        'training':{'hidden':64,'epochs':35,'learning_rate':.015,'batch_size':256,'weight_decay':.001,
            'threads':4,'epoch_selection':'highest development class accuracy; earliest tie'},
        'roles':'Exact-group stratified train60/development10/target10/protected10/calibration remainder; fresh fixed salt',
        'preprocessing':'Banking training-role-only text vocabulary width2048; digits divide16',
        'fit':{'helper':'compile_arm','mode':'soft','ridge':.01,'retention_weight':8.,'target_cap':32,'retain_cap':128,
               'selection':'One fixed candidate, no grid or feedback selection'},
        'arms':list(ARMS),'known_control_gauges':list(GAUGES),'assignments':assignments,
        'expected_independent_candidates':72,'expected_known_controls':48,
        'native_gate':{'atol':1e-5,'rtol':1e-5,'argmax':'lowest index; equality required','nonfinite':0},
        'alignment_ambiguity_policy':'Retain zero/duplicate-row diagnostics; helper deterministic assignment and unit-scale fallback for zero rows; no rejection replacement or reselection',
        'admission':{'structural':{'writable':['w2','b2'],'max_edits':1024,'max_parameter_abs':256.,'max_change_abs':256.},
            'real_box':[0.,1.],'score_drift_budget_per_output':1.,'witness':'one fixed all-zero input',
            'native':'Torch CPU float32 empirical error; round endpoint cast',
            'scope':'Exact box score bound and one real/native witness; sampled calibration retention separate; no whole-domain decision proof'},
        'useful_sampled':{'target_gain_minimum':.05,'protected_accuracy_loss_maximum':.02,'requires':'valid literal realization, native bridge/control gates; sampled only'},
        'useful_admitted':'Useful sampled and accepted integrated admission; no-op never useful',
        'analysis':{'bootstrap_seed':161616,'resamples':10000,'interval':'paired pair-level percentile bootstrap95; descriptive, no powered confirmation',
            'primary_contrasts':['heuristic-minus-raw target gain','heuristic-minus-none target gain'],
            'confirmation_margins_preserved':{'aligned_minus_raw':.02,'aligned_minus_none':.05,'protected_loss':.02,
                'subgroup_loss':.05,'coverage_observed':.8,'coverage_lower':.6},
            'missing_bounds':{'gain':[-1.,1.],'harm':[-1.,1.],'paired_contrast':[-2.,2.],'accuracy':[0.,1.],'coverage':[0.,1.]}},
        'missing_lanes':['DeepAlign','NFN','DWS','ScaleGMN'],'stopping':'Complete every assignment once; failures retained, no replacements',
        'data_access':'Train CSV/ZIP training member only before seal; official tests parsed after external global seal',
        'gate_sha256':GATE_HASH,'release_sha256':sha(public/'artifacts/research/live-studies/evidence-release.json'),
        'license_scope':{'banking77':'PolyAI task-specific-datasets CC BY4.0; LICENSE pinned',
             'optical-digits':'E. Alpaydin and C. Kaynak, UCI dataset80, DOI10.24432/C50P49; CC BY4.0 notice in pinned source-page.html; archive has no separate license text'},
        'limitations':['shared public corpus','no learned comparator pool','no cross-architecture lane','no universal floating-point proof',
            'no private/OS hermeticity attestation','acquisition network time, peak memory, energy, hardware price unknown']}
    write(experiment/'protocol.json',protocol)
    snapshot = experiment/'source-freeze'
    for path in [public/'scripts/run_trained_symmetry_pilot.py', *sorted((public/'rlcd').rglob('*.py'))]:
        relative = path.relative_to(public)
        destination = snapshot/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
    sources = {p.relative_to(snapshot).as_posix():sha(p) for p in sorted(snapshot.rglob('*.py'))}
    # Public raw bytes are pinned rather than duplicated into the export.
    manifest = {'schema':'wsp-trained-symmetry-pilot-freeze/1','protocol_sha256':sha(experiment/'protocol.json'),
        'source_files':sources,'raw_files':raw_hashes,'gate_sha256':GATE_HASH,'evaluation_started':False,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':__import__('torch').__version__,
            'platform':platform.platform(),'processor':platform.processor(),'machine':platform.machine(),
            'logical_cpu_count':os.cpu_count(),'native_device':'cpu','threads':4}}
    write(experiment/'freeze.json',manifest)
    write(anchors/'freeze-anchor.json',{'freeze_sha256':sha(experiment/'freeze.json'),'protocol_sha256':sha(experiment/'protocol.json'),
        'source_script_sha256':sources['scripts/run_trained_symmetry_pilot.py'],'retained_at_utc':now()})
    print(json.dumps({'phase':'frozen','freeze_sha256':sha(experiment/'freeze.json'),
                      'protocol_sha256':sha(experiment/'protocol.json'),'variants':variants}),flush=True)

def check_freeze(public,anchors,raw_root=None):
    experiment = public/EXPERIMENT
    manifest,anchor = read(experiment/'freeze.json'),read(anchors/'freeze-anchor.json')
    if sha(experiment/'freeze.json') != anchor['freeze_sha256'] or sha(experiment/'protocol.json') != anchor['protocol_sha256']:
        raise ValueError('External freeze anchor mismatch')
    for name, expected in manifest['source_files'].items():
        if sha(experiment/'source-freeze'/name) != expected:
            raise ValueError('Frozen source mismatch: '+name)
    if Path(__file__).resolve() != (experiment/'source-freeze/scripts/run_trained_symmetry_pilot.py').resolve():
        raise ValueError('Execute frozen script after freeze')
    if raw_root:
        for name,row in manifest['raw_files'].items():
            if sha(raw_root/name) != row['sha256']:
                raise ValueError('Frozen raw bytes mismatch: '+name)
    return read(experiment/'protocol.json')

def admission(parent,candidate,program,pair):
    policy = StructuralPolicy(('w2','b2'),1024,256.,256.)
    request = bind_admission(parent,candidate,program,
        task=TaskSpec(pair['pair_id'],'edit','Route '+pair['variant']+' to specialist; retain other classes',
                      RELU_ARCHITECTURE,parent.sha256,{'variant':pair['variant']}),policy=policy,
        box=InputBox(np.zeros(parent.shape[0]),np.ones(parent.shape[0])),
        max_output_drift=[1.]*parent.shape[2],decision=DecisionRule('argmax','first_index'),
        witnesses=np.zeros((1,parent.shape[0])),execution=ExecutionSpec('torch','float32','cpu','matmul','round','empirical'))
    started = time.perf_counter()
    record = admit(request).to_dict()
    return record, time.perf_counter()-started

def select(public,raw_root,anchors):
    protocol = check_freeze(public,anchors,raw_root)
    artifact = public/ARTIFACT
    if (artifact/'global-selection-seal.json').exists():
        raise ValueError('Selection already sealed')
    started, accounting = time.perf_counter(), []
    for domain in ('banking77','optical-digits'):
        x,y,labels,groups,roles,prep = raw_data(raw_root,domain)
        preparation = {'preprocessing':prep,'roles':{k:v.tolist() for k,v in roles.items()},
            'group_ids':groups,'group_ids_sha256':digest(groups),'labels_sha256':digest(y.tolist()),'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
            'cross_role_exact_group_count':None,
            'role_counts':{k:len(v) for k,v in roles.items()},'near_duplicate_exclusion_verified':False}
        # Exact role membership is already enforced by grouped_roles; audit linearly.
        group_roles = {}
        for role,indices in roles.items():
            for i in indices:
                group_roles.setdefault(groups[i],set()).add(role)
        preparation['cross_role_exact_group_count'] = sum(len(r)>1 for r in group_roles.values())
        write(artifact/'preparation'/f'{domain}.json',preparation)
        for pair in (a for a in protocol['assignments'] if a['domain'] == domain):
            directory = artifact/'pairs'/pair['pair_id']
            if directory.exists():
                raise ValueError('Existing pair directory; partial invocation retained, no implicit retry')
            directory.mkdir(parents=True)
            try:
                target_label = labels.index(pair['variant'])
                targets = roles['target'][y[roles['target']] == target_label][:32]
                retained = roles['protected'][y[roles['protected']] != target_label][:128]
                if not len(targets) or not len(retained):
                    raise ValueError('Empty assigned fit subset')
                control_x = x[roles['development'][:128]]
                models,heads,records = [],[],[]
                for role in ('donor','recipient'):
                    seed = pair[role+'_seed']
                    trained = train_encoder(x,y,roles['train'],roles['development'],seed=seed,hidden=64,epochs=35)
                    model,head = bridge(trained,prep,labels,seed,domain)
                    original_scores = evaluate_head(head,trained['features'][roles['development'][:128]],native=True)
                    bridge_native = agreement(original_scores,native(model,control_x))
                    bridge_binary = agreement(original_scores,binary(model,control_x))
                    record = {k:v for k,v in trained.items() if k not in ('encoder','head','features')}
                    record.update({'lineage_seed':seed,'role':role,'head':head.to_dict(),'model_sha256':model.sha256,
                        'encoder_binding_sha256':head.metadata['encoder_sha256'],'preprocessing_sha256':digest(prep),
                        'native_bridge':bridge_native,'binary64_bridge':bridge_binary,
                        'zero_incoming_affine_rows':int(np.count_nonzero(np.linalg.norm(np.column_stack((model.w1,model.b1)),axis=1)==0)),
                        'duplicate_incoming_affine_rows':64-len(np.unique(np.column_stack((model.w1,model.b1)),axis=0)),
                        'training_batches':35*int(np.ceil(len(roles['train'])/256)),
                        'training_forward_rows':35*len(roles['train']),'development_forward_rows':35*len(roles['development']),
                        'selected_feature_forward_rows':len(x),'gradient_backward_batches':35*int(np.ceil(len(roles['train'])/256))})
                    save_model(directory/f'{role}.npz',model)
                    write(directory/f'{role}-lineage.json.gz',record)
                    if not bridge_native['pass'] or not bridge_binary['pass']:
                        raise ValueError('Full model native/binary64 bridge failed')
                    models.append(model);heads.append(head);records.append(record)
                    print(f"trained {pair['pair_id']} {role} {seed}: {trained['elapsed_seconds']:.2f}s",flush=True)
                donor,recipient = models
                donor_candidate,donor_program,donor_fit = fit(donor,heads[0],x[targets],x[retained])
                save_model(directory/'donor-candidate.npz',donor_candidate)
                write(directory/'donor-program.json',donor_program.to_dict())
                alignment,diagnostics = align_hidden(donor,recipient)
                wrong = Gauge(tuple(map(int,np.roll(np.array(alignment.permutation),1))),alignment.scales)
                candidates = {'none':(recipient,difference(recipient,recipient,'No transfer'))}
                for arm,g in (('raw',None),('heuristic',alignment),('wrong',wrong)):
                    candidates[arm] = realize(recipient,transfer_edit(donor,donor_program,recipient,g))
                random_candidate,random_program,random_diagnostics = random_edit(recipient,candidates['heuristic'][0],pair['control_seed'])
                candidates['matched_random'] = (random_candidate,random_program)
                fitted,fitted_program,recipient_fit = fit(recipient,heads[1],x[targets],x[retained])
                candidates['fixed_target_fit'] = fitted,fitted_program
                known = []
                for j,kind in enumerate(GAUGES):
                    g = gauge(kind,pair['gauge_seed']+j)
                    parent,program = transport(donor,donor_program,g)
                    candidate = apply_program(parent,program)
                    recovered,recovery = align_hidden(donor,parent)
                    correctly_transferred = apply_program(parent,transfer_edit(donor,donor_program,parent,recovered))
                    raw_control,_ = realize(parent,transfer_edit(donor,donor_program,parent))
                    wrong_control,_ = realize(parent,transfer_edit(donor,donor_program,parent,wrong))
                    controls = {'parent':agreement(native(donor,control_x),native(parent,control_x)),
                        'edited':agreement(native(donor_candidate,control_x),native(candidate,control_x)),
                        'recovered':agreement(native(candidate,control_x),native(correctly_transferred,control_x)),
                        'raw':agreement(native(candidate,control_x),native(raw_control,control_x)),
                        'wrong':agreement(native(candidate,control_x),native(wrong_control,control_x))}
                    for name,model in (('parent',parent),('candidate',candidate),('raw',raw_control),('wrong',wrong_control),('recovered',correctly_transferred)):
                        save_model(directory/'known'/f'{kind}-{name}.npz',model)
                    known.append({'gauge':kind,'action':g.to_dict(),'recovered_action':recovered.to_dict(),
                        'recovery_diagnostics':recovery,'controls':controls,
                        'encoder_feature_identity_sha256':digest({'parent':parent.sha256,'preprocessing':digest(prep),'native':'Torchfloat32'}),
                        'analytic_scope':'Positive dyadic/permutation ReLU algebra on lifted exact parameters; finite native tolerance, no universal float proof'})
                all_controls_pass = all(k['controls'][name]['pass'] for k in known for name in ('parent','edited','recovered'))
                selection = {'pair':pair,'terminal_status':'sealed','evaluation_started':False,
                    'target_label':target_label,'target_fit_rows':targets.tolist(),'retain_fit_rows':retained.tolist(),
                    'alignment':alignment.to_dict(),'alignment_diagnostics':diagnostics,'random_matching':random_diagnostics,
                    'donor_fit':donor_fit,'recipient_fit':recipient_fit,'known_controls':known,'all_known_controls_pass':all_controls_pass,'arms':{}}
                for arm,(candidate,program) in candidates.items():
                    save_model(directory/f'{arm}-candidate.npz',candidate)
                    write(directory/f'{arm}-program.json',program.to_dict())
                    structural = StructuralPolicy(('w2','b2'),1024,256.,256.).check(recipient,program)
                    rollback_pass = apply_program(candidate,rollback(recipient,program)).sha256 == recipient.sha256
                    native_binary = agreement(native(candidate,control_x),binary(candidate,control_x))
                    try:
                        record,seconds = admission(recipient,candidate,program,pair)
                        write(directory/f'{arm}-admission.json.gz',record)
                    except Exception as error:
                        selection['arms'][arm] = {'terminal_status':'failed','error_type':type(error).__name__,'error':str(error),
                            'candidate_sha256':candidate.sha256,'program_sha256':program.sha256}
                        print(f"failed {pair['pair_id']} {arm}: {type(error).__name__}: {error}",flush=True)
                        continue
                    selection['arms'][arm] = {'candidate_sha256':candidate.sha256,'program_sha256':program.sha256,
                        'terminal_status':'sealed','native_binary64':native_binary,
                        'binding_sha256':record['binding_sha256'],'admission_sha256':record['sha256'],
                        'admission_accepted':record['accepted'],'admission_gate_statuses':{k:v['status'] for k,v in record['gates'].items()},
                        'admission_seconds':seconds,'structural':structural,'rollback_pass':rollback_pass,
                        'no_op':candidate.sha256 == recipient.sha256,'edit_count':len(program.edits),
                        'edit_l2':float(np.sqrt(sum(np.square(candidate.tensors[k]-v).sum() for k,v in recipient.tensors.items())))}
                    print(f"selected {pair['pair_id']} {arm}; admission={record['accepted']} {seconds:.2f}s",flush=True)
                write(directory/'selection.json',selection)
                accounting.append({'pair_id':pair['pair_id'],'status':'sealed','selection_sha256':sha(directory/'selection.json'),
                    'arm_statuses':{arm:selection['arms'][arm]['terminal_status'] for arm in ARMS}})
            except Exception as error:
                write(directory/'failure.json',{'pair':pair,'terminal_status':'failed','error_type':type(error).__name__,
                    'error':str(error),'arms':{arm:{'terminal_status':'failed','reason':'Pair-level failure before complete arm selection'} for arm in ARMS},
                    'evaluation_started':False,'retained_at_utc':now()})
                accounting.append({'pair_id':pair['pair_id'],'status':'failed','failure_sha256':sha(directory/'failure.json'),
                    'arm_statuses':{arm:'failed' for arm in ARMS}})
                print(f"failed {pair['pair_id']}: {type(error).__name__}: {error}",flush=True)
    files = {p.relative_to(artifact).as_posix():sha(p) for p in sorted(artifact.rglob('*')) if p.is_file()}
    seal = {'schema':'wsp-trained-symmetry-pilot-global-seal/1','phase':'selection_complete_before_evaluation',
        'freeze_sha256':sha(public/EXPERIMENT/'freeze.json'),'accounting':accounting,'assigned_pairs':12,
        'all_assigned_accounted':len(accounting)==12,'evaluation_started':False,'files':files,'sealed_at_utc':now(),
        'selection_seconds':time.perf_counter()-started,'native_accounting':dict(LEDGER)}
    write(artifact/'global-selection-seal.json',seal)
    write(anchors/'selection-anchor.json',{'seal_sha256':sha(artifact/'global-selection-seal.json'),
        'freeze_sha256':seal['freeze_sha256'],'retained_at_utc':now()})
    print(json.dumps({'phase':'sealed','seal_sha256':sha(artifact/'global-selection-seal.json'),'accounting':accounting}),flush=True)

def check_seal(public,anchors):
    artifact = public/ARTIFACT
    seal,anchor = read(artifact/'global-selection-seal.json'),read(anchors/'selection-anchor.json')
    if sha(artifact/'global-selection-seal.json') != anchor['seal_sha256'] or not seal['all_assigned_accounted'] or seal['evaluation_started']:
        raise ValueError('Externally retained global selection seal invalid')
    for name,expected in seal['files'].items():
        if sha(artifact/name) != expected:
            raise ValueError('Sealed evidence mismatch: '+name)
    return seal

def metrics(parent,candidate,x,y,label):
    a,b = native(parent,x),native(candidate,x)
    pa,pb = a.argmax(1),b.argmax(1)
    target,retain = y == label,y != label
    expected = y.copy(); expected[target] = parent.shape[2]-1
    before,after = pa == expected,pb == expected
    subgroup = [{'label':int(l),'rows':int((y == l).sum()),'accuracy_loss':float((before[y == l].astype(float)-after[y == l]).mean())}
                for l in np.unique(y[retain])]
    return {'target_rows':int(target.sum()),'retain_rows':int(retain.sum()),
        'parent_target_accuracy':float(before[target].mean()),'target_accuracy':float(after[target].mean()),
        'target_gain':float((after[target].astype(float)-before[target]).mean()),
        'parent_retain_accuracy':float(before[retain].mean()),'retain_accuracy':float(after[retain].mean()),
        'protected_accuracy_loss':float((before[retain].astype(float)-after[retain]).mean()),
        'protected_decision_flip_rate':float((pa[retain] != pb[retain]).mean()),
        'maximum_protected_score_drift':float(np.abs(a[retain].astype(float)-b[retain]).max()),
        'maximum_subgroup_accuracy_loss':max(s['accuracy_loss'] for s in subgroup),'subgroups':subgroup,
        'nonfinite_values':int((~np.isfinite(a)).sum()+(~np.isfinite(b)).sum())}

def estimate(values, assigned, bounds, rng):
    a = np.array(values,dtype=float)
    missing = assigned-len(a)
    if len(a):
        interval = np.quantile(rng.choice(a,(10000,len(a)),replace=True).mean(1),[.025,.975]).tolist()
        mean = float(a.mean())
    else:
        interval,mean = None,None
    return {'observed_count':len(a),'assigned_count':assigned,'missing_count':missing,'observed_mean':mean,
        'bootstrap_95_percentile_interval':interval,
        'all_assigned_worst_best_mean':[(float(a.sum())+missing*b)/assigned for b in bounds]}

def evaluate(public,raw_root,anchors,replay=False):
    protocol = check_freeze(public,anchors,raw_root)
    seal = check_seal(public,anchors)
    artifact = public/ARTIFACT
    if (artifact/'evaluation.json').exists() and not replay:
        raise ValueError('Evaluation already exists; no outcome replacement')
    started, rows, known_rows, replay_checks, overlap = time.perf_counter(), [], [], [], {}
    for domain in ('banking77','optical-digits'):
        preparation = read(artifact/'preparation'/f'{domain}.json')
        prep = preparation['preprocessing']
        # First official held-out member/CSV parse occurs here, after anchor authentication.
        x,y,labels,groups,_,_ = raw_data(raw_root,domain,True,prep)
        shared = set(groups) & set(preparation['group_ids'])
        overlap[domain] = {'training_rows':len(preparation['group_ids']),'official_test_rows':len(groups),
            'shared_exact_groups':len(shared),'test_rows_in_shared_exact_groups':sum(g in shared for g in groups),
            'policy':'Report full official test results without post-seal exclusions; near duplicates not audited'}
        for assignment in (a for a in protocol['assignments'] if a['domain']==domain):
            directory = artifact/'pairs'/assignment['pair_id']
            if not (directory/'selection.json').exists():
                continue
            selection = read(directory/'selection.json')
            parent = load_model(directory/'recipient.npz')
            donor = load_model(directory/'donor.npz')
            donor_candidate = load_model(directory/'donor-candidate.npz')
            for arm in ARMS:
                if selection['arms'][arm]['terminal_status'] != 'sealed':
                    continue
                candidate = load_model(directory/f'{arm}-candidate.npz')
                program = Program.from_dict(read(directory/f'{arm}-program.json'))
                identity_pass = (candidate.sha256 == selection['arms'][arm]['candidate_sha256'] and
                    program.sha256 == selection['arms'][arm]['program_sha256'] and
                    apply_program(parent,program).sha256 == candidate.sha256 and apply_program(candidate,rollback(parent,program)).sha256 == parent.sha256)
                if not identity_pass:
                    raise ValueError('Endpoint/atomic/rollback replay failure')
                if replay:
                    original = read(directory/f'{arm}-admission.json.gz')
                    recomputed,_ = admission(parent,candidate,program,assignment)
                    admission_pass = digest(recomputed) == digest(original)
                    if not admission_pass:
                        raise ValueError('Fresh-process admission differs')
                    replay_checks.append({'pair_id':assignment['pair_id'],'arm':arm,'endpoint_rollback':True,'admission':True})
                m = metrics(parent,candidate,x,y,selection['target_label'])
                heldout_native_binary = agreement(native(candidate,x),binary(candidate,x))
                valid = (selection['arms'][arm]['structural']['structural_accepted'] and selection['all_known_controls_pass']
                    and selection['arms'][arm]['native_binary64']['pass'] and selection['arms'][arm]['rollback_pass'] and heldout_native_binary['pass'])
                useful = bool(valid and not selection['arms'][arm]['no_op'] and m['nonfinite_values']==0 and
                    m['target_gain']>=.05 and m['protected_accuracy_loss']<=.02)
                rows.append({'pair_id':assignment['pair_id'],'domain':domain,'variant':assignment['variant'],'arm':arm,
                    **m,'heldout_native_binary64':heldout_native_binary,'valid_literal_native':bool(valid),'useful_sampled':useful,
                    'integrated_admission_accepted':selection['arms'][arm]['admission_accepted'],
                    'useful_admitted':useful and selection['arms'][arm]['admission_accepted']})
            for kind in GAUGES:
                kp = load_model(directory/'known'/f'{kind}-parent.npz')
                kc = load_model(directory/'known'/f'{kind}-candidate.npz')
                kr = load_model(directory/'known'/f'{kind}-raw.npz')
                kw = load_model(directory/'known'/f'{kind}-wrong.npz')
                recovered = load_model(directory/'known'/f'{kind}-recovered.npz')
                known_rows.append({'pair_id':assignment['pair_id'],'domain':domain,'gauge':kind,
                    'parent_agreement':agreement(native(donor,x),native(kp,x)),
                    'edited_agreement':agreement(native(donor_candidate,x),native(kc,x)),
                    'recovered_agreement':agreement(native(kc,x),native(recovered,x)),
                    'correct':metrics(kp,kc,x,y,selection['target_label']),
                    'raw':metrics(kp,kr,x,y,selection['target_label']),
                    'wrong':metrics(kp,kw,x,y,selection['target_label']),
                    'raw_edited_agreement':agreement(native(kc,x),native(kr,x)),
                    'wrong_edited_agreement':agreement(native(kc,x),native(kw,x))})
            print(f"{'replayed' if replay else 'evaluated'} {assignment['pair_id']}",flush=True)
    rng = np.random.default_rng(161616)
    summaries = {}
    for domain in ('all','banking77','optical-digits'):
        assigned = 12 if domain=='all' else 6
        summary = {}
        for arm in ARMS:
            chosen = [r for r in rows if r['arm']==arm and (domain=='all' or r['domain']==domain)]
            summary[arm] = {k:estimate([r[k] for r in chosen],assigned,bounds,rng) for k,bounds in
                (('target_gain',[-1.,1.]),('protected_accuracy_loss',[-1.,1.]),('useful_sampled',[0.,1.]),
                 ('useful_admitted',[0.,1.]),('integrated_admission_accepted',[0.,1.]))}
        contrasts = {}
        for comparator in ('raw','none','wrong','matched_random','fixed_target_fit'):
            values=[]
            for pair in protocol['assignments']:
                selected = {r['arm']:r for r in rows if r['pair_id']==pair['pair_id'] and (domain=='all' or r['domain']==domain)}
                if 'heuristic' in selected and comparator in selected:
                    values.append(selected['heuristic']['target_gain']-selected[comparator]['target_gain'])
            contrasts['heuristic_minus_'+comparator] = estimate(values,assigned,[-2.,2.],rng)
        summaries[domain] = {'arms':summary,'paired_target_gain_contrasts':contrasts}
    result = {'schema':'wsp-trained-symmetry-pilot-evaluation/1','seal_sha256':sha(artifact/'global-selection-seal.json'),
        'freeze_sha256':sha(public/EXPERIMENT/'freeze.json'),'assigned_pairs':12,'evaluated_pairs':len({r['pair_id'] for r in rows}),
        'rows':rows,'known_controls':known_rows,'summaries':summaries,'official_test_overlap_audit':overlap,'missing_lanes':protocol['missing_lanes'],
        'claim_scope':'Descriptive finite public-domain initialization pilot; no learned-method superiority, circuit or whole-domain float proof'}
    if replay:
        original = read(artifact/'evaluation.json')
        passed = digest(original)==digest(result)
        if not passed:
            raise ValueError('Fresh-process full held-out metrics differ')
        write(artifact/'independent-replay.json',{'schema':'wsp-trained-symmetry-pilot-replay/1','pass':passed,
            'evaluation_sha256':sha(artifact/'evaluation.json'),'canonical_evaluation_sha256':digest(result),
            'checks':replay_checks,'elapsed_seconds':time.perf_counter()-started,'native_accounting':LEDGER,
            'scope':'Fresh frozen-source process: external anchors, actual native full-model scores/metrics, programs, rollback and integrated admissions; no retraining'})
    else:
        write(artifact/'evaluation.json',result)
        write(artifact/'evaluation-accounting.json',{'elapsed_seconds':time.perf_counter()-started,'native_accounting':LEDGER,
              'evaluated_at_utc':now(),'torch_operator_accounting_internal_to_admission':'One fixed witness per endpoint per arm; internal arithmetic work not instrumented',
              'unmeasured':['acquisition network time','peak memory','energy','hardware price','human setup/editor cost']})
    print(json.dumps({'phase':'replayed' if replay else 'evaluated','canonical_result_sha256':digest(result),
        'evaluated_pairs':result['evaluated_pairs']}),flush=True)

def selfcheck():
    base = Network(np.eye(2,dtype=np.float32),np.zeros(2,dtype=np.float32),
        np.array([[1.,0.],[0.,1.]],dtype=np.float32),np.zeros(2,dtype=np.float32))
    candidate = base.replace(w2=np.array([[2.,0.],[0.,1.]],dtype=np.float32))
    program = difference(base,candidate,'check')
    g = Gauge((1,0),(2.,.5))
    transformed,p = transport(base,program,g)
    inputs = np.array([[1.,2.],[2.,1.]],dtype=np.float32)
    assert agreement(native(candidate,inputs),native(apply_program(transformed,p),inputs))['pass']
    assert not agreement(native(candidate,inputs),native(apply_program(transformed,transfer_edit(base,program,transformed)),inputs))['pass']
    assert apply_program(candidate,rollback(base,program)).sha256 == base.sha256
    try:
        apply_program(transformed,program)
    except ValueError:
        pass
    else:
        raise AssertionError('Stale parent accepted')
    random,_,matching = random_edit(base,candidate,12)
    assert all(v['count_matched'] and v['relative_l2_mismatch'] < 1e-6 for v in matching.values())
    assert to_tensor_artifact(base).architecture == RELU_ARCHITECTURE
    receipt,_ = admission(base,candidate,program,{'pair_id':'synthetic','variant':'0'})
    assert receipt['gates']['binding']['status'] == 'pass'
    assert receipt['gates']['structural']['status'] == 'pass'
    assert receipt['gates']['real_box_drift']['status'] == 'pass'
    assert receipt['gates']['native_execution']['status'] == 'pass'
    temporary = Path.cwd()/'.runs/trained-symmetry-selfcheck'/uuid.uuid4().hex
    temporary.mkdir(parents=True,exist_ok=False)
    # Retain the workspace fixture for inspection; no temporary-tree deletion.
    public,anchors = temporary/'public',temporary/'anchors'
    try:
        check_seal(public,anchors)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Evaluation without seal accepted')
    evidence = public/ARTIFACT/'evidence.json'
    write(evidence,{'value':1})
    seal_path = public/ARTIFACT/'global-selection-seal.json'
    write(seal_path,{'all_assigned_accounted':True,'evaluation_started':False,'files':{'evidence.json':sha(evidence)}})
    write(anchors/'selection-anchor.json',{'seal_sha256':sha(seal_path)})
    check_seal(public,anchors)
    write(evidence,{'value':2})
    try:
        check_seal(public,anchors)
    except ValueError:
        pass
    else:
        raise AssertionError('Tampered sealed evidence accepted')
    print('synthetic selfcheck passed',flush=True)

def main():
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('selfcheck','freeze','select','evaluate','replay'))
    parser.add_argument('--public-root',type=Path,default=ROOT)
    parser.add_argument('--raw-root',type=Path)
    parser.add_argument('--anchors',type=Path)
    args = parser.parse_args()
    if args.phase=='selfcheck':
        selfcheck();return
    if args.raw_root is None or args.anchors is None:
        parser.error('--raw-root and --anchors required')
    public,raw_root,anchors = args.public_root.resolve(),args.raw_root.resolve(),args.anchors.resolve()
    if anchors == public/EXPERIMENT or public/EXPERIMENT in anchors.parents or anchors == public/ARTIFACT or public/ARTIFACT in anchors.parents:
        raise ValueError('Anchors must stay outside mutable study directories')
    if args.phase=='freeze':freeze(public,raw_root,anchors)
    elif args.phase=='select':select(public,raw_root,anchors)
    else:evaluate(public,raw_root,anchors,args.phase=='replay')

if __name__=='__main__':
    main()
