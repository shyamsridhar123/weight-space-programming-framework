"""Independent stdlib authentication/accounting audit; never edits sealed inputs."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import uuid

EXPERIMENT = 'experiments/research/trained-symmetry-pilot'
ARTIFACT = 'artifacts/research/trained-symmetry-pilot'

def read(path):
    with (gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz')
          else Path(path).open(encoding='utf-8')) as stream:
        return json.load(stream)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def verify_files(root, files):
    root = root.resolve()
    for name, expected in files.items():
        path = (root/name).resolve()
        path.relative_to(root)
        if sha(path) != expected:
            raise ValueError('Hash mismatch: '+name)

def audit(public,anchors,fixtures):
    experiment,artifact = public/EXPERIMENT,public/ARTIFACT
    freeze,anchor = read(experiment/'freeze.json'),read(anchors/'freeze-anchor.json')
    assert sha(experiment/'freeze.json') == anchor['freeze_sha256']
    assert sha(experiment/'protocol.json') == anchor['protocol_sha256'] == freeze['protocol_sha256']
    verify_files(experiment/'source-freeze',freeze['source_files'])
    protocol = read(experiment/'protocol.json')
    seal = read(artifact/'global-selection-seal.json')
    assert sha(artifact/'global-selection-seal.json') == read(anchors/'selection-anchor.json')['seal_sha256']
    assert seal['freeze_sha256'] == sha(experiment/'freeze.json')
    assert seal['all_assigned_accounted'] and not seal['evaluation_started']
    verify_files(artifact,seal['files'])
    assigned = {r['pair_id']:r for r in protocol['assignments']}
    accounting = {r['pair_id']:r for r in seal['accounting']}
    assert len(assigned) == len(accounting) == 12 and set(assigned) == set(accounting)
    assert len({r[key] for r in assigned.values() for key in ('donor_seed','recipient_seed')}) == 24
    arm_statuses,gate_statuses = Counter(),Counter()
    selections, lineages, admissions = {},[],[]
    controls = 0
    for pair_id,row in accounting.items():
        directory = artifact/'pairs'/pair_id
        for role in ('donor','recipient'):
            path = directory/f'{role}-lineage.json.gz'
            if path.exists():
                lineage = read(path)
                assert lineage['lineage_seed'] == assigned[pair_id][role+'_seed']
                lineages.append(lineage)
        if row['status'] == 'sealed':
            assert sha(directory/'selection.json') == row['selection_sha256']
            selection = read(directory/'selection.json')
            assert selection['pair'] == assigned[pair_id] and not selection['evaluation_started']
            assert set(selection['arms']) == set(protocol['arms'])
            assert [r['gauge'] for r in selection['known_controls']] == protocol['known_control_gauges']
            controls += len(selection['known_controls'])
            selections[pair_id] = selection
            for arm,record in selection['arms'].items():
                assert record['terminal_status'] == row['arm_statuses'][arm]
                arm_statuses[record['terminal_status']] += 1
                if record['terminal_status'] != 'sealed':
                    continue
                program = read(directory/f'{arm}-program.json')
                assert digest({k:v for k,v in program.items() if k != 'sha256'}) == record['program_sha256']
                admission = read(directory/f'{arm}-admission.json.gz')
                assert admission['sha256'] == record['admission_sha256']
                assert digest({k:v for k,v in admission.items() if k != 'sha256'}) == admission['sha256']
                assert admission['binding_sha256'] == record['binding_sha256']
                assert admission['accepted'] == record['admission_accepted']
                for gate,body in admission['gates'].items():
                    assert body['binding_sha256'] == admission['binding_sha256']
                    gate_statuses[gate+'/'+body['status']] += 1
                admissions.append(admission['accepted'])
        else:
            failure = read(directory/'failure.json')
            assert sha(directory/'failure.json') == row['failure_sha256']
            assert set(failure['arms']) == set(protocol['arms'])
            arm_statuses.update(r['terminal_status'] for r in failure['arms'].values())
    assert sum(arm_statuses.values()) == 72
    assert len({r['initialization_sha256'] for r in lineages}) == len(lineages)
    supplemental = read(artifact/'supplemental-control-replay.json')
    assert supplemental['pass'] and len(supplemental['checks']) == controls
    assert supplemental['seal_sha256'] == sha(artifact/'global-selection-seal.json')
    assert supplemental['replay_source_sha256'] == sha(public/'scripts/replay_trained_symmetry_controls.py')
    for check in supplemental['checks']:
        assert digest(check['transported_program']) == check['program_sha256']
    attempts = read(artifact/'supplemental-control-replay-attempts.json')
    assert attempts['frozen_study_retries'] == 0
    assert len(attempts['attempts']) == 2
    assert attempts['attempts'][0]['status'] == 'timeout'
    assert attempts['attempts'][0]['shell_exit_code'] == 124
    assert attempts['attempts'][0]['stdout_sha256'] == sha(anchors/'supplemental-control-replay.log')
    assert attempts['attempts'][1]['status'] == 'completed'
    assert attempts['attempts'][1]['exit_code'] == 0
    assert attempts['attempts'][1]['log_sha256'] == sha(anchors/'supplemental-control-replay-attempt-2.log')
    evaluation = read(artifact/'evaluation.json')
    failure = read(artifact/'independent-replay-failure.json')
    assert evaluation['seal_sha256'] == sha(artifact/'global-selection-seal.json')
    assert failure['status'] == 'failed' and not failure['pass']
    assert failure['canonical_evaluation_sha256'] == digest(evaluation)
    assert failure['evaluation_sha256'] == sha(artifact/'evaluation.json')
    assert failure['freeze_sha256'] == sha(experiment/'freeze.json')
    assert failure['seal_sha256'] == sha(artifact/'global-selection-seal.json')
    assert failure['attempts'] == 1 and failure['retries'] == 0
    assert failure['completed_pair_replays'] == failure['completed_admission_replays'] == 0
    assert not failure['successful_native_replay']
    assert not (artifact/'independent-replay.json').exists()
    for name,evidence in failure['external_evidence'].items():
        assert evidence['sha256'] == sha(anchors/name)
        assert evidence['bytes'] == (anchors/name).stat().st_size
    assert failure['error'] in (anchors/'replay-stderr.log').read_text(encoding='utf-8')
    rows = {(r['pair_id'],r['arm']):r for r in evaluation['rows']}
    assert len(rows) == len(evaluation['rows']) == arm_statuses['sealed']
    assert len(evaluation['known_controls']) == controls
    assert evaluation['assigned_pairs'] == evaluation['evaluated_pairs'] == 12
    for (pair_id,arm),row in rows.items():
        assert row['domain'] == assigned[pair_id]['domain']
        assert row['variant'] == assigned[pair_id]['variant']
        assert row['integrated_admission_accepted'] == selections[pair_id]['arms'][arm]['admission_accepted']
        assert not row['useful_admitted'] or row['useful_sampled'] and row['integrated_admission_accepted']
    for domain,summary in evaluation['summaries'].items():
        count = 12 if domain == 'all' else 6
        for arm,metrics in summary['arms'].items():
            chosen = [r for r in rows.values() if r['arm'] == arm and (domain == 'all' or r['domain'] == domain)]
            for key,estimate in metrics.items():
                assert estimate['assigned_count'] == count
                assert estimate['observed_count'] == len(chosen)
                assert estimate['missing_count'] == count-len(chosen)
                if chosen:
                    assert abs(estimate['observed_mean']-sum(r[key] for r in chosen)/len(chosen)) < 1e-12
    fixtures.mkdir(parents=True,exist_ok=True)
    fixture = fixtures/uuid.uuid4().hex
    fixture.mkdir()
    evidence = fixture/'evidence.json'
    evidence.write_text('{"value":1}\n',encoding='utf-8')
    fixture_hash = sha(evidence)
    verify_files(fixture,{'evidence.json':fixture_hash})
    evidence.write_text('{"value":2}\n',encoding='utf-8')
    try:
        verify_files(fixture,{'evidence.json':fixture_hash})
    except ValueError:
        tamper = True
    else:
        raise AssertionError('Tampered fixture accepted')
    receipt = {'schema':'wsp-trained-symmetry-pilot-stdlib-audit/2','pass':True,
        'audited_at_utc':datetime.now(timezone.utc).isoformat(),
        'audit_source_sha256':sha(Path(__file__)),'freeze_sha256':sha(experiment/'freeze.json'),
        'protocol_sha256':sha(experiment/'protocol.json'),'seal_sha256':sha(artifact/'global-selection-seal.json'),
        'canonical_evaluation_sha256':digest(evaluation),'evaluation_sha256':sha(artifact/'evaluation.json'),
        'native_replay_failure_sha256':sha(artifact/'independent-replay-failure.json'),
        'supplemental_control_replay_sha256':sha(artifact/'supplemental-control-replay.json'),
        'supplemental_control_attempts_sha256':sha(artifact/'supplemental-control-replay-attempts.json'),
        'sealed_file_count':len(seal['files']),'source_file_count':len(freeze['source_files']),
        'assigned_pairs':12,'assigned_lineages':24,'retained_lineages':len(lineages),
        'assigned_arms':72,'arm_terminal_status_counts':dict(arm_statuses),
        'assigned_known_controls':48,'retained_known_controls':controls,
        'integrated_admission_accepted':sum(admissions),'admission_gate_status_counts':dict(gate_statuses),
        'synthetic_tamper_rejected':tamper,
        'native_replay_verified':False,'completed_native_admission_replays':0,
        'literal_control_reconstruction_verified':True,
        'verification_gaps':['Frozen native admission/metric replay failed before first completed admission; this audit does not establish independent native recomputation'],
        'scope':'Independent stdlib source/file/external-anchor authentication, original admission receipt digests, complete denominators and saved-summary arithmetic, successful separate literal-control reconstruction, failed native replay and both unfrozen control-audit attempts. No dataset parse, selection, or native metric/admission recomputation.'}
    path = artifact/'stdlib-audit.json'
    assert not path.exists()
    path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(receipt,sort_keys=True))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-root',type=Path,required=True)
    parser.add_argument('--anchors',type=Path,required=True)
    parser.add_argument('--fixtures',type=Path,required=True)
    args=parser.parse_args()
    audit(args.public_root.resolve(),args.anchors.resolve(),args.fixtures.resolve())
