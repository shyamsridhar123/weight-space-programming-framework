"""Separate stdlib authentication of final pilot documentation and retained failures.

Reads saved JSON and hashes bytes. Does not import study code, parse raw observations,
retrain, fit, select, evaluate, recompute admissions, replay, or replace any receipt.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

EXPERIMENT='experiments/research/trained-symmetry-pilot'
ARTIFACT='artifacts/research/trained-symmetry-pilot'
HISTORY='.runs/research-2026-11-08'

EXPECTED={
 'experiments/research/trained-symmetry-pilot/freeze.json':'f5e067f5f61d2c59830e6c895d697b02106b0945e224635fc3cda39129cfe4c7',
 'experiments/research/trained-symmetry-pilot/protocol.json':'f3187be850bb9d413fbce8951b8215865f9c10a982fe054da0b8c86afcd716d8',
 'scripts/run_trained_symmetry_pilot.py':'118c741586369582e8bab39325c95f1d2fd7251544b8551ba30f4a9fb8d71564',
 'scripts/audit_trained_symmetry_pilot.py':'0818fbcf29c544b94182b326cafc491b6839c6668ca8da2760de3ecce078e38c',
 'scripts/replay_trained_symmetry_normalized_names.py':'7978d979c09e2d7ce2051fcb3f9c8b20ef60c5a02dc9cd1bbbf8688eb67c3840',
 'scripts/summarize_trained_symmetry_pilot.py':'53472d76d0db6a86493c62fb559b75a46afe5ded416edc6f5e3d7582b594be2c',
 ARTIFACT+'/global-selection-seal.json':'4641b45692eeb61f4564c6f1de5458e8bf408ec0aa07b9a2c2541dcb38276bae',
 ARTIFACT+'/evaluation.json':'8a452a4a8ce3d61bd1203096d1985be8e7664c561f4359c8a58f93713cd3acf0',
 ARTIFACT+'/independent-replay-failure.json':'fa9198aa51febfabbd610d0a46600cca7a14176d2c323092266a3469258d351b',
 ARTIFACT+'/supplemental-control-replay.json':'55a7a851dbcf1a9818c79161122e04f8189ab2b0a3eeeca83025ef763291db43',
 ARTIFACT+'/supplemental-control-replay-attempts.json':'0f037c9429da3d46d5cb58e3f19e10624416ea01e628f292ab34a8501e6e8494',
 ARTIFACT+'/stdlib-audit.json':'2c69e7706a2b48f177f1f3a9c8f7c354baa15818b47d99fc3036a083b9181e69',
 ARTIFACT+'/posthoc-normalized-interface-verification.json':'eaff45ff6f327bc2a2cca8349374bcdc00c7f12043de2e482da6c8cc4c74453f',
 ARTIFACT+'/posthoc-normalized-interface-progress.json':'22018b52d4812c9990e91b7b7c34f99ba9916c3a97e0fab685ef8894ea376638',
 ARTIFACT+'/saved-artifact-summary.json':'e9d1d682454097ae5370195e2f032acaf1c0a63d3b5539e38b54d021b5253a45',
 HISTORY+'/issue-transfer.json':'b505957631f99c56faf8f15ce5c5d356b022b10da6b2f9c72a6a9fb19759b3ed',
 HISTORY+'/remaining-study-design.md':'3f1d3533af423d962cce4fbe9ebf396b7046e7e61790bd5051d7c05a949c1a14',
 HISTORY+'/remaining-study-checklist.json':'e160db2ece031cda07a06288a05914528a9b9a2107d50741b2e0e4168b201f3e',
}

def read(path):
    return json.loads(path.read_bytes())

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def verify(root,name,expected):
    path=(root/name).resolve()
    path.relative_to(root)
    actual=sha(path)
    assert actual==expected, 'Hash mismatch: '+name
    return actual

def authenticate(public,anchors,raw,documentation_source):
    public,anchors,raw=public.resolve(),anchors.resolve(),raw.resolve()
    artifact=public/ARTIFACT
    output=artifact/'supplemental-documentation-authentication.json'
    assert not output.exists(),'Do not replace an authentication receipt'
    pinned={name:verify(public,name,expected) for name,expected in EXPECTED.items()}
    freeze=read(public/EXPERIMENT/'freeze.json')
    protocol=read(public/EXPERIMENT/'protocol.json')
    seal=read(artifact/'global-selection-seal.json')
    freeze_anchor=read(anchors/'freeze-anchor.json')
    selection_anchor=read(anchors/'selection-anchor.json')
    assert pinned[EXPERIMENT+'/freeze.json']==freeze_anchor['freeze_sha256']==selection_anchor['freeze_sha256']
    assert pinned[EXPERIMENT+'/protocol.json']==freeze_anchor['protocol_sha256']==freeze['protocol_sha256']
    assert freeze_anchor['source_script_sha256']==EXPECTED['scripts/run_trained_symmetry_pilot.py']
    assert pinned[ARTIFACT+'/global-selection-seal.json']==selection_anchor['seal_sha256']
    assert seal['freeze_sha256']==freeze_anchor['freeze_sha256']
    assert seal['all_assigned_accounted'] and not seal['evaluation_started']
    assert len(freeze['source_files'])==31 and len(seal['files'])==542
    for name,expected in freeze['source_files'].items():
        verify((public/EXPERIMENT/'source-freeze').resolve(),name,expected)
    for name,expected in seal['files'].items():
        verify(artifact.resolve(),name,expected)
    raw_checks={}
    for name,evidence in freeze['raw_files'].items():
        path=(raw/name).resolve()
        path.relative_to(raw)
        assert path.stat().st_size==evidence['bytes']
        verify(raw,name,evidence['sha256'])
        raw_checks[name]=dict(evidence)
    evaluation=read(artifact/'evaluation.json')
    original=read(artifact/'independent-replay-failure.json')
    audit=read(artifact/'stdlib-audit.json')
    normalized=read(artifact/'posthoc-normalized-interface-verification.json')
    progress=read(artifact/'posthoc-normalized-interface-progress.json')
    summary=read(artifact/'saved-artifact-summary.json')
    assert digest(evaluation)=='a218166e568e691340312b50020adc62c83e241931fc1d146c1f7e2e67fba347'
    assert not original['pass'] and original['status']=='failed'
    assert original['attempts']==1 and original['retries']==0
    assert original['completed_admission_replays']==original['completed_pair_replays']==0
    assert audit['pass'] and not audit['native_replay_verified'] and audit['completed_native_admission_replays']==0
    literal=read(artifact/'supplemental-control-replay.json')
    attempts=read(artifact/'supplemental-control-replay-attempts.json')
    assert literal['pass'] and len(literal['checks'])==48
    assert literal['seal_sha256']==selection_anchor['seal_sha256']
    verify(public,'scripts/replay_trained_symmetry_controls.py',literal['replay_source_sha256'])
    assert attempts['frozen_study_retries']==0 and len(attempts['attempts'])==2
    assert attempts['attempts'][0]['status']=='timeout' and attempts['attempts'][0]['shell_exit_code']==124
    assert not attempts['attempts'][0]['success_receipt_present_after_attempt']
    assert not attempts['attempts'][0]['worker_remaining_after_attempt']
    assert attempts['attempts'][1]['status']=='completed' and attempts['attempts'][1]['exit_code']==0
    verify(anchors,'supplemental-control-replay.log',attempts['attempts'][0]['stdout_sha256'])
    verify(anchors,'supplemental-control-replay-attempt-2.log',attempts['attempts'][1]['log_sha256'])
    assert normalized['status']=='failed' and not normalized['pass'] and normalized['exit_code']==1
    assert normalized['attempts']==1 and normalized['retries']==0 and normalized['timeout_seconds']==1800
    assert normalized['cap_authorization_history'][0]['timeout_seconds']==300
    assert normalized['cap_authorization_history'][1]['timeout_seconds']==1800
    assert all(not x['invocation_started'] for x in normalized['cap_authorization_history'])
    assert normalized['wrapper_source_sha256']==EXPECTED['scripts/replay_trained_symmetry_normalized_names.py']
    assert normalized['progress_sha256']==EXPECTED[ARTIFACT+'/posthoc-normalized-interface-progress.json']
    assert normalized['original_evidence_unchanged'] and not normalized['native_replay_verified']
    for name,expected in normalized['original_evidence_sha256'].items():
        verify(artifact.resolve(),name,expected)
    assert normalized['completed_derivative_admissions']==progress['completed_admissions']==1
    assert normalized['normalized_model_loads']==progress['normalized_model_loads']==4
    assert len(progress['model_checks'])==4
    for row in progress['model_checks']:
        assert row['tensor_bytes_dtype_shape_identical'] and row['ordered_textual_names_identical']
        assert row['original_name_types']==['str_'] and row['normalized_name_type']=='str'
        assert row['file'] in seal['files']
    assert 'Fresh-process admission differs' in progress['worker_error']
    success_absence={name:not (artifact/name).exists() for name in
        ('independent-replay.json','posthoc-normalized-interface-replay.json')}
    assert all(success_absence.values()) and not normalized['derivative_success_receipt_present']
    launch=read(anchors/'posthoc-normalized-interface-launch.json')
    assert launch['pid']==43408 and launch['actual_timeout_seconds']==1800 and launch['attempts']==1
    assert launch['prior_300_second_authorization_was_not_launched']
    assert launch['wrapper_source_sha256']==normalized['wrapper_source_sha256']
    ext=normalized['external_log']
    assert ext['sha256']=='ea0981a771f998e2cfe27cc8cde38e3bf36768949e6381c46e771cf0c7d0c344'
    assert (anchors/ext['name']).stat().st_size==ext['bytes']==814
    verify(anchors,ext['name'],ext['sha256'])
    assert 'Fresh-process admission differs' in (anchors/ext['name']).read_text(encoding='utf-8')
    for name,evidence in original['external_evidence'].items():
        verify(anchors,name,evidence['sha256'])
        assert (anchors/name).stat().st_size==evidence['bytes']
    for name,expected in summary['inputs_sha256'].items():
        verify(artifact.resolve(),name,expected)
    assert summary['summary_source_sha256']==EXPECTED['scripts/summarize_trained_symmetry_pilot.py']
    assert summary['saved_summaries']==evaluation['summaries']
    assert len(evaluation['rows'])==len(summary['all_assigned_usefulness_margins'])==72
    rows={(r['pair_id'],r['arm']):r for r in evaluation['rows']}
    for margin in summary['all_assigned_usefulness_margins']:
        row=rows[(margin['pair_id'],margin['arm'])]
        for key,value in margin.items():
            if key=='target_gain_margin':assert value==row['target_gain']-.05
            elif key=='protected_loss_headroom':assert value==.02-row['protected_accuracy_loss']
            else:assert value==row[key]
    assert summary['totals']['useful_admitted']==sum(r['useful_admitted'] for r in rows.values())==0
    assert summary['totals']['useful_sampled']==sum(r['useful_sampled'] for r in rows.values())==5
    assert summary['totals']['valid_literal_native']==sum(r['valid_literal_native'] for r in rows.values())==33
    assert summary['selected_artifact_bytes']==sum((artifact/name).stat().st_size for name in seal['files'])==416169359
    criterion=read(artifact/'criterion-assessment.json')
    issue=read(public/HISTORY/'issue-transfer.json')
    checklist=read(public/HISTORY/'remaining-study-checklist.json')
    study=next(x for x in checklist['studies'] if x['issue']==16)
    for name,expected in criterion['original_sources'].items():verify(public,name,expected)
    assert criterion['original_shared_checks']==checklist['common_checks']
    assert criterion['original_issue16_design_checklist']==study
    assert not criterion['issue_complete'] and not criterion['native_replay_verified']
    assert not criterion['final_claim_dependencies_resolved_by_pilot']
    entries={e['id']:e for e in criterion['entries']}
    assert len(entries)==len(criterion['entries'])==45
    for key in ('deliverables','acceptance'):
        prefix='issue-deliverable-' if key=='deliverables' else 'issue-acceptance-'
        for i,value in enumerate(issue[key],1):assert entries[prefix+str(i)]['original_requirement']==value
    for item in checklist['common_checks']:assert entries['shared-'+item['id']]['original_requirement']==item['requirement']
    for item in study['pending_checks']:assert entries['study-'+item['id']]['original_requirement']==item['requirement']
    for item in study['documented_acceptance']:assert entries['design-acceptance-'+item]['original_requirement']==item
    for name,value in study['proposed_gates'].items():assert entries['confirmation-'+name]['original_requirement']=={'name':name,'value':value}
    new_names=['docs/research/trained-symmetry-pilot.md','docs/research/trained-symmetry-pilot-criterion-map.md',
        ARTIFACT+'/criterion-assessment.json','scripts/authenticate_trained_symmetry_documentation.py']
    new_hashes={name:sha(public/name) for name in new_names}
    external_names=('freeze-anchor.json','selection-anchor.json','posthoc-normalized-interface-launch.json',
        'posthoc-normalized-interface-supervisor-stdout.log','posthoc-normalized-interface-supervisor-stderr.log',
        'posthoc-normalized-interface-verification.log','supplemental-control-replay.log',
        'supplemental-control-replay-attempt-2.log')
    external_hashes={name:{'sha256':sha(anchors/name),'bytes':(anchors/name).stat().st_size} for name in external_names}
    receipt={'schema':'wsp-trained-symmetry-pilot-supplemental-documentation-authentication/1','pass':True,
        'authenticated_at_utc':datetime.now(timezone.utc).isoformat(),'authentication_source_sha256':sha(Path(__file__)),
        'scope':'Separate saved-JSON/byte authentication of final documentation, original unchanged audit and failures, normalized-interface derivative evidence, external anchors/logs, frozen sources/sealed selection, raw acquisition bytes and historical criterion sources. No raw observation parsing or native/admission/metric recomputation.',
        'pinned_public_files_sha256':pinned,'new_documentation_files_sha256':new_hashes,
        'documentation_builder_sha256':sha(documentation_source),'external_evidence':external_hashes,
        'sealed_file_count':542,'frozen_source_file_count':31,'raw_byte_files':raw_checks,
        'canonical_evaluation_sha256':digest(evaluation),'criterion_count':45,'original_criteria_transcribed_exactly':True,
        'saved_intervals_copied_verbatim':True,'saved_margins_arithmetic_authenticated':True,
        'original_audit_preserved':True,'original_native_replay_verified':False,
        'original_completed_admission_replays':0,'derivative_admissions_recomputed_before_mismatch':1,
        'derivative_canonical_comparisons_passed':0,'completed_pair_replays':0,'native_replay_verified':False,
        'success_receipt_absence':success_absence,'literal_control_reconstruction_verified':True,
        'issue16_complete':False,
        'limitations':['No successful independent native admission/metric replay',
            'Exact normalized derivative mismatch cause unknown; recomputed fields not retained',
            'Held-out known edited native score controls pass only 36/48',
            'Learned lanes and causal intervention evidence missing; confirmation proposal unexecuted',
            'Authentication does not attest private/OS hermeticity or complete workflow efficiency']}
    output.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'path':str(output),'sha256':sha(output),'pass':True,'sealed_files':542,'source_files':31,
        'criteria':45,'native_replay_verified':False,'issue16_complete':False},sort_keys=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-root',type=Path,required=True)
    parser.add_argument('--anchors',type=Path,required=True)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--documentation-source',type=Path,required=True)
    args=parser.parse_args()
    authenticate(args.public_root,args.anchors,args.raw_root,args.documentation_source)
