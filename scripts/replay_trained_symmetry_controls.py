"""Supplement frozen metric replay with full donor/known-program reconstruction."""
from __future__ import annotations
import argparse
from pathlib import Path
import runpy

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--public-root',type=Path,required=True)
parser.add_argument('--anchors',type=Path,required=True)
args=parser.parse_args()
public,anchors=args.public_root.resolve(),args.anchors.resolve()
script=public/'experiments/research/trained-symmetry-pilot/source-freeze/scripts/run_trained_symmetry_pilot.py'
f=runpy.run_path(str(script),run_name='frozen_pilot_control_replay')
protocol=f['check_freeze'](public,anchors)
f['check_seal'](public,anchors)
artifact=public/f['ARTIFACT']
started=f['time'].perf_counter()
checks=[]
for assignment in protocol['assignments']:
    directory=artifact/'pairs'/assignment['pair_id']
    if not (directory/'selection.json').exists():
        continue
    selection=f['read'](directory/'selection.json')
    donor=f['load_model'](directory/'donor.npz')
    donor_candidate=f['load_model'](directory/'donor-candidate.npz')
    donor_program=f['Program'].from_dict(f['read'](directory/'donor-program.json'))
    assert f['apply_program'](donor,donor_program).sha256==donor_candidate.sha256
    assert f['apply_program'](donor_candidate,f['rollback'](donor,donor_program)).sha256==donor.sha256
    for role in ('donor','recipient'):
        lineage=f['read'](directory/f'{role}-lineage.json.gz')
        assert f['load_model'](directory/f'{role}.npz').sha256==lineage['model_sha256']
        assert lineage['head']['metadata']['lineage_seed']==assignment[role+'_seed']
    alignment=f['Gauge'](tuple(selection['alignment']['permutation']),tuple(selection['alignment']['scales']))
    wrong=f['Gauge'](tuple(map(int,f['np'].roll(f['np'].array(alignment.permutation),1))),alignment.scales)
    for control in selection['known_controls']:
        kind=control['gauge']
        action=f['Gauge'](tuple(control['action']['permutation']),tuple(control['action']['scales']))
        parent,program=f['transport'](donor,donor_program,action)
        candidate=f['apply_program'](parent,program)
        assert parent.sha256==f['load_model'](directory/'known'/f'{kind}-parent.npz').sha256
        assert candidate.sha256==f['load_model'](directory/'known'/f'{kind}-candidate.npz').sha256
        assert f['apply_program'](candidate,f['rollback'](parent,program)).sha256==parent.sha256
        recovered,diagnostics=f['align_hidden'](donor,parent)
        assert recovered.to_dict()==control['recovered_action']
        assert diagnostics==control['recovery_diagnostics']
        recovered_candidate=f['apply_program'](parent,f['transfer_edit'](donor,donor_program,parent,recovered))
        raw,_=f['realize'](parent,f['transfer_edit'](donor,donor_program,parent))
        incorrect,_=f['realize'](parent,f['transfer_edit'](donor,donor_program,parent,wrong))
        for name,model in (('recovered',recovered_candidate),('raw',raw),('wrong',incorrect)):
            assert model.sha256==f['load_model'](directory/'known'/f'{kind}-{name}.npz').sha256
        checks.append({'pair_id':assignment['pair_id'],'gauge':kind,'parent_sha256':parent.sha256,
            'candidate_sha256':candidate.sha256,'program_sha256':program.sha256,'transported_program':program.to_dict(),
            'donor_endpoint_rollback':True,'known_endpoint_rollback':True,'recovery':True,'raw_wrong_endpoints':True})
    print('control reconstruction '+assignment['pair_id'],flush=True)
path=artifact/'supplemental-control-replay.json'
assert not path.exists()
f['write'](path,{'schema':'wsp-trained-symmetry-pilot-control-replay/1','pass':True,
    'replay_source_sha256':f['sha'](Path(__file__)),'frozen_source_sha256':f['sha'](script),
    'elapsed_seconds':f['time'].perf_counter()-started,
    'seal_sha256':f['sha'](artifact/'global-selection-seal.json'),'checks':checks,
    'scope':'Fresh process reconstructs donor and known transported/recovered/raw/wrong literal endpoints and rollback from frozen helper bytes; no data parse or candidate selection'})
print('supplemental control replay passed',flush=True)
