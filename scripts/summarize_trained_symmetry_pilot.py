"""Post-hoc descriptive extraction from saved pilot JSON; no native/data evaluation."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

ARTIFACT = 'artifacts/research/trained-symmetry-pilot'
ARMS = ('none', 'raw', 'heuristic', 'wrong', 'matched_random', 'fixed_target_fit')

def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def agreement_summary(records):
    return {'count': len(records), 'passed': sum(r['pass'] for r in records),
        'maximum_absolute_score_discrepancy': max(r['maximum_absolute_score_discrepancy'] for r in records),
        **{k:sum(r[k] for r in records) for k in ('decision_mismatches', 'nonfinite_values', 'near_ties_at_1e-5')}}

def summarize(public):
    artifact = public/ARTIFACT
    evaluation = read(artifact/'evaluation.json')
    selections = [read(path) for path in sorted((artifact/'pairs').glob('*/selection.json'))]
    lineages = [read(path) for path in sorted((artifact/'pairs').glob('*/*-lineage.json.gz'))]
    rows = evaluation['rows']
    assert len(selections) == 12 and len(lineages) == 24 and len(rows) == 72
    assert len({(r['pair_id'], r['arm']) for r in rows}) == 72
    margins = [{k:r[k] for k in ('pair_id','domain','variant','arm','target_gain','protected_accuracy_loss',
        'valid_literal_native','integrated_admission_accepted','useful_sampled','useful_admitted')} | {
        'target_gain_margin':r['target_gain']-.05,
        'protected_loss_headroom':.02-r['protected_accuracy_loss']} for r in rows]
    variants = []
    for domain,variant in sorted({(r['domain'],r['variant']) for r in rows}):
        selected = [r for r in rows if r['domain'] == domain and r['variant'] == variant]
        arms = {}
        for arm in ARMS:
            chosen = [r for r in selected if r['arm'] == arm]
            assert len(chosen) == 2
            arms[arm] = {'assigned_count':2, 'observed_count':2, 'missing_count':0,
                **{k:sum(r[k] for r in chosen)/2 for k in ('target_gain','protected_accuracy_loss')},
                **{k:sum(r[k] for r in chosen) for k in ('useful_sampled','useful_admitted','valid_literal_native')},
                'maximum_subgroup_accuracy_loss':max(r['maximum_subgroup_accuracy_loss'] for r in chosen)}
        variants.append({'domain':domain,'variant':variant,'arms':arms,
            'scope':'Post-hoc two-pair descriptive means; no resampled intervals or confirmation tests'})
    domains = {}
    for domain in ('banking77','optical-digits'):
        selected = [r for r in rows if r['domain'] == domain]
        selection = [s for s in selections if s['pair']['domain'] == domain]
        known = [r for r in evaluation['known_controls'] if r['domain'] == domain]
        native_controls = {}
        for gauge in ('identity','permutation','dyadic','combined'):
            native_controls[gauge] = {kind:agreement_summary([r[kind+'_agreement'] for r in known if r['gauge'] == gauge])
                for kind in ('parent','edited','recovered','raw_edited','wrong_edited')}
        arms = {}
        for arm in ARMS:
            chosen = [r for r in selected if r['arm'] == arm]
            m = [r for r in margins if r['domain'] == domain and r['arm'] == arm]
            subgroup_max = max(sg['accuracy_loss'] for r in chosen for sg in r['subgroups'])
            arms[arm] = {'assigned_count':6,
                **{k:sum(r[k] for r in chosen) for k in ('useful_sampled','useful_admitted','valid_literal_native','integrated_admission_accepted')},
                'target_gain_margin_range':[min(r['target_gain_margin'] for r in m),max(r['target_gain_margin'] for r in m)],
                'protected_loss_headroom_range':[min(r['protected_loss_headroom'] for r in m),max(r['protected_loss_headroom'] for r in m)],
                'metric_thresholds_met':sum(r['target_gain'] >= .05 and r['protected_accuracy_loss'] <= .02 for r in chosen),
                'maximum_subgroup_accuracy_loss':subgroup_max,
                'maximum_subgroup_occurrences':[{'pair_id':r['pair_id'], **sg} for r in chosen for sg in r['subgroups'] if sg['accuracy_loss'] == subgroup_max],
                'maximum_protected_score_drift':max(r['maximum_protected_score_drift'] for r in chosen),
                'heldout_native_binary64':agreement_summary([r['heldout_native_binary64'] for r in chosen])}
        random = [(v['relative_l2_mismatch'],s['pair']['pair_id'],key) for s in selection for key,v in s['random_matching'].items()]
        random_max = max(random)
        domains[domain] = {'role_counts':read(artifact/'preparation'/f'{domain}.json')['role_counts'],
            'arms':arms,'heldout_known_native_controls':native_controls,
            'development_known_native_controls':{kind:agreement_summary([r['controls'][kind] for s in selection for r in s['known_controls']])
                for kind in ('parent','edited','recovered','raw','wrong')},
            'alignment_diagnostic_ranges':{k:[min(s['alignment_diagnostics'][k] for s in selection),max(s['alignment_diagnostics'][k] for s in selection)]
                for k in ('mean_matching_cost','relative_parameter_residual')},
            'random_all_tensor_counts_matched':all(v['count_matched'] for s in selection for v in s['random_matching'].values()),
            'random_maximum_relative_tensor_l2_mismatch':{'value':random_max[0],'pair_id':random_max[1],'tensor':random_max[2]},
            'admission_gate_counts':dict(Counter(k+'/'+v for s in selection for arm in s['arms'].values() for k,v in arm['admission_gate_statuses'].items()))}
    seal = read(artifact/'global-selection-seal.json')
    summary = {'schema':'wsp-trained-symmetry-pilot-saved-artifact-summary/1',
        'created_at_utc':datetime.now(timezone.utc).isoformat(), 'summary_source_sha256':sha(Path(__file__)),
        'scope':'Post-hoc saved-JSON extraction and arithmetic only. Frozen bootstrap intervals are copied verbatim; no raw parse, training, fitting, alignment, selection, admission or native evaluation.',
        'inputs_sha256':{name:sha(artifact/name) for name in ('evaluation.json','global-selection-seal.json','evaluation-accounting.json','stdlib-audit.json')},
        'saved_summaries':evaluation['summaries'], 'official_test_overlap_audit':evaluation['official_test_overlap_audit'],
        'domains':domains,'variant_slices':variants,'all_assigned_usefulness_margins':margins,
        'totals':{'assigned_pairs':12,'retained_lineages':24,'assigned_arms':72,'assigned_known_controls':48,
            'assignment_failures':0,'missing_outcomes':0,
            **{k:sum(r[k] for r in rows) for k in ('useful_sampled','useful_admitted','valid_literal_native','integrated_admission_accepted')}},
        'lineage_native_bridge':agreement_summary([l['native_bridge'] for l in lineages]),
        'lineage_binary64_bridge':agreement_summary([l['binary64_bridge'] for l in lineages]),
        'lineage_diagnostics':{k:sum(l[k] for l in lineages) for k in ('zero_incoming_affine_rows','duplicate_incoming_affine_rows')},
        'training_ledger_totals':{k:sum(l[k] for l in lineages) for k in ('elapsed_seconds','training_batches','gradient_backward_batches','training_forward_rows','development_forward_rows','selected_feature_forward_rows')},
        'selection_ledger':{'elapsed_seconds':seal['selection_seconds'],**seal['native_accounting']},
        'evaluation_ledger':read(artifact/'evaluation-accounting.json'),
        'selected_artifact_bytes':sum((artifact/name).stat().st_size for name in seal['files']),
        'native_replay_verified':False}
    path = artifact/'saved-artifact-summary.json'
    assert not path.exists(), 'Do not replace a previous summary'
    path.write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'path':str(path),'sha256':sha(path),'totals':summary['totals'],'training':summary['training_ledger_totals']},sort_keys=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-root',type=Path,required=True)
    args = parser.parse_args()
    summarize(args.public_root.resolve())
