"""Read-only input audit for Zhou et al. (2026), TTHP (NOT a trainer).

Run from baselineReproduction/cloud_rca_reproduction using:
    python -m scripts.audit_tthp_inputs

This checks alignment, never substitutes DAG.npy for network topology, and
never reports the paper reproduction as ready based on Wireless alone.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


class AuditError(ValueError):
    """An input has failed a structural or provenance check."""


def audit_inputs(root, physical_topology=None):
    """Audit supplied Wireless files; return a JSON-serializable summary."""
    root = Path(root)
    raw_path = root / 'data/raw/wireless/18V_55N_Wireless/Alarm.csv'
    dag_path = root / 'data/raw/wireless/18V_55N_Wireless/DAG.npy'
    events_path = root / 'data/processed/wireless_thp_events.csv'
    mapping_path = root / 'data/processed/wireless_device_mapping.csv'
    for path in (raw_path, dag_path, events_path, mapping_path):
        if not path.is_file():
            raise AuditError('Missing required input: {}'.format(path))

    raw = pd.read_csv(raw_path)
    events = pd.read_csv(events_path)
    mapping = pd.read_csv(mapping_path)
    dag = np.load(dag_path, allow_pickle=False)
    raw_cols = {'alarm_id', 'device_id', 'start_timestamp', 'end_timestamp'}
    event_cols = {'event', 'timestamp', 'node'}
    map_cols = {'node_index', 'original_device_id'}
    if not raw_cols.issubset(raw.columns) or not event_cols.issubset(events.columns):
        raise AuditError('Alarm.csv or processed file is missing required columns')
    if not map_cols.issubset(mapping.columns):
        raise AuditError('Device mapping is missing required columns')
    if raw.empty or len(events) != len(raw):
        raise AuditError('Raw and processed event counts are empty or differ')
    if raw[list(raw_cols)].isna().any().any() or events[list(event_cols)].isna().any().any():
        raise AuditError('Raw or processed events contain missing values')
    if dag.ndim != 2 or dag.shape[0] != dag.shape[1] or not np.isfinite(dag).all():
        raise AuditError('Ground-truth event DAG must be finite and square')

    kinds = sorted(int(x) for x in raw.alarm_id.unique())
    if kinds != list(range(dag.shape[0])):
        raise AuditError('Raw alarm_id values do not align with DAG row/column indices')
    devices = sorted(int(x) for x in raw.device_id.unique())
    if mapping.original_device_id.duplicated().any() or mapping.node_index.duplicated().any():
        raise AuditError('Device mapping contains duplicates')
    if mapping.original_device_id.tolist() != devices or mapping.node_index.tolist() != list(range(len(devices))):
        raise AuditError('Device mapping must preserve sorted original IDs to 0..N-1')
    if sorted(int(x) for x in events.event.unique()) != kinds:
        raise AuditError('Processed event types differ from raw event types')
    if sorted(int(x) for x in events.node.unique()) != list(range(len(devices))):
        raise AuditError('Processed device IDs are not dense and aligned')

    lookup = dict(zip(mapping.original_device_id.tolist(), mapping.node_index.tolist()))
    expected = pd.DataFrame({
        'event': raw['alarm_id'].to_numpy(),
        'timestamp': raw['start_timestamp'].to_numpy(dtype=float),
        'node': raw['device_id'].map(lookup).to_numpy(),
    })
    cols = ['event', 'timestamp', 'node']
    expected = expected[cols].sort_values(cols).reset_index(drop=True)
    observed = events[cols].sort_values(cols).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(expected.astype(float), observed.astype(float), check_exact=True)
    except AssertionError as exc:
        raise AuditError('Processed events do not preserve raw (type,start_time,device) records') from exc

    topology_status = 'missing'
    topology_edges = None
    if physical_topology is not None:
        path = Path(physical_topology)
        if not path.is_file():
            raise AuditError('Specified physical topology file does not exist: {}'.format(path))
        if path.suffix.lower() == '.npy':
            topo = np.load(path, allow_pickle=False)
        elif path.suffix.lower() == '.csv':
            topo = np.loadtxt(path, delimiter=',')
        else:
            raise AuditError('Physical topology must be .npy or headerless .csv')
        if topo.shape != (len(devices), len(devices)) or not np.isfinite(topo).all():
            raise AuditError('Physical topology must be finite {}x{} in mapped node order'.format(len(devices), len(devices)))
        if (topo < 0).any():
            raise AuditError('Physical topology adjacency weights cannot be negative')
        topology_status = 'structure-check-passed; provenance-and-node-order-not-confirmed'
        topology_edges = int(np.count_nonzero(topo))

    warnings = []
    if len(raw) != 34839:
        warnings.append('Paper §5.1.1 reports 34,839 records; local file has {:,}. Investigate provenance; never add fake rows.'.format(len(raw)))
    warnings.append('DAG.npy is an {}x{} event-type ground-truth graph, NOT the physical {}-node topology.'.format(dag.shape[0], dag.shape[1], len(devices)))
    warnings.append('Exact delay, time binning, EM equations, physical topology and original splits remain unverified.')
    return {
        'paper': 'Zhou et al. (2026), EAAI 174:114451',
        'status': 'INPUTS_AUDITED_NOT_PAPER_READY',
        'events': int(len(raw)),
        'alarm_types': len(kinds),
        'devices': len(devices),
        'event_dag_shape': list(dag.shape),
        'event_dag_nonzero_entries': int(np.count_nonzero(dag)),
        'preprocessed_events_match_raw': True,
        'physical_topology': topology_status,
        'physical_topology_nonzero_entries': topology_edges,
        'can_claim_paper_tthp_reproduction': False,
        'can_claim_paper_rca_reproduction': False,
        'warnings': warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.', help='cloud_rca_reproduction directory; default: current directory')
    parser.add_argument('--physical-topology', help='OPTIONAL real 55x55 physical adjacency .npy/.csv; never use DAG.npy here')
    args = parser.parse_args()
    try:
        result = audit_inputs(args.root, args.physical_topology)
    except (AuditError, OSError, ValueError) as exc:
        parser.exit(1, 'INPUT AUDIT FAILED: {}\n'.format(exc))
    print(json.dumps(result, indent=2))
    print('NO TRAINING EXECUTED. Missing specifications are listed in setup/STEP22_PAPER_SPEC_AUDIT.md')


if __name__ == '__main__':
    main()
