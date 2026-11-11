"""Tests for the read-only Wireless input audit; uses artificial fixtures only."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_tthp_inputs import audit_inputs, AuditError


class TestInputAudit(unittest.TestCase):
    def fixture(self, root):
        raw = root / 'data/raw/wireless/18V_55N_Wireless'
        processed = root / 'data/processed'
        raw.mkdir(parents=True, exist_ok=True)
        processed.mkdir(parents=True, exist_ok=True)
        types = [i % 18 for i in range(55)]
        devices = list(range(55))
        pd.DataFrame({
            'alarm_id': types,
            'device_id': devices,
            'start_timestamp': list(range(55)),
            'end_timestamp': [i + 1 for i in range(55)],
        }).to_csv(raw / 'Alarm.csv', index=False)
        dag = np.zeros((18, 18), dtype=int)
        dag[0, 1] = 1
        np.save(raw / 'DAG.npy', dag)
        pd.DataFrame({
            'event': types,
            'timestamp': list(range(55)),
            'node': devices,
        }).to_csv(processed / 'wireless_thp_events.csv', index=False)
        pd.DataFrame({
            'node_index': devices,
            'original_device_id': devices,
        }).to_csv(processed / 'wireless_device_mapping.csv', index=False)

    def test_valid_inputs_remain_blocked_for_reproduction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            status = audit_inputs(root)
            self.assertTrue(status['preprocessed_events_match_raw'])
            self.assertEqual(status['physical_topology'], 'missing')
            self.assertFalse(status['can_claim_paper_tthp_reproduction'])

    def test_tampering_preprocessing_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            path = root / 'data/processed/wireless_thp_events.csv'
            table = pd.read_csv(path)
            table.loc[0, 'timestamp'] += 99
            table.to_csv(path, index=False)
            with self.assertRaisesRegex(AuditError, 'do not preserve'):
                audit_inputs(root)

    def test_ground_truth_dag_cannot_masquerade_as_topology(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            dag = root / 'data/raw/wireless/18V_55N_Wireless/DAG.npy'
            with self.assertRaisesRegex(AuditError, '55x55'):
                audit_inputs(root, dag)


if __name__ == '__main__':
    unittest.main()
