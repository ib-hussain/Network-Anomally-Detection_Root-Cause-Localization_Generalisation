"""Input validation for the COSI/GDN reproduction. No model assumptions are changed here."""
from pathlib import Path

import numpy as np
import pandas as pd


def load_csv_pair(train_path, test_path):
    """Return (train features, test features, test binary labels)."""
    for path in (train_path, test_path):
        if not Path(path).is_file():
            raise ValueError(f"Missing CSV file: {path}. See setup/02_COSI.md")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    # Drop only known index columns; never guess by deleting the first KPI.
    drop_cols = lambda frame: [col for col in frame.columns if str(col).startswith('Unnamed:') or str(col).lower() in ('timestamp', 'time')]
    train = train.drop(columns=drop_cols(train), errors='ignore')
    test = test.drop(columns=drop_cols(test), errors='ignore')
    if 'attack' not in test.columns:
        raise ValueError("Test CSV requires binary 'attack' column (0=normal,1=anomaly).")
    if 'attack' in train.columns:
        if not (pd.to_numeric(train['attack'], errors='coerce') == 0).all():
            raise ValueError('Training data contains anomalies. Use normal-only training samples.')
        train = train.drop(columns=['attack'])
    labels = pd.to_numeric(test.pop('attack'), errors='coerce')
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError('Test attack labels must be 0 or 1, with no missing values.')
    if labels.nunique() != 2:
        raise ValueError('Test attack labels must contain both normal (0) and anomaly (1) to compute F1/AUC.')
    if train.columns.duplicated().any() or test.columns.duplicated().any():
        raise ValueError('Repeated feature names are not allowed.')
    if set(train.columns) != set(test.columns):
        raise ValueError('Train/test feature names differ. Make preprocessing consistent.')
    # Exact train-column order is essential: adjacency rows/columns use this order.
    test = test.reindex(columns=train.columns)
    if train.empty or train.shape[1] < 2 or test.empty:
        raise ValueError('Train/test data must be nonempty and have at least 2 KPIs.')
    try:
        train = train.astype('float32')
        test = test.astype('float32')
    except (TypeError, ValueError) as exc:
        raise ValueError('Non-numeric KPI values present; preprocess CSV without changing labels.') from exc
    if not np.isfinite(train.to_numpy()).all() or not np.isfinite(test.to_numpy()).all():
        raise ValueError('NaN or infinite values found. Clean inputs explicitly before training.')
    return train, test, labels.astype('int64').to_numpy()


def load_adjacency(path, features, edge_policy='directed'):
    """Read an NxN headerless numeric matrix. Set policy='upper' for legacy conversion."""
    if not Path(path).is_file():
        raise ValueError(f'Missing adjacency CSV: {path}. Generate it in COSI notebooks.')
    matrix = np.genfromtxt(path, delimiter=',')
    n = len(features)
    if matrix.shape != (n, n):
        raise ValueError(f'Adjacency shape {matrix.shape} but {n} features found; expected ({n},{n}). Use headerless CSV, ordered like training KPI columns.')
    if not np.isfinite(matrix).all():
        raise ValueError('Adjacency contains NaN/infinity. Remove CSV header/row labels.')
    if edge_policy == 'upper':
        src, dst = np.where(np.triu(matrix, 1) > 0)
    elif edge_policy == 'directed':
        src, dst = np.where((matrix > 0) & ~np.eye(n, dtype=bool))
    else:
        raise ValueError('edge_policy must be directed or upper')
    if len(src) == 0:
        raise ValueError('Adjacency has no positive off-diagonal edges. Check threshold or files.')
    return np.stack((src, dst)).astype('int64')
