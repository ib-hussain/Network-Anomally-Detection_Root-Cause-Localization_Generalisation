"""Create NON-COSI integration-test inputs from the bundled MSL demo.

This makes a Pearson correlation graph, NOT the paper's semantic COSI graph.
Do not cite this as reproduction or publish its scores as COSI results.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='smoke_data')
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    gdn = root.parent / 'GDN' / 'data' / 'msl'
    train = pd.read_csv(gdn / 'train.csv')
    test = pd.read_csv(gdn / 'test.csv')
    features = [c for c in train.columns if c not in ('timestamp', 'attack')]
    if set(features) != set(test.columns) - {'timestamp', 'attack'}:
        raise SystemExit('MSL columns are inconsistent.')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train[features].to_csv(out/'train.csv', index=False)
    test[features + ['attack']].to_csv(out/'test.csv', index=False)
    corr = np.nan_to_num(train[features].corr().to_numpy(), nan=0.0)
    adj = (np.abs(corr) > 0.1).astype(int)
    np.fill_diagonal(adj, 0)
    # At least a sparse connected integration-test graph. NOT COSI.
    for i in range(len(features) - 1):
        adj[i, i+1] = adj[i+1, i] = 1
    np.savetxt(out/'adjacency.csv', adj, delimiter=',', fmt='%d')
    print(f'NON-PAPER SMOKE DATA: {out.resolve()}')
    print('Never report MSL/Pearson graph metrics as COSI benchmark results.')


if __name__ == '__main__':
    main()
