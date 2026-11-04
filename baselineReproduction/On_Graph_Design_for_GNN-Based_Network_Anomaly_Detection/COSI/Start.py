"""CLI entrypoint for training the supplied-graph COSI/GDN implementation.

Scientific caution: the graph must be built separately using the original
COSI notebooks and appropriately sourced data to reproduce the paper.
"""
import argparse
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_io import load_adjacency, load_csv_pair
from util.env import set_device
from util.preprocess import construct_data
from util.TimeDataset import TimeDataset
from util.evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores


def build_parser():
    parser = argparse.ArgumentParser(description='Run supplied-adjacency COSI/GDN with separate normal training and labeled test data.')
    parser.add_argument('--train', required=True, help='Normal-only train CSV (columns = KPI names)')
    parser.add_argument('--test', required=True, help='Test CSV, same KPIs plus an attack column')
    parser.add_argument('--adjacency', required=True, help='Headerless NxN numeric CSV, ordered like training KPI columns')
    parser.add_argument('--edge-policy', choices=('directed', 'upper'), default='directed', help='Directed=all nonzero off-diagonal edges; upper=legacy upper triangle only')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--slide-win', type=int, default=30)
    parser.add_argument('--slide-stride', type=int, default=5)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--topk', type=int, default=20, help='Kept for config compatibility; supplied-graph model does not use learned top-K edges')
    parser.add_argument('--val-ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', default='pretrained/cosi_runs')
    parser.add_argument('--report', choices=('val', 'best'), default='val', help='BEST uses test labels to optimize threshold: diagnostic upper bound, not valid held-out metric')
    parser.add_argument('--load-model', help='Optional checkpoint path; skip model training')
    return parser


def main():
    args = build_parser().parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA was requested but PyTorch cannot see the GPU. Use --device cpu or follow TROUBLESHOOTING.md.')
    if min(args.epochs, args.batch, args.slide_win, args.slide_stride, args.dim) < 1:
        raise SystemExit('epochs, batch, slide-win, slide-stride and dim must be positive')
    if not 0 < args.val_ratio < 0.5:
        raise SystemExit('--val-ratio must be between 0 and 0.5')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_device(args.device)

    try:
        train_df, test_df, test_labels = load_csv_pair(args.train, args.test)
        edge_array = load_adjacency(args.adjacency, list(train_df.columns), args.edge_policy)
    except ValueError as exc:
        raise SystemExit(f'INPUT ERROR: {exc}') from exc
    print(f'[DATA] normal train: {train_df.shape}, test: {test_df.shape}, adjacency edges: {edge_array.shape[1]}')
    feature_names = list(train_df.columns)
    edge_index = torch.from_numpy(edge_array).long()
    cfg = {'slide_win': args.slide_win, 'slide_stride': args.slide_stride}
    if len(train_df) <= args.slide_win * 3 or len(test_df) <= args.slide_win:
        raise SystemExit('Not enough rows for windows and a leak-safe validation split. Use longer train and test CSVs.')
    train_ds = TimeDataset(construct_data(train_df, feature_names, labels=0), edge_index, mode='train', config=cfg)
    # Test labels are never used to optimize model weights.
    test_ds = TimeDataset(construct_data(test_df, feature_names, labels=test_labels.tolist()), edge_index, mode='test', config=cfg)
    n = len(train_ds)
    cutoff = int(n * (1 - args.val_ratio))
    gap = int(np.ceil(args.slide_win / args.slide_stride))
    if cutoff < 1 or cutoff + gap >= n:
        raise SystemExit('Too few training windows for chronological validation with an overlap gap. Lower --slide-win or collect more data.')
    train_loader = DataLoader(Subset(train_ds, range(cutoff)), batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(train_ds, range(cutoff + gap, n)), batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    # Delay PyG-dependent imports until after input validation and --help.
    from models.GDN import GDN
    from test import test as evaluate_model
    from util.train import train as train_model
    model = GDN([edge_index], len(feature_names), dim=args.dim, input_dim=args.slide_win,
                out_layer_num=1, out_layer_inter_dim=256, topk=args.topk).to(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    checkpoint = Path(args.load_model) if args.load_model else output_dir / f'cosi_{run_id}.pt'
    print(f'[MODEL] Device: {args.device}; supplied-edge mode; checkpoint: {checkpoint}')
    if not args.load_model:
        train_model(model, str(checkpoint), config={'seed': args.seed, 'batch': args.batch, 'epoch': args.epochs, 'decay': 0},
                    train_dataloader=train_loader, val_dataloader=val_loader, feature_map=feature_names,
                    test_dataloader=test_loader, test_dataset=test_ds, train_dataset=train_ds, dataset_name='COSI')
    if not checkpoint.is_file():
        raise SystemExit(f'Model checkpoint not found: {checkpoint}')
    model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
    model.eval()
    _, test_result = evaluate_model(model, test_loader)
    _, val_result = evaluate_model(model, val_loader)
    score_test, score_val = get_full_err_scores(test_result, val_result)
    labels = np.asarray(test_result[2])[:, 0].astype(int).tolist()
    if args.report == 'best':
        print('WARNING: BEST selects threshold using TEST labels; not a valid held-out estimate!')
        info = get_best_performance_data(score_test, labels, topk=1)
    else:
        info = get_val_performance_data(score_test, score_val, labels, topk=1)
    print(f'[RESULT] report={args.report} F1={info[0]:.5f} precision={info[1]:.5f} recall={info[2]:.5f}')
    print('[CAUTION] Metrics are meaningful as paper reproduction only with author-consistent data, graphs, settings, and splits.')


if __name__ == '__main__':
    main()
