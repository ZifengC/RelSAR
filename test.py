from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from main import parse_global_args
from models import UniSAR
from utils import const, utils
from utils.Runner import SarRunner

TOPK_EXPORT = 10


def init_setting(data_name: str) -> None:
    if data_name == 'KuaiSAR':
        const.init_setting_KuaiSAR()
    elif data_name == 'Amazon':
        const.init_setting_Amazon()
    else:
        raise ValueError(f'Unsupported data name: {data_name}')


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export PC-SAR intent posteriors')
    parser = parse_global_args(parser)
    parser = UniSAR.parse_model_args(parser)
    parser = SarRunner.parse_runner_args(parser)
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to export.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='intermediate/pcsar_intent_posteriors.csv',
        help='CSV file to write.'
    )
    parser.add_argument(
        '--export_trajectory',
        type=int,
        default=1,
        choices=[0, 1],
        help='Whether to export a user-level trajectory CSV from user_vocab.'
    )
    parser.add_argument(
        '--trajectory_output',
        type=str,
        default='intermediate/pcsar_user_trajectory.csv',
        help='CSV file for the reconstructed user trajectory.'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default='',
        help='Path to best.pt. Falls back to --test_path when empty.'
    )
    return parser.parse_args()


def resolve_ckpt_path(args: argparse.Namespace) -> Path:
    ckpt = args.ckpt or args.test_path
    if not ckpt:
        raise ValueError('Missing checkpoint path. Pass --ckpt or --test_path.')
    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    return ckpt_path


def infer_intent_num(state_dict: dict) -> int:
    for key, value in state_dict.items():
        if key.endswith('intent_discovery.intent_slots'):
            return int(value.shape[0])
    raise KeyError('Could not find intent_discovery.intent_slots in checkpoint.')


def get_split_loaders(runner: SarRunner, split: str):
    if split == 'train':
        return runner.rec_train_loader, runner.src_train_loader, runner.traindata['rec'], runner.traindata['src']
    if split == 'val':
        return runner.rec_val_loader, runner.src_val_loader, runner.valdata['rec'], runner.valdata['src']
    if split == 'test':
        return runner.rec_test_loader, runner.src_test_loader, runner.testdata['rec'], runner.testdata['src']
    raise ValueError(f'Unsupported split: {split}')


def compute_entropy(pi: np.ndarray) -> float:
    pi = np.asarray(pi, dtype=np.float64)
    k = max(int(pi.shape[-1]), 1)
    denom = np.log(k) if k > 1 else 1.0
    return float(-(pi * np.log(np.clip(pi, 1e-12, 1.0))).sum() / denom)


def compute_js_distance(pi_a: np.ndarray, pi_b: np.ndarray) -> float:
    """Return Jensen-Shannon distance without SciPy.

    The value is sqrt(JS divergence) and stays in [0, 1] after normalization.
    """
    a = np.asarray(pi_a, dtype=np.float64)
    b = np.asarray(pi_b, dtype=np.float64)
    a = a / np.clip(a.sum(), 1e-12, None)
    b = b / np.clip(b.sum(), 1e-12, None)
    m = 0.5 * (a + b)
    kl_am = np.sum(a * (np.log(np.clip(a, 1e-12, 1.0)) - np.log(np.clip(m, 1e-12, 1.0))))
    kl_bm = np.sum(b * (np.log(np.clip(b, 1e-12, 1.0)) - np.log(np.clip(m, 1e-12, 1.0))))
    js = 0.5 * (kl_am + kl_bm)
    return float(np.sqrt(max(js, 0.0)))


def extract_state_metrics(pi: np.ndarray) -> dict:
    pi = np.asarray(pi, dtype=np.float64)
    return {
        'dominant_intent': int(np.argmax(pi)),
        'dominant_intent_prob': float(np.max(pi)),
        'intent_entropy': compute_entropy(pi),
        'intent_mass_mean': float(pi.mean()),
    }


def to_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().mean().item())
    return float(np.asarray(value).mean())


def extract_loss_diagnostics(loss_dict: dict) -> dict:
    keys = [
        'intent_assign_entropy',
        'intent_confidence',
        'intent_usage_entropy',
        'intent_usage_max',
        'intent_residual_mean',
        'intent_proto_sim_mean',
        'intent_proto_sim_max',
        'intent_proto_sim_min',
        'intent_proto_margin_violation',
        'belief_uncertainty_mean',
        'belief_entropy_mean',
        'belief_confidence_mean',
        'belief_variance_mean',
        'belief_uncertainty_std',
        'belief_uncertainty_early_mean',
        'belief_uncertainty_mid_mean',
        'belief_uncertainty_late_mean',
        'attention_temp_mean',
        'attention_temp_std',
        'attention_temp_early_mean',
        'attention_temp_mid_mean',
        'attention_temp_late_mean',
        'attention_temp_max',
        'attention_temp_min',
        'belief_mass_mean',
        'belief_mass_max',
        'belief_mass_min',
        'cf_mask_mean',
        'cf_necessity_mean',
        'cf_potential_mean',
        'cf_self_mean',
        'rec_mix_mean',
        'src_mix_mean',
        'rec_same_delta_mean',
        'rec_cross_delta_mean',
        'src_same_delta_mean',
        'src_cross_delta_mean',
        'rec_cross_gate_mean',
        'src_cross_gate_mean',
        'cross_mix_effective_mean',
        'cf_consistency_reg',
    ]
    return {key: to_scalar(loss_dict[key]) for key in keys if key in loss_dict}


def tensor_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def add_vector_fields(row: dict, prefix: str, vector: np.ndarray | torch.Tensor | None) -> None:
    if vector is None:
        return
    arr = tensor_to_numpy(vector).reshape(-1)
    for idx, value in enumerate(arr):
        row[f'{prefix}_{idx}'] = float(value)


def add_matrix_fields(row: dict, prefix: str, matrix: np.ndarray | torch.Tensor | None) -> None:
    if matrix is None:
        return
    arr = tensor_to_numpy(matrix)
    if arr.ndim != 2:
        raise ValueError(f'Expected 2D matrix for {prefix}, got shape {arr.shape}')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            row[f'{prefix}_{i}_{j}'] = float(arr[i, j])


def masked_mean_embeddings(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = (~mask).float().unsqueeze(-1)
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (emb * valid).sum(dim=1) / denom


def summarize_logits(logits: np.ndarray, item_ids: np.ndarray | None = None) -> dict:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f'Expected 2D logits, got shape {logits.shape}')
    pos_scores = logits[:, 0]
    top1_idx = np.argmax(logits, axis=1)
    top1_scores = logits[np.arange(logits.shape[0]), top1_idx]
    pos_rank = (logits > pos_scores[:, None]).sum(axis=1) + 1
    if logits.shape[1] > 1:
        best_neg = np.max(logits[:, 1:], axis=1)
    else:
        best_neg = np.full_like(pos_scores, np.nan)
    summary = {
        'top1_index': top1_idx,
        'top1_score': top1_scores,
        'pos_score': pos_scores,
        'pos_rank': pos_rank,
        'pos_margin': pos_scores - best_neg,
        'top1_is_pos': (top1_idx == 0).astype(int),
    }
    if item_ids is not None:
        item_ids = np.asarray(item_ids)
        summary['top1_item_id'] = item_ids[np.arange(logits.shape[0]), top1_idx]
    return summary


@torch.no_grad()
def export_channel(
    model: UniSAR,
    loader,
    raw_df: pd.DataFrame,
    channel: str,
    split: str,
) -> list[dict]:
    rows = []
    offset = 0

    for batch in loader:
        batch = utils.batch_to_gpu(batch, model.device)

        pos_item = batch['item']
        if pos_item.dim() == 1:
            pos_item = pos_item.unsqueeze(1)
        items = torch.cat([pos_item, batch['neg_items']], dim=1)
        items_emb = model.session_embedding.get_item_emb(items)
        query_emb = None
        if channel == 'S' and 'query' in batch:
            query_emb = model.session_embedding.get_query_emb(batch['query'])

        all_his_emb, all_his_mask, _ = model.get_all_his_emb(
            batch['all_his'],
            batch['all_his_type'],
        )
        rec_his_emb, src_his_emb = model.split_rec_src(all_his_emb, batch['all_his_type'])

        rec_his_mask = torch.masked_select(
            all_his_mask,
            (batch['all_his_type'] == 1),
        ).reshape((all_his_emb.shape[0], const.max_rec_his_len))
        src_his_mask = torch.masked_select(
            all_his_mask,
            (batch['all_his_type'] == 2),
        ).reshape((all_his_emb.shape[0], const.max_src_session_his_len))

        if channel == 'R':
            hist_mask = rec_his_mask
        elif channel == 'S':
            hist_mask = src_his_mask
        else:
            raise ValueError(f'Unknown channel: {channel}')

        global_intents, global_prior_assign, _ = model.compute_intent_state(all_his_emb, all_his_mask)
        rec_intents, rec_prior_assign, _ = model.compute_intent_state(rec_his_emb, rec_his_mask)
        src_intents, src_prior_assign, _ = model.compute_intent_state(src_his_emb, src_his_mask)

        global_posterior, global_uncertainty, _ = model.compute_belief_dynamics(
            all_his_emb,
            global_intents,
            global_prior_assign,
            all_his_mask,
        )
        rec_posterior, rec_uncertainty, _ = model.compute_belief_dynamics(
            rec_his_emb,
            rec_intents,
            rec_prior_assign,
            rec_his_mask,
        )
        src_posterior, src_uncertainty, _ = model.compute_belief_dynamics(
            src_his_emb,
            src_intents,
            src_prior_assign,
            src_his_mask,
        )

        global_posterior = global_posterior.detach().cpu().numpy()
        rec_posterior = rec_posterior.detach().cpu().numpy()
        src_posterior = src_posterior.detach().cpu().numpy()
        global_prior_assign = global_prior_assign.detach().cpu().numpy()
        rec_prior_assign = rec_prior_assign.detach().cpu().numpy()
        src_prior_assign = src_prior_assign.detach().cpu().numpy()
        global_uncertainty = global_uncertainty.detach().cpu().numpy()
        rec_uncertainty = rec_uncertainty.detach().cpu().numpy()
        src_uncertainty = src_uncertainty.detach().cpu().numpy()
        global_history_mean = masked_mean_embeddings(all_his_emb, all_his_mask).detach().cpu().numpy()
        rec_history_mean = masked_mean_embeddings(rec_his_emb, rec_his_mask).detach().cpu().numpy()
        src_history_mean = masked_mean_embeddings(src_his_emb, src_his_mask).detach().cpu().numpy()
        pos_item_emb = items_emb[:, 0, :].detach().cpu().numpy()
        query_emb_np = tensor_to_numpy(query_emb) if query_emb is not None else None
        user_feats, _, _, _ = model.forward(
            batch['user'],
            batch['all_his'],
            batch['all_his_type'],
            items,
            items_emb,
            domain='rec' if channel == 'R' else 'src',
            query_emb=query_emb,
        )
        user_feats_np = [tensor_to_numpy(feat) for feat in user_feats]
        _, _, _, batch_regularization = model.forward(
            batch['user'],
            batch['all_his'],
            batch['all_his_type'],
            items,
            items_emb,
            domain='rec' if channel == 'R' else 'src',
            query_emb=query_emb,
        )
        batch_regularization = {
            key: to_scalar(value) if isinstance(value, torch.Tensor) else float(value)
            for key, value in batch_regularization.items()
        }
        rec_logits = tensor_to_numpy(model.rec_predict(batch))
        src_logits = tensor_to_numpy(model.src_predict(batch)) if 'query' in batch else None
        rec_topk_idx = np.argsort(-rec_logits, axis=1)[:, :min(TOPK_EXPORT, rec_logits.shape[1])]
        rec_items_np = items.detach().cpu().numpy()
        rec_topk_item_ids = np.take_along_axis(rec_items_np, rec_topk_idx, axis=1)
        rec_topk_scores = np.take_along_axis(rec_logits, rec_topk_idx, axis=1)
        rec_topk_item_emb = np.take_along_axis(
            items_emb.detach().cpu().numpy(),
            rec_topk_idx[:, :, None],
            axis=1,
        )
        rec_topk_mean_emb = rec_topk_item_emb.mean(axis=1)
        hist_len = (~hist_mask).sum(dim=1).detach().cpu().numpy().astype(int)
        global_hist_len = (~all_his_mask).sum(dim=1).detach().cpu().numpy().astype(int)
        rec_hist_len = (~rec_his_mask).sum(dim=1).detach().cpu().numpy().astype(int)
        src_hist_len = (~src_his_mask).sum(dim=1).detach().cpu().numpy().astype(int)
        batch_size = int(batch['batch_size'])
        rec_pred_summary = summarize_logits(rec_logits, items.detach().cpu().numpy())
        src_pred_summary = summarize_logits(src_logits, items.detach().cpu().numpy()) if src_logits is not None else None

        for i in range(batch_size):
            raw_row = raw_df.iloc[offset + i]
            valid_len = int(hist_len[i])
            state_idx = max(valid_len - 1, 0)
            global_state_idx = max(int(global_hist_len[i]) - 1, 0)
            rec_state_idx = max(int(rec_hist_len[i]) - 1, 0)
            src_state_idx = max(int(src_hist_len[i]) - 1, 0)

            pi = global_posterior[i, global_state_idx]
            pi_rec = rec_posterior[i, rec_state_idx]
            pi_src = src_posterior[i, src_state_idx]
            prior_pi = global_prior_assign[i, global_state_idx]
            prior_pi_rec = rec_prior_assign[i, rec_state_idx]
            prior_pi_src = src_prior_assign[i, src_state_idx]

            global_metrics = extract_state_metrics(pi)
            rec_metrics = extract_state_metrics(pi_rec)
            src_metrics = extract_state_metrics(pi_src)

            row = {
                'split': split,
                'channel': channel,
                'search': int(channel == 'S'),
                'use_counterfactual': int(bool(model.use_counterfactual)),
                'use_intent_logit_bias': int(bool(model.use_intent_logit_bias)),
                'use_uncertainty_attention': int(bool(model.use_uncertainty_attention)),
                'sample_index': int(offset + i),
                'user_id': int(raw_row['user_id']),
                'intent_state_index': int(state_idx),
                'history_length': int(valid_len),
                'global_history_length': int(global_hist_len[i]),
                'rec_history_length': int(rec_hist_len[i]),
                'src_history_length': int(src_hist_len[i]),
                'history_rec_share': float(rec_hist_len[i] / max(rec_hist_len[i] + src_hist_len[i], 1)),
                'history_src_share': float(src_hist_len[i] / max(rec_hist_len[i] + src_hist_len[i], 1)),
                'global_dominant_intent': global_metrics['dominant_intent'],
                'global_dominant_intent_prob': global_metrics['dominant_intent_prob'],
                'global_intent_entropy': global_metrics['intent_entropy'],
                'global_posterior_uncertainty': float(global_uncertainty[i, global_state_idx]),
                'global_belief_uncertainty_mean': batch_regularization.get('belief_uncertainty_mean', float('nan')),
                'global_belief_confidence_mean': batch_regularization.get('belief_confidence_mean', float('nan')),
                'global_attention_temp_mean': batch_regularization.get('attention_temp_mean', float('nan')),
                'global_belief_entropy_mean': batch_regularization.get('belief_entropy_mean', float('nan')),
                'rec_dominant_intent': rec_metrics['dominant_intent'],
                'rec_dominant_intent_prob': rec_metrics['dominant_intent_prob'],
                'rec_intent_entropy': rec_metrics['intent_entropy'],
                'rec_posterior_uncertainty': float(rec_uncertainty[i, rec_state_idx]),
                'rec_belief_uncertainty_mean': batch_regularization.get('belief_uncertainty_mean', float('nan')),
                'rec_belief_confidence_mean': batch_regularization.get('belief_confidence_mean', float('nan')),
                'rec_attention_temp_mean': batch_regularization.get('attention_temp_mean', float('nan')),
                'rec_belief_entropy_mean': batch_regularization.get('belief_entropy_mean', float('nan')),
                'src_dominant_intent': src_metrics['dominant_intent'],
                'src_dominant_intent_prob': src_metrics['dominant_intent_prob'],
                'src_intent_entropy': src_metrics['intent_entropy'],
                'src_posterior_uncertainty': float(src_uncertainty[i, src_state_idx]),
                'src_belief_uncertainty_mean': batch_regularization.get('belief_uncertainty_mean', float('nan')),
                'src_belief_confidence_mean': batch_regularization.get('belief_confidence_mean', float('nan')),
                'src_attention_temp_mean': batch_regularization.get('attention_temp_mean', float('nan')),
                'src_belief_entropy_mean': batch_regularization.get('belief_entropy_mean', float('nan')),
                'attribution_source_proxy': 'R' if rec_metrics['dominant_intent_prob'] >= src_metrics['dominant_intent_prob'] else 'S',
                'attribution_confidence_gap': float(rec_metrics['dominant_intent_prob'] - src_metrics['dominant_intent_prob']),
                'attribution_entropy_gap': float(src_metrics['intent_entropy'] - rec_metrics['intent_entropy']),
                'rec_src_intent_shift_dot': float(1.0 - np.dot(pi_rec, pi_src)),
                'rec_src_intent_shift_js': compute_js_distance(pi_rec, pi_src),
                'batch_intent_assign_entropy': batch_regularization.get('intent_assign_entropy', float('nan')),
                'batch_intent_confidence': batch_regularization.get('intent_confidence', float('nan')),
                'batch_intent_usage_entropy': batch_regularization.get('intent_usage_entropy', float('nan')),
                'batch_belief_uncertainty_mean': batch_regularization.get('belief_uncertainty_mean', float('nan')),
                'batch_belief_confidence_mean': batch_regularization.get('belief_confidence_mean', float('nan')),
                'batch_belief_entropy_mean': batch_regularization.get('belief_entropy_mean', float('nan')),
                'batch_attention_temp_mean': batch_regularization.get('attention_temp_mean', float('nan')),
                'batch_cf_mask_mean': batch_regularization.get('cf_mask_mean', float('nan')),
                'batch_cf_necessity_mean': batch_regularization.get('cf_necessity_mean', float('nan')),
                'batch_cf_potential_mean': batch_regularization.get('cf_potential_mean', float('nan')),
                'batch_cf_self_mean': batch_regularization.get('cf_self_mean', float('nan')),
            }

            row['candidate_count'] = int(rec_logits.shape[1])
            row['neg_item_count'] = int(rec_logits.shape[1] - 1)
            row['gt_item_id'] = int(raw_row['item_id']) if 'item_id' in raw_df.columns and pd.notna(raw_row['item_id']) else int(raw_row.get('item_id', -1))
            row['rec_pred_top1_index'] = int(rec_pred_summary['top1_index'][i])
            row['rec_pred_top1_score'] = float(rec_pred_summary['top1_score'][i])
            row['rec_pred_pos_score'] = float(rec_pred_summary['pos_score'][i])
            row['rec_pred_pos_rank'] = int(rec_pred_summary['pos_rank'][i])
            row['rec_pred_pos_margin'] = float(rec_pred_summary['pos_margin'][i])
            row['rec_pred_top1_is_pos'] = int(rec_pred_summary['top1_is_pos'][i])
            row['rec_pred_top1_item_id'] = int(rec_pred_summary['top1_item_id'][i]) if 'top1_item_id' in rec_pred_summary else int(items.detach().cpu().numpy()[i, int(rec_pred_summary['top1_index'][i])])
            if src_pred_summary is not None:
                row['src_pred_top1_index'] = int(src_pred_summary['top1_index'][i])
                row['src_pred_top1_score'] = float(src_pred_summary['top1_score'][i])
                row['src_pred_pos_score'] = float(src_pred_summary['pos_score'][i])
                row['src_pred_pos_rank'] = int(src_pred_summary['pos_rank'][i])
                row['src_pred_pos_margin'] = float(src_pred_summary['pos_margin'][i])
                row['src_pred_top1_is_pos'] = int(src_pred_summary['top1_is_pos'][i])
                row['src_pred_top1_item_id'] = int(src_pred_summary['top1_item_id'][i]) if 'top1_item_id' in src_pred_summary else int(items.detach().cpu().numpy()[i, int(src_pred_summary['top1_index'][i])])

            add_vector_fields(row, 'global_history_mean_emb', global_history_mean[i])
            add_vector_fields(row, 'rec_history_mean_emb', rec_history_mean[i])
            add_vector_fields(row, 'src_history_mean_emb', src_history_mean[i])
            add_vector_fields(row, 'pos_item_emb', pos_item_emb[i])
            add_vector_fields(row, 'global_prior_assign', prior_pi)
            add_vector_fields(row, 'rec_prior_assign', prior_pi_rec)
            add_vector_fields(row, 'src_prior_assign', prior_pi_src)
            add_vector_fields(row, 'rec_user_feat', user_feats_np[0][i])
            add_vector_fields(row, 'src_user_feat', user_feats_np[1][i])
            add_vector_fields(row, 'shared_user_feat', user_feats_np[2][i])
            add_vector_fields(row, 'rec_logits', rec_logits[i])
            row['rec_topk_count'] = int(rec_topk_idx.shape[1])
            for k in range(rec_topk_idx.shape[1]):
                row[f'rec_topk_item_id_{k}'] = int(rec_topk_item_ids[i, k])
                row[f'rec_topk_score_{k}'] = float(rec_topk_scores[i, k])
            add_vector_fields(row, 'rec_topk_mean_emb', rec_topk_mean_emb[i])
            if src_logits is not None:
                add_vector_fields(row, 'src_logits', src_logits[i])
            if query_emb_np is not None:
                add_vector_fields(row, 'query_emb', query_emb_np[i])

            for key in ('timestamp', 'item_id', 'keyword', 'search_session_id'):
                if key in raw_df.columns:
                    value = raw_row.get(key)
                    if pd.notna(value):
                        if key in ('item_id', 'search_session_id'):
                            row[key] = int(value)
                        else:
                            row[key] = value

            for k, p in enumerate(pi):
                row[f'global_pi_{k}'] = float(p)
            for k, p in enumerate(pi_rec):
                row[f'rec_pi_{k}'] = float(p)
            for k, p in enumerate(pi_src):
                row[f'src_pi_{k}'] = float(p)

            rows.append(row)

        offset += batch_size

    return rows


def export_user_trajectory(user_vocab: dict, split: str) -> list[dict]:
    rows = []
    for user_id, info in user_vocab.items():
        rec_his = list(info.get('rec_his', []))
        rec_his_ts = list(info.get('rec_his_ts', []))
        src_his = list(info.get('src_session_his', []))
        src_his_ts = list(info.get('src_session_his_ts', []))

        user_rows = []
        for idx, (item_id, ts) in enumerate(zip(rec_his, rec_his_ts)):
            if int(item_id) <= 0:
                continue
            user_rows.append({
                'split': split,
                'user_id': int(user_id),
                'channel': 'R',
                'search': 0,
                'trajectory_source': 'rec_his',
                'trajectory_local_index': int(idx),
                'trajectory_event_id': int(item_id),
                'timestamp': float(ts),
                'item_id': int(item_id),
                'event_type': 'history',
            })

        for idx, (session_id, ts) in enumerate(zip(src_his, src_his_ts)):
            if int(session_id) <= 0:
                continue
            user_rows.append({
                'split': split,
                'user_id': int(user_id),
                'channel': 'S',
                'search': 1,
                'trajectory_source': 'src_session_his',
                'trajectory_local_index': int(idx),
                'trajectory_event_id': int(session_id),
                'timestamp': float(ts),
                'search_session_id': int(session_id),
                'event_type': 'history',
            })

        user_rows.sort(
            key=lambda row: (
                float(row.get('timestamp', np.inf)),
                0 if row['channel'] == 'R' else 1,
                int(row['trajectory_local_index']),
            )
        )
        for traj_idx, row in enumerate(user_rows):
            row['trajectory_index'] = int(traj_idx)
            rows.append(row)

    return rows


def main() -> None:
    args = build_args()
    ckpt_path = resolve_ckpt_path(args)

    init_setting(args.data)
    utils.load_hyperparam(args)
    utils.setup_seed(args.random_seed)

    device = torch.device('cpu')
    if args.gpu != 'cpu' and torch.cuda.is_available():
        os_visible = str(args.gpu)
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = os_visible
        device = torch.device('cuda:0')
    args.device = device
    args.model_path = str(ckpt_path.parent)
    args.train = 0
    args.test_path = str(ckpt_path)

    state_dict = torch.load(ckpt_path, map_location='cpu')
    intent_num = infer_intent_num(state_dict)
    print(f'Loaded checkpoint: {ckpt_path}')
    print(f'intent_num K = {intent_num}')

    model = UniSAR(args).to(device)
    model.load_model(str(ckpt_path))
    model.eval()

    runner = SarRunner(args)
    rec_loader, src_loader, rec_data, src_data = get_split_loaders(runner, args.split)

    records = []
    records.extend(export_channel(model, rec_loader, rec_data.sampler.data.reset_index(drop=True), 'R', args.split))
    records.extend(export_channel(model, src_loader, src_data.sampler.data.reset_index(drop=True), 'S', args.split))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f'saved: {out_path}')
    print(df.head())

    if args.export_trajectory:
        user_vocab = utils.load_pickle(const.user_vocab)
        trajectory_rows = export_user_trajectory(user_vocab, args.split)
        trajectory_path = Path(args.trajectory_output)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_df = pd.DataFrame(trajectory_rows)
        trajectory_df.to_csv(trajectory_path, index=False)
        print(f'saved trajectory: {trajectory_path}')
        print(trajectory_df.head())


if __name__ == '__main__':
    main()
