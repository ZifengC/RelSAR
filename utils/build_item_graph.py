import argparse
import math
from collections import Counter, defaultdict

import torch
from tqdm import tqdm

from utils import const, utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build a static top-k item graph for UniSAR.')
    parser.add_argument('--data',
                        type=str,
                        default='KuaiSAR',
                        choices=['KuaiSAR', 'Amazon'])
    parser.add_argument('--topk', type=int, default=16)
    parser.add_argument('--transition_weight', type=float, default=1.0)
    parser.add_argument('--cooc_weight', type=float, default=0.3)
    parser.add_argument('--cooc_window', type=int, default=3)
    parser.add_argument('--min_weight', type=float, default=1.0)
    parser.add_argument('--output',
                        type=str,
                        default='data/item_graph_topk.pt')
    return parser.parse_args()


def init_const(data_name):
    if data_name == 'KuaiSAR':
        const.init_setting_KuaiSAR()
    elif data_name == 'Amazon':
        const.init_setting_Amazon()
    else:
        raise ValueError(f'Unsupported data: {data_name}')


def build_src_proxy_items(session_vocab):
    pos_items = session_vocab['pos_items']
    proxy_items = torch.zeros(pos_items.shape[0], dtype=torch.long)
    non_zero_mask = pos_items != 0
    has_click = non_zero_mask.any(dim=1)
    if has_click.any():
        first_click_idx = non_zero_mask.float().argmax(dim=1)
        row_index = torch.arange(pos_items.shape[0])
        picked = pos_items[row_index, first_click_idx]
        picked = torch.where(has_click, picked, torch.zeros_like(picked))
        proxy_items = picked.long()
    return proxy_items


def add_edge(edge_counter, src_item, dst_item, weight):
    if src_item <= 0 or dst_item <= 0 or src_item == dst_item:
        return
    edge_counter[src_item][dst_item] += weight


def build_item_graph(user_vocab,
                     src_proxy_items,
                     transition_weight=1.0,
                     cooc_weight=0.3,
                     cooc_window=3):
    edge_counter = defaultdict(Counter)
    skipped_users = 0

    user_iter = tqdm(user_vocab.items(), desc='Building graph', unit='user')
    for _, info in user_iter:
        rec_items = info.get('rec_his', [])
        rec_ts = info.get('rec_his_ts', [])
        src_sessions = info.get('src_session_his', [])
        src_ts = info.get('src_session_his_ts', [])

        events = []
        for item, ts in zip(rec_items, rec_ts):
            item = int(item)
            if item > 0 and math.isfinite(ts):
                events.append((float(ts), item))

        for session_id, ts in zip(src_sessions, src_ts):
            session_id = int(session_id)
            if session_id <= 0 or not math.isfinite(ts):
                continue
            if session_id >= src_proxy_items.numel():
                continue
            proxy_item = int(src_proxy_items[session_id].item())
            if proxy_item > 0:
                events.append((float(ts), proxy_item))

        if len(events) <= 1:
            skipped_users += 1
            continue

        events.sort(key=lambda x: x[0])
        item_seq = [item for _, item in events]

        for idx in range(len(item_seq) - 1):
            src_item = item_seq[idx]
            dst_item = item_seq[idx + 1]
            add_edge(edge_counter, src_item, dst_item, transition_weight)
            add_edge(edge_counter, dst_item, src_item, transition_weight)

        for idx in range(len(item_seq)):
            src_item = item_seq[idx]
            upper = min(len(item_seq), idx + cooc_window + 1)
            for jdx in range(idx + 1, upper):
                dst_item = item_seq[jdx]
                weight = cooc_weight / float(jdx - idx)
                add_edge(edge_counter, src_item, dst_item, weight)
                add_edge(edge_counter, dst_item, src_item, weight)

    return edge_counter, skipped_users


def convert_to_topk(edge_counter, item_num, topk, min_weight):
    neighbor_ids = torch.zeros((item_num, topk), dtype=torch.long)
    neighbor_weights = torch.zeros((item_num, topk), dtype=torch.float32)

    item_iter = tqdm(range(item_num), desc='Top-k pruning', unit='item')
    for item_id in item_iter:
        if item_id not in edge_counter:
            continue
        ranked = [(nbr, weight) for nbr, weight in edge_counter[item_id].items()
                  if weight >= min_weight and nbr > 0]
        if not ranked:
            continue
        ranked.sort(key=lambda x: (-x[1], x[0]))
        ranked = ranked[:topk]

        ids = torch.tensor([nbr for nbr, _ in ranked], dtype=torch.long)
        weights = torch.tensor([weight for _, weight in ranked],
                               dtype=torch.float32)
        weight_sum = weights.sum().clamp_min(1e-8)
        weights = weights / weight_sum

        neighbor_ids[item_id, :ids.numel()] = ids
        neighbor_weights[item_id, :weights.numel()] = weights

    return neighbor_ids, neighbor_weights


def main():
    args = parse_args()
    init_const(args.data)

    user_vocab = utils.load_pickle(const.user_vocab)
    session_vocab = utils.load_pickle(const.session_map_vocab)
    src_proxy_items = build_src_proxy_items(session_vocab)

    edge_counter, skipped_users = build_item_graph(
        user_vocab=user_vocab,
        src_proxy_items=src_proxy_items,
        transition_weight=args.transition_weight,
        cooc_weight=args.cooc_weight,
        cooc_window=args.cooc_window)

    neighbor_ids, neighbor_weights = convert_to_topk(
        edge_counter=edge_counter,
        item_num=const.item_id_num,
        topk=args.topk,
        min_weight=args.min_weight)

    payload = {
        'neighbor_ids': neighbor_ids,
        'neighbor_weights': neighbor_weights,
        'meta': {
            'data': args.data,
            'topk': args.topk,
            'transition_weight': args.transition_weight,
            'cooc_weight': args.cooc_weight,
            'cooc_window': args.cooc_window,
            'min_weight': args.min_weight,
            'skipped_users': skipped_users
        }
    }
    utils.check_dir(args.output)
    torch.save(payload, args.output)
    non_empty = int((neighbor_weights.sum(dim=1) > 0).sum().item())
    print(
        f"saved graph to {args.output} | non_empty_items={non_empty} | skipped_users={skipped_users}"
    )


if __name__ == '__main__':
    main()
