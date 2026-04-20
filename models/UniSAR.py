import math
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from utils import const

from .BaseModel import BaseModel
from .layers import FullyConnectedLayer, feature_align, PositionalEmbedding, PLE_layer


class UniSAR(BaseModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--num_heads', type=int, default=2)

        parser.add_argument('--q_i_cl_temp', type=float, default=0.5)
        parser.add_argument('--q_i_cl_weight', type=float, default=0.001)

        parser.add_argument('--his_cl_temp', type=float, default=0.1)
        parser.add_argument('--his_cl_weight', type=float, default=0.1)

        parser.add_argument('--pred_hid_units',
                            type=List,
                            default=[200, 80, 1])
        parser.add_argument('--intent_temp', type=float, default=0.5)
        parser.add_argument('--cf_gate_scale', type=float, default=10.0)
        parser.add_argument('--cf_consistency_weight',
                            type=float,
                            default=0.01)
        parser.add_argument('--item_graph_path', type=str, default='')
        parser.add_argument('--uncertainty_reg_weight', type=float, default=0.0001)
        parser.add_argument('--cf_sparsity_weight', type=float, default=0.0001)
        parser.add_argument('--path_competition_weight',
                            type=float,
                            default=0.0)
        parser.add_argument('--intent_separation_weight',
                            type=float,
                            default=0.0)
        parser.add_argument('--intent_separation_margin',
                            type=float,
                            default=0.2)
        parser.add_argument('--intent_num', type=int, default=4)
        parser.add_argument('--intent_heads', type=int, default=2)
        parser.add_argument('--intent_dropout', type=float, default=0.1)
        parser.add_argument('--intent_mu_mix', type=float, default=0.5)
        parser.add_argument('--intent_collapse_weight',
                            type=float,
                            default=0.001)
        parser.add_argument('--intent_diversity_margin',
                            type=float,
                            default=0.2)
        parser.add_argument('--transition_decay', type=float, default=0.2)
        parser.add_argument('--explore_temp_scale', type=float, default=2.0)
        parser.add_argument('--exploit_temp_scale', type=float, default=1.5)
        parser.add_argument('--attention_temp_min', type=float, default=0.7)
        parser.add_argument('--attention_temp_max', type=float, default=1.5)
        parser.add_argument('--intent_assign_bias_weight',
                            type=float,
                            default=0.1)
        parser.add_argument('--intent_mu_bias_weight',
                            type=float,
                            default=0.1)
        parser.add_argument('--uncertainty_bias_weight',
                            type=float,
                            default=0.1)
        parser.add_argument('--post_intent_attention_weight',
                            type=float,
                            default=1.0)
        parser.add_argument('--intent_peak_ceiling', type=float, default=0.80)
        parser.add_argument('--cf_mask_floor', type=float, default=0.05)
        parser.add_argument('--cf_mask_ceiling', type=float, default=0.95)

        return BaseModel.parse_model_args(parser)

    def __init__(self, args):
        super().__init__(args)
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.batch_size = args.batch_size
        self.intent_temp = args.intent_temp
        self.cf_gate_scale = args.cf_gate_scale
        self.cf_consistency_weight = args.cf_consistency_weight
        self.item_graph_path = args.item_graph_path
        self.uncertainty_reg_weight = args.uncertainty_reg_weight
        self.cf_sparsity_weight = args.cf_sparsity_weight
        self.path_competition_weight = args.path_competition_weight
        self.intent_separation_weight = args.intent_separation_weight
        self.intent_separation_margin = args.intent_separation_margin
        self.intent_num = args.intent_num
        self.intent_mu_mix = args.intent_mu_mix
        self.intent_collapse_weight = args.intent_collapse_weight
        self.intent_diversity_margin = args.intent_diversity_margin
        self.transition_decay = args.transition_decay
        self.explore_temp_scale = args.explore_temp_scale
        self.exploit_temp_scale = args.exploit_temp_scale
        self.attention_temp_min = args.attention_temp_min
        self.attention_temp_max = args.attention_temp_max
        self.intent_assign_bias_weight = args.intent_assign_bias_weight
        self.intent_mu_bias_weight = args.intent_mu_bias_weight
        self.uncertainty_bias_weight = args.uncertainty_bias_weight
        self.post_intent_attention_weight = args.post_intent_attention_weight
        self.intent_peak_ceiling = args.intent_peak_ceiling
        self.cf_mask_floor = args.cf_mask_floor
        self.cf_mask_ceiling = args.cf_mask_ceiling

        self.src_pos = PositionalEmbedding(const.max_src_session_his_len,
                                           self.item_size)
        self.rec_pos = PositionalEmbedding(const.max_rec_his_len,
                                           self.item_size)
        self.global_pos_emb = PositionalEmbedding(
            const.max_rec_his_len + const.max_src_session_his_len,
            self.item_size)

        self.rec_transformer = Transformer(emb_size=self.item_size,
                                           num_heads=self.num_heads,
                                           num_layers=self.num_layers,
                                           dropout=self.dropout,
                                           intent_assign_bias_weight=self.intent_assign_bias_weight,
                                           intent_mu_bias_weight=self.intent_mu_bias_weight,
                                           uncertainty_bias_weight=self.uncertainty_bias_weight,
                                           explore_temp_scale=self.explore_temp_scale,
                                           exploit_temp_scale=self.exploit_temp_scale,
                                           attention_temp_min=self.attention_temp_min,
                                           attention_temp_max=self.attention_temp_max)
        self.src_transformer = Transformer(emb_size=self.item_size,
                                           num_heads=self.num_heads,
                                           num_layers=self.num_layers,
                                           dropout=self.dropout,
                                           intent_assign_bias_weight=self.intent_assign_bias_weight,
                                           intent_mu_bias_weight=self.intent_mu_bias_weight,
                                           uncertainty_bias_weight=self.uncertainty_bias_weight,
                                           explore_temp_scale=self.explore_temp_scale,
                                           exploit_temp_scale=self.exploit_temp_scale,
                                           attention_temp_min=self.attention_temp_min,
                                           attention_temp_max=self.attention_temp_max)
        self.global_transformer = Transformer(emb_size=self.item_size,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_layers,
                                              dropout=self.dropout,
                                              intent_assign_bias_weight=self.intent_assign_bias_weight,
                                              intent_mu_bias_weight=self.intent_mu_bias_weight,
                                              uncertainty_bias_weight=self.uncertainty_bias_weight,
                                              explore_temp_scale=self.explore_temp_scale,
                                              exploit_temp_scale=self.exploit_temp_scale,
                                              attention_temp_min=self.attention_temp_min,
                                              attention_temp_max=self.attention_temp_max)

        self.q_i_cl_temp = args.q_i_cl_temp
        self.q_i_cl_weight = args.q_i_cl_weight
        if self.q_i_cl_weight > 0:
            self.query_item_alignment = True
            self.feature_alignment = feature_align(self.q_i_cl_temp,
                                                   self.item_size)

        self.his_cl_temp = args.his_cl_temp
        self.his_cl_weight = args.his_cl_weight
        if self.his_cl_weight > 0:
            self.rec_his_cl = TransAlign(batch_size=self.batch_size,
                                         hidden_dim=self.item_size,
                                         device=self.device,
                                         infoNCE_temp=self.his_cl_temp)
            self.src_his_cl = TransAlign(batch_size=self.batch_size,
                                         hidden_dim=self.item_size,
                                         device=self.device,
                                         infoNCE_temp=self.his_cl_temp)

        self.intent_mean_proj = nn.Linear(self.item_size, self.item_size)
        self.intent_logvar_proj = nn.Linear(self.item_size * 4,
                                            self.item_size)
        self.intent_discovery = LatentIntentDiscovery(
            emb_dim=self.item_size,
            num_intents=self.intent_num,
            num_heads=args.intent_heads,
            dropout=args.intent_dropout)
        self.intent_query_proj = nn.Linear(self.item_size, self.item_size)
        self.intent_key_proj = nn.Linear(self.item_size, self.item_size)
        self.intent_value_proj = nn.Linear(self.item_size, self.item_size)
        self.intent_token_proj = nn.Linear(self.item_size, self.item_size)
        self.intent_target_proj = nn.Linear(self.item_size, self.item_size)

        self.rec_his_attn_pooling = Target_Attention(self.item_size,
                                                     self.item_size)
        self.src_his_attn_pooling = Target_Attention(self.item_size,
                                                     self.item_size)

        self.rec_query = torch.nn.parameter.Parameter(torch.randn(
            (1, self.query_size), requires_grad=True),
                                                      requires_grad=True)
        nn.init.xavier_normal_(self.rec_query)

        self.hidden_unit = args.pred_hid_units

        input_dim = 3 * self.item_size + self.user_size + self.query_size
        self.ple_layer = PLE_layer(orig_input_dim=input_dim,
                                   bottom_mlp_dims=[64],
                                   tower_mlp_dims=[128, 64],
                                   task_num=2,
                                   shared_expert_num=4,
                                   specific_expert_num=4,
                                   dropout=self.dropout)
        self.rec_fc_layer = FullyConnectedLayer(input_size=64,
                                                hidden_unit=self.hidden_unit,
                                                batch_norm=False,
                                                sigmoid=True,
                                                activation='relu',
                                                dropout=self.dropout)
        self.src_fc_layer = FullyConnectedLayer(input_size=64,
                                                hidden_unit=self.hidden_unit,
                                                batch_norm=False,
                                                sigmoid=True,
                                                activation='relu',
                                                dropout=self.dropout)

        self.loss_fn = nn.BCELoss()
        self.register_buffer('item_graph_neighbor_ids',
                             torch.empty(0, dtype=torch.long))
        self.register_buffer('item_graph_neighbor_weights',
                             torch.empty(0, dtype=torch.float32))
        self.item_graph = self.load_item_graph(self.item_graph_path)
        self._init_weights()
        self.to(self.device)

    def src_feat_process(self, src_feat):
        query_emb, q_click_item_emb, click_item_mask = src_feat

        q_i_align_used = [query_emb, click_item_mask, q_click_item_emb]

        mean_click_item_emb = torch.sum(torch.mul(
            q_click_item_emb, click_item_mask.unsqueeze(-1)),
                                        dim=-2)  # batch, max_src_len, dim
        mean_click_item_emb = mean_click_item_emb / (torch.max(
            click_item_mask.sum(-1, keepdim=True),
            torch.ones_like(click_item_mask.sum(-1, keepdim=True))))
        query_his_emb = query_emb
        click_item_his_emb = mean_click_item_emb

        return query_his_emb + click_item_his_emb, q_i_align_used

    def get_all_his_emb(self, all_his, all_his_type):
        rec_his = torch.masked_fill(all_his, all_his_type != 1, 0)
        rec_his_emb = self.session_embedding.get_item_emb(rec_his)
        rec_his_emb = torch.masked_fill(rec_his_emb,
                                        (all_his_type != 1).unsqueeze(-1), 0)

        src_session_his = torch.masked_fill(all_his, all_his_type != 2, 0)
        src_his_emb, q_i_align_used = self.src_feat_process(
            self.session_embedding(src_session_his))
        src_his_emb = torch.masked_fill(src_his_emb,
                                        (all_his_type != 2).unsqueeze(-1), 0)

        all_his_emb = rec_his_emb + src_his_emb
        all_his_mask = torch.where(all_his == 0, 1, 0).bool()

        return all_his_emb, all_his_mask, q_i_align_used

    def repeat_feat(self, feature_list, items_emb):
        repeat_feature_list = [
            torch.repeat_interleave(feat, items_emb.size(1), dim=0)
            for feat in feature_list
        ]
        items_emb = items_emb.reshape(-1, items_emb.size(-1))

        return repeat_feature_list, items_emb

    def mean_pooling(self, output, his_len):
        return torch.sum(output, dim=1) / his_len.unsqueeze(-1)

    def split_rec_src(self, all_his_emb, all_his_type):
        rec_his_emb = torch.masked_select(
            all_his_emb, (all_his_type == 1).unsqueeze(-1)).reshape(
                (all_his_emb.shape[0], const.max_rec_his_len,
                 all_his_emb.shape[2]))
        src_his_emb = torch.masked_select(
            all_his_emb, (all_his_type == 2).unsqueeze(-1)).reshape(
                (all_his_emb.shape[0], const.max_src_session_his_len,
                 all_his_emb.shape[2]))
        return rec_his_emb, src_his_emb

    def split_rec_src_ids(self, all_his, all_his_type):
        rec_his = torch.masked_select(all_his, all_his_type == 1).reshape(
            (all_his.shape[0], const.max_rec_his_len))
        src_his = torch.masked_select(all_his, all_his_type == 2).reshape(
            (all_his.shape[0], const.max_src_session_his_len))
        return rec_his, src_his

    def load_item_graph(self, graph_path):
        if graph_path == '' or not os.path.exists(graph_path):
            return None
        if graph_path.endswith('.pt') or graph_path.endswith('.pth'):
            graph = torch.load(graph_path, map_location='cpu')
        elif graph_path.endswith('.pkl'):
            with open(graph_path, 'rb') as f:
                graph = pickle.load(f)
        else:
            raise ValueError('Unsupported graph format: {}'.format(graph_path))

        if isinstance(graph, dict) and 'neighbor_ids' in graph and 'neighbor_weights' in graph:
            self.item_graph_neighbor_ids = graph['neighbor_ids'].long()
            self.item_graph_neighbor_weights = graph['neighbor_weights'].float()
            return {'format': 'topk'}
        if torch.is_tensor(graph):
            return graph.float()
        return graph

    def sequence_mean(self, seq_emb, mask):
        valid = (~mask).unsqueeze(-1).float()
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (seq_emb * valid).sum(dim=1) / denom

    def safe_masked_mean(self, values, mask):
        valid_values = values.masked_select(mask)
        if valid_values.numel() == 0:
            return values.new_tensor(0.0)
        return valid_values.mean()

    def safe_masked_std(self, values, mask):
        valid_values = values.masked_select(mask)
        if valid_values.numel() <= 1:
            return values.new_tensor(0.0)
        return valid_values.std(unbiased=False)

    def compute_intent_state(self, all_his_emb, all_his_mask):
        intents = self.intent_discovery(all_his_emb, all_his_mask)
        assign_logits = torch.matmul(all_his_emb, intents.transpose(-1, -2))
        assign_logits = assign_logits / max(self.intent_temp, 1e-6)
        assign = torch.softmax(assign_logits, dim=-1)
        assign = assign.masked_fill(all_his_mask.unsqueeze(-1), 0.0)

        intent_mu = torch.matmul(assign, intents)
        base_mu = self.intent_mean_proj(all_his_emb)
        mu_mix = float(min(max(self.intent_mu_mix, 0.0), 1.0))
        mu = mu_mix * intent_mu + (1.0 - mu_mix) * base_mu

        residual = (all_his_emb - intent_mu).pow(2)
        assign_prob = assign.clamp_min(1e-8)
        entropy = -(assign_prob * assign_prob.log()).sum(dim=-1,
                                                         keepdim=True)
        if self.intent_num > 1:
            entropy = entropy / math.log(self.intent_num)
        logvar_input = torch.cat(
            [all_his_emb, intent_mu, residual,
             entropy.expand_as(all_his_emb)],
            dim=-1)
        logvar = self.intent_logvar_proj(logvar_input).clamp(min=-6.0, max=2.0)
        mu = mu.masked_fill(all_his_mask.unsqueeze(-1), 0.0)
        logvar = logvar.masked_fill(all_his_mask.unsqueeze(-1), 0.0)
        sigma = torch.exp(0.5 * logvar)
        valid_mask = ~all_his_mask
        collapse_reg, proto_sim_mean, proto_sim_max = \
            self.compute_intent_collapse_diagnostics(intents, assign,
                                                     valid_mask)
        diagnostics = {
            'intent_collapse_reg':
            collapse_reg,
            'intent_proto_sim_mean':
            proto_sim_mean,
            'intent_proto_sim_max':
            proto_sim_max,
            'intent_assign_entropy':
            self.safe_masked_mean(entropy.squeeze(-1), valid_mask),
            'intent_usage_max':
            self.compute_intent_usage(assign, valid_mask).max(),
            'intent_residual_mean':
            self.safe_masked_mean(residual.mean(dim=-1), valid_mask)
        }
        return mu, logvar, sigma, assign, diagnostics

    def compute_path_transition_dynamics(self, assign, mask):
        if assign.size(1) <= 1:
            zeros = assign.new_zeros(assign.size(0), assign.size(1))
            return zeros, zeros

        assign_prob = assign.clamp_min(1e-8)
        entropy = -(assign_prob * assign_prob.log()).sum(dim=-1)
        if self.intent_num > 1:
            entropy = entropy / math.log(self.intent_num)

        curr = assign_prob.unsqueeze(2)
        prev = assign_prob.unsqueeze(1)
        midpoint = (curr + prev).mul(0.5).clamp_min(1e-8)
        js_div = 0.5 * (
            curr * (curr.log() - midpoint.log())).sum(dim=-1) + 0.5 * (
                prev * (prev.log() - midpoint.log())).sum(dim=-1)
        if self.intent_num > 1:
            js_div = js_div / math.log(2.0)

        entropy_delta = entropy.unsqueeze(2) - entropy.unsqueeze(1)
        intent_similarity = (curr * prev).sum(dim=-1)
        curr_confidence = assign_prob.max(dim=-1).values.unsqueeze(2)

        positions = torch.arange(assign.size(1), device=assign.device)
        distance = positions.view(1, -1, 1) - positions.view(1, 1, -1)
        pair_mask = distance > 0
        valid_pair = pair_mask & (~mask).unsqueeze(2) & (~mask).unsqueeze(1)
        decay = torch.exp(-self.transition_decay *
                          distance.clamp_min(0).float())
        pair_weight = decay * valid_pair.float()
        denom = pair_weight.sum(dim=-1).clamp_min(1e-8)

        explore_pair = js_div * F.relu(entropy_delta)
        exploit_pair = intent_similarity * F.relu(-entropy_delta) * \
            curr_confidence
        explore = (explore_pair * pair_weight).sum(dim=-1) / denom
        exploit = (exploit_pair * pair_weight).sum(dim=-1) / denom
        explore = explore.masked_fill(mask, 0.0)
        exploit = exploit.masked_fill(mask, 0.0)
        return explore, exploit

    def compute_intent_usage(self, assign, valid_mask):
        valid_count = valid_mask.float().sum()
        if valid_count <= 0:
            return assign.new_full((self.intent_num, ), 1.0 / self.intent_num)
        token_weights = valid_mask.float().unsqueeze(-1)
        usage = (assign * token_weights).sum(dim=(0, 1))
        usage = usage / valid_count.clamp_min(1.0)
        return usage / usage.sum().clamp_min(1e-8)

    def compute_intent_collapse_diagnostics(self, intents, assign, valid_mask):
        if self.intent_num <= 1:
            zero = intents.new_tensor(0.0)
            return zero, zero, zero
        norm_intents = F.normalize(intents, dim=-1)
        proto_sim = torch.matmul(norm_intents,
                                 norm_intents.transpose(-1, -2))
        off_diag_mask = ~torch.eye(self.intent_num,
                                   dtype=torch.bool,
                                   device=intents.device).unsqueeze(0)
        off_diag_sim = proto_sim.masked_select(off_diag_mask)
        proto_sim_mean = off_diag_sim.mean()
        proto_sim_max = off_diag_sim.max()
        proto_violation = F.relu(proto_sim - self.intent_diversity_margin)
        proto_loss = proto_violation.masked_select(off_diag_mask).mean()

        valid_count = valid_mask.float().sum()
        if valid_count <= 0:
            return proto_loss, proto_sim_mean, proto_sim_max
        usage = self.compute_intent_usage(assign, valid_mask)
        usage_entropy = -(usage.clamp_min(1e-8) *
                          usage.clamp_min(1e-8).log()).sum()
        if self.intent_num > 1:
            usage_entropy = usage_entropy / math.log(self.intent_num)
        usage_loss = 1.0 - usage_entropy
        return proto_loss + usage_loss, proto_sim_mean, proto_sim_max

    def compute_intent_separation_loss(self, mu, logvar):
        valid_mask = ~(mu.abs().sum(dim=-1) == 0)
        flat_mu = mu[valid_mask]
        flat_logvar = logvar[valid_mask]

        if flat_mu.size(0) <= 1:
            return mu.new_tensor(0.0)

        max_tokens = 512
        if flat_mu.size(0) > max_tokens:
            sample_idx = torch.randperm(flat_mu.size(0),
                                        device=flat_mu.device)[:max_tokens]
            flat_mu = flat_mu[sample_idx]
            flat_logvar = flat_logvar[sample_idx]

        norm_mu = F.normalize(flat_mu, dim=-1)
        variance_scale = torch.exp(flat_logvar).mean(dim=-1).clamp_min(1e-6)
        pair_scale = torch.sqrt(
            variance_scale.unsqueeze(1) * variance_scale.unsqueeze(0))
        sim_matrix = torch.matmul(norm_mu, norm_mu.transpose(0, 1)) / pair_scale
        confidence = torch.exp(-flat_logvar.mean(dim=-1))
        pair_confidence = confidence.unsqueeze(1) * confidence.unsqueeze(0)

        off_diag_mask = ~torch.eye(flat_mu.size(0),
                                   dtype=torch.bool,
                                   device=mu.device)
        margin_violation = F.relu(sim_matrix - self.intent_separation_margin)
        weighted_violation = margin_violation * pair_confidence
        return weighted_violation.masked_select(off_diag_mask).mean()

    def get_src_proxy_item_ids(self, src_session_ids):
        flat_src = src_session_ids.reshape(-1)
        proxy_items = torch.zeros_like(flat_src)
        valid_mask = flat_src != 0
        if valid_mask.sum() == 0:
            return proxy_items.reshape_as(src_session_ids)

        clicked_items = self.session_map_vocab['pos_items'][flat_src[valid_mask]]
        non_zero_mask = clicked_items != 0
        has_click = non_zero_mask.any(dim=1)
        if has_click.any():
            first_click_idx = non_zero_mask.float().argmax(dim=1)
            row_index = torch.arange(first_click_idx.size(0),
                                     device=first_click_idx.device)
            picked = clicked_items[row_index, first_click_idx]
            picked = torch.where(has_click, picked, torch.zeros_like(picked))
            proxy_items[valid_mask] = picked
        return proxy_items.reshape_as(src_session_ids)

    def get_graph_prior(self, target_item_ids, token_item_ids):
        if target_item_ids is None:
            return torch.ones((token_item_ids.size(0), token_item_ids.size(1)),
                              device=token_item_ids.device)
        target_item_ids = target_item_ids.long()
        token_item_ids = token_item_ids.long()
        if isinstance(self.item_graph, dict) and self.item_graph.get('format') == 'topk':
            neighbor_ids = self.item_graph_neighbor_ids.to(token_item_ids.device)
            neighbor_weights = self.item_graph_neighbor_weights.to(
                token_item_ids.device)
            max_row = neighbor_ids.size(0) - 1
            tgt = target_item_ids.clamp(min=0, max=max_row)
            row_neighbor_ids = neighbor_ids[tgt]
            row_neighbor_weights = neighbor_weights[tgt]
            match_mask = token_item_ids.unsqueeze(-1) == row_neighbor_ids.unsqueeze(1)
            matched_weights = torch.where(
                match_mask, row_neighbor_weights.unsqueeze(1),
                torch.zeros_like(row_neighbor_weights.unsqueeze(1)))
            graph_scores = matched_weights.max(dim=-1).values
            return torch.where(graph_scores > 0, graph_scores,
                               torch.ones_like(graph_scores))
        if self.item_graph is None:
            return torch.ones((target_item_ids.size(0), token_item_ids.size(1)),
                              device=target_item_ids.device)

        if torch.is_tensor(self.item_graph):
            graph = self.item_graph.to(target_item_ids.device)
            max_row = graph.size(0) - 1
            max_col = graph.size(1) - 1
            tgt = target_item_ids.clamp(min=0, max=max_row)
            src = token_item_ids.clamp(min=0, max=max_col)
            return graph[tgt.unsqueeze(1), src]

        graph_scores = torch.ones((target_item_ids.size(0), token_item_ids.size(1)),
                                  device=target_item_ids.device)
        for row_idx in range(target_item_ids.size(0)):
            neighbors = self.item_graph.get(int(target_item_ids[row_idx].item()),
                                            {})
            for col_idx in range(token_item_ids.size(1)):
                graph_scores[row_idx, col_idx] = neighbors.get(
                    int(token_item_ids[row_idx, col_idx].item()), 1.0)
        return graph_scores

    def apply_intent_attention(self, query_seq, same_seq, cross_seq, same_mask,
                               cross_mask, same_item_ids, cross_item_ids,
                               target_emb, target_item_ids, same_mu,
                               same_sigma, cross_mu, cross_sigma, same_gate,
                               cross_gate, query_explore, query_exploit):
        token_bank = torch.cat([same_seq, cross_seq], dim=1)
        token_mask = torch.cat([same_mask, cross_mask], dim=1)
        token_item_ids = torch.cat([same_item_ids, cross_item_ids], dim=1)
        token_mu = torch.cat([same_mu, cross_mu], dim=1)
        token_sigma = torch.cat([same_sigma, cross_sigma], dim=1)

        query_proj = self.intent_query_proj(query_seq)
        key_proj = self.intent_key_proj(token_bank)
        value_proj = self.intent_value_proj(token_bank)

        raw_logits = torch.matmul(query_proj,
                                  key_proj.transpose(-1, -2)) / math.sqrt(
                                      query_proj.size(-1))

        token_intent = self.intent_token_proj(token_bank)
        target_intent = self.intent_target_proj(target_emb).unsqueeze(1)
        gaussian_scale = token_sigma.pow(2).clamp_min(1e-6)
        token_uncertainty = -((token_bank - token_mu).pow(2) /
                              gaussian_scale).mean(dim=-1, keepdim=True)
        target_uncertainty = -((target_intent - token_mu).pow(2) /
                               gaussian_scale).mean(dim=-1, keepdim=True)
        intent_match_logits = torch.matmul(query_proj, token_intent.transpose(
            -1, -2)) / math.sqrt(query_proj.size(-1))
        intent_match_logits = intent_match_logits + token_uncertainty.transpose(
            -1, -2) + target_uncertainty.transpose(-1, -2)
        intent_match_logits = intent_match_logits / self.intent_temp
        intent_match_logits = self.post_intent_attention_weight * intent_match_logits

        graph_prior = self.get_graph_prior(target_item_ids, token_item_ids)
        graph_prior = graph_prior.unsqueeze(1)

        same_gate = same_gate.clamp_min(1e-8).unsqueeze(1).unsqueeze(2)
        cross_gate = cross_gate.clamp_min(1e-8).unsqueeze(1).unsqueeze(2)
        token_path_gate = torch.cat([
            same_gate.expand(-1, 1, same_seq.size(1)),
            cross_gate.expand(-1, 1, cross_seq.size(1))
        ],
                                    dim=-1)

        attn_logits = raw_logits + intent_match_logits
        attn_logits = attn_logits + torch.log(token_path_gate)
        attn_logits = attn_logits + torch.log(graph_prior.clamp_min(1e-8))
        attn_logits = attn_logits.masked_fill(token_mask.unsqueeze(1), -1e16)

        temperature = 1.0 + self.explore_temp_scale * query_explore - \
            self.exploit_temp_scale * query_exploit
        temperature = temperature.clamp(min=self.attention_temp_min,
                                        max=self.attention_temp_max)
        temperature = temperature.masked_fill(same_mask, 1.0).unsqueeze(-1)
        attn_probs = torch.softmax(attn_logits / temperature, dim=-1)
        attended = torch.matmul(attn_probs, value_proj)
        output = F.layer_norm(attended + query_seq, (query_seq.size(-1), ))
        output = output.masked_fill(same_mask.unsqueeze(-1), 0.0)
        valid_token_count = (~same_mask).sum().clamp_min(1)
        attention_peak = attn_probs.max(dim=-1).values.masked_fill(
            same_mask, 0.0).sum() / valid_token_count
        valid_temperature = temperature.squeeze(-1).masked_fill(same_mask, 0.0)
        temperature_mean = valid_temperature.sum() / valid_token_count
        temperature_std = temperature.squeeze(-1).masked_select(
            ~same_mask).std(unbiased=False) if (~same_mask).any() else \
            temperature.new_tensor(0.0)
        return output, attention_peak, temperature_mean, temperature_std

    def apply_path_aware_attention(self, query_seq, same_seq, cross_seq,
                                   same_mask, cross_mask, same_item_ids,
                                   cross_item_ids, target_emb, target_item_ids,
                                   same_mu, same_sigma, cross_mu, cross_sigma,
                                   query_explore, query_exploit,
                                   target_domain):
        ones = query_seq.new_ones(query_seq.size(0))
        zeros = query_seq.new_zeros(query_seq.size(0))

        same_only_output, _, _, _ = self.apply_intent_attention(
            query_seq, same_seq, cross_seq, same_mask, cross_mask,
            same_item_ids, cross_item_ids, target_emb, target_item_ids, same_mu,
            same_sigma, cross_mu, cross_sigma, ones, zeros, query_explore,
            query_exploit)
        cross_only_output, _, _, _ = self.apply_intent_attention(
            query_seq, same_seq, cross_seq, same_mask, cross_mask,
            same_item_ids, cross_item_ids, target_emb, target_item_ids, same_mu,
            same_sigma, cross_mu, cross_sigma, zeros, ones, query_explore,
            query_exploit)
        full_output, _, _, _ = self.apply_intent_attention(
            query_seq, same_seq, cross_seq, same_mask, cross_mask,
            same_item_ids, cross_item_ids, target_emb, target_item_ids, same_mu,
            same_sigma, cross_mu, cross_sigma, ones, ones, query_explore,
            query_exploit)
        return {
            'same_only': same_only_output,
            'cross_only': cross_only_output,
            'full': full_output
        }

    def compute_counterfactual_gates(self, full_pred, wo_cross_pred,
                                     wo_same_pred):
        cross_delta = F.relu(full_pred - wo_cross_pred).squeeze(-1)
        same_delta = F.relu(full_pred - wo_same_pred).squeeze(-1)
        gate_logits = self.cf_gate_scale * torch.stack([same_delta, cross_delta],
                                                       dim=-1)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        return gate_probs[:, 0], gate_probs[:, 1], gate_probs

    def compute_cross_supplement_gates(self, full_pred, wo_cross_pred,
                                       wo_same_pred):
        cross_delta = F.relu(full_pred - wo_cross_pred).squeeze(-1)
        same_delta = F.relu(full_pred - wo_same_pred).squeeze(-1)
        gate_logits = self.cf_gate_scale * torch.stack([same_delta, cross_delta],
                                                       dim=-1)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        cross_gate = gate_probs[:, 1]
        same_gate = torch.ones_like(cross_gate)
        return same_gate, cross_gate, gate_probs

    def forward(self,
                user,
                all_his,
                all_his_type,
                items,
                items_emb,
                domain,
                query_emb=None):
        assert domain in ['rec', 'src']
        user_emb = self.session_embedding.get_user_emb(user)

        all_his_emb, all_his_mask, q_i_align_used = self.get_all_his_emb(
            all_his, all_his_type)

        rec_his_mask = torch.masked_select(all_his_mask,
                                           (all_his_type == 1)).reshape(
                                               (all_his_emb.shape[0],
                                                const.max_rec_his_len))
        src_his_mask = torch.masked_select(all_his_mask,
                                           (all_his_type == 2)).reshape(
                                               (all_his_emb.shape[0],
                                                const.max_src_session_his_len))

        all_mu, all_logvar, all_sigma, all_assign, intent_diagnostics = self.compute_intent_state(
            all_his_emb, all_his_mask)
        rec_mu, src_mu = self.split_rec_src(all_mu, all_his_type)
        rec_sigma, src_sigma = self.split_rec_src(all_sigma, all_his_type)
        rec_assign, src_assign = self.split_rec_src(all_assign, all_his_type)
        global_explore, global_exploit = self.compute_path_transition_dynamics(
            all_assign, all_his_mask)
        rec_explore, rec_exploit = self.compute_path_transition_dynamics(
            rec_assign, rec_his_mask)
        src_explore, src_exploit = self.compute_path_transition_dynamics(
            src_assign, src_his_mask)

        all_his_emb_w_pos = all_his_emb + self.global_pos_emb(all_his_emb)

        global_mask = all_his_type[:, :, None] == all_his_type[:, None, :]

        global_encoded = self.global_transformer(all_his_emb_w_pos,
                                                 all_his_mask,
                                                 global_mask,
                                                 intent_mu=all_mu,
                                                 intent_sigma=all_sigma,
                                                 intent_assign=all_assign,
                                                 explore=global_explore,
                                                 exploit=global_exploit)
        src2rec, rec2src = self.split_rec_src(global_encoded, all_his_type)
        rec_his_ids, src_his_ids = self.split_rec_src_ids(all_his, all_his_type)
        src_proxy_item_ids = self.get_src_proxy_item_ids(src_his_ids)

        rec_his_emb, src_his_emb = self.split_rec_src(all_his_emb,
                                                      all_his_type)
        rec_his_emb_w_pos = rec_his_emb + self.rec_pos(rec_his_emb)
        src_his_emb_w_pos = src_his_emb + self.src_pos(src_his_emb)

        rec2rec = self.rec_transformer(rec_his_emb_w_pos,
                                       rec_his_mask,
                                       intent_mu=rec_mu,
                                       intent_sigma=rec_sigma,
                                       intent_assign=rec_assign,
                                       explore=rec_explore,
                                       exploit=rec_exploit)
        src2src = self.src_transformer(src_his_emb_w_pos,
                                       src_his_mask,
                                       intent_mu=src_mu,
                                       intent_sigma=src_sigma,
                                       intent_assign=src_assign,
                                       explore=src_explore,
                                       exploit=src_exploit)

        his_cl_used = [
            src2rec, rec2rec, rec_his_mask, rec2src, src2src, src_his_mask
        ]

        regularization = {
            'uncertainty_reg': 0.5 *
            (all_mu.pow(2) + torch.exp(all_logvar) - all_logvar - 1.0).
            masked_fill(all_his_mask.unsqueeze(-1), 0.0).sum() /
            ((~all_his_mask).sum() * all_mu.size(-1)).clamp_min(1),
            'cf_sparsity_reg': torch.tensor(0.0, device=all_his.device),
            'path_competition_reg': torch.tensor(0.0, device=all_his.device),
            'intent_separation_reg': self.compute_intent_separation_loss(
                all_mu, all_logvar),
            'intent_collapse_reg':
            intent_diagnostics['intent_collapse_reg'],
            'intent_proto_sim_mean':
            intent_diagnostics['intent_proto_sim_mean'],
            'intent_proto_sim_max':
            intent_diagnostics['intent_proto_sim_max'],
            'cf_consistency_reg':
            torch.tensor(0.0, device=all_his.device),
            'intent_mu_norm':
            self.safe_masked_mean(all_mu.norm(dim=-1), ~all_his_mask),
            'intent_assign_entropy':
            intent_diagnostics['intent_assign_entropy'],
            'intent_usage_max':
            intent_diagnostics['intent_usage_max'],
            'intent_residual_mean':
            intent_diagnostics['intent_residual_mean'],
            'transition_explore_mean':
            0.5 * (self.safe_masked_mean(rec_explore, ~rec_his_mask) +
                   self.safe_masked_mean(src_explore, ~src_his_mask)),
            'transition_exploit_mean':
            0.5 * (self.safe_masked_mean(rec_exploit, ~rec_his_mask) +
                   self.safe_masked_mean(src_exploit, ~src_his_mask)),
            'uncertainty_mean':
            self.safe_masked_mean(all_sigma, ~all_his_mask.unsqueeze(-1)),
            'uncertainty_std':
            self.safe_masked_std(all_sigma, ~all_his_mask.unsqueeze(-1)),
            'cf_mask_mean':
            torch.tensor(0.0, device=all_his.device),
            'cf_necessity_mean':
            torch.tensor(0.0, device=all_his.device),
            'cf_potential_mean':
            torch.tensor(0.0, device=all_his.device),
            'cf_self_mean':
            torch.tensor(0.0, device=all_his.device),
            'path_s2s':
            torch.tensor(0.0, device=all_his.device),
            'path_r2s':
            torch.tensor(0.0, device=all_his.device),
            'path_r2r':
            torch.tensor(0.0, device=all_his.device),
            'path_s2r':
            torch.tensor(0.0, device=all_his.device),
            'rec_same_delta_mean':
            torch.tensor(0.0, device=all_his.device),
            'rec_cross_delta_mean':
            torch.tensor(0.0, device=all_his.device),
            'src_same_delta_mean':
            torch.tensor(0.0, device=all_his.device),
            'src_cross_delta_mean':
            torch.tensor(0.0, device=all_his.device),
            'rec_cross_gate_mean':
            torch.tensor(0.0, device=all_his.device),
            'src_cross_gate_mean':
            torch.tensor(0.0, device=all_his.device),
            'attention_peak':
            torch.tensor(0.0, device=all_his.device),
            'attention_temp_mean':
            torch.tensor(0.0, device=all_his.device),
            'attention_temp_std':
            torch.tensor(0.0, device=all_his.device)
        }

        feature_list = [
            rec2rec, src2rec, rec_his_mask, src2src, rec2src, src_his_mask,
            user_emb, rec_mu, rec_sigma, src_mu, src_sigma, rec_his_ids,
            src_proxy_item_ids, rec_explore, rec_exploit, src_explore,
            src_exploit
        ]
        if domain == 'src':
            assert query_emb is not None
            feature_list.append(query_emb)
        repeat_feature_list, flat_items_emb = self.repeat_feat(
            feature_list, items_emb)

        if domain == 'rec':
            rec2rec, src2rec, rec_his_mask, src2src, rec2src, src_his_mask, \
                user_emb, rec_mu, rec_sigma, src_mu, src_sigma, rec_his_ids, \
                src_proxy_item_ids, rec_explore, rec_exploit, src_explore, \
                src_exploit = repeat_feature_list
            attention_target_emb = flat_items_emb
        else:
            rec2rec, src2rec, rec_his_mask, src2src, rec2src, src_his_mask, \
                user_emb, rec_mu, rec_sigma, src_mu, src_sigma, rec_his_ids, \
                src_proxy_item_ids, rec_explore, rec_exploit, src_explore, \
                src_exploit, repeated_query_emb = repeat_feature_list
            attention_target_emb = repeated_query_emb

        rec_target_item_ids = items.reshape(-1)
        src_target_item_ids = items.reshape(-1) if domain == 'rec' else None

        rec_outputs = self.apply_path_aware_attention(
            query_seq=rec2rec,
            same_seq=rec2rec,
            cross_seq=src2rec,
            same_mask=rec_his_mask,
            cross_mask=src_his_mask,
            same_item_ids=rec_his_ids,
            cross_item_ids=src_proxy_item_ids,
            target_emb=attention_target_emb,
            target_item_ids=rec_target_item_ids,
            same_mu=rec_mu,
            same_sigma=rec_sigma,
            cross_mu=src_mu,
            cross_sigma=src_sigma,
            query_explore=rec_explore,
            query_exploit=rec_exploit,
            target_domain='rec')

        src_outputs = self.apply_path_aware_attention(
            query_seq=src2src,
            same_seq=src2src,
            cross_seq=rec2src,
            same_mask=src_his_mask,
            cross_mask=rec_his_mask,
            same_item_ids=src_proxy_item_ids,
            cross_item_ids=rec_his_ids,
            target_emb=attention_target_emb,
            target_item_ids=src_target_item_ids,
            same_mu=src_mu,
            same_sigma=src_sigma,
            cross_mu=rec_mu,
            cross_sigma=rec_sigma,
            query_explore=src_explore,
            query_exploit=src_exploit,
            target_domain='src')

        rec_same_only = self.rec_his_attn_pooling(rec_outputs['same_only'],
                                                  flat_items_emb, rec_his_mask)
        rec_cross_only = self.rec_his_attn_pooling(rec_outputs['cross_only'],
                                                   flat_items_emb, rec_his_mask)
        rec_full = self.rec_his_attn_pooling(rec_outputs['full'], flat_items_emb,
                                             rec_his_mask)
        src_same_only = self.src_his_attn_pooling(src_outputs['same_only'],
                                                  flat_items_emb, src_his_mask)
        src_cross_only = self.src_his_attn_pooling(src_outputs['cross_only'],
                                                   flat_items_emb, src_his_mask)
        src_full = self.src_his_attn_pooling(src_outputs['full'], flat_items_emb,
                                             src_his_mask)

        if domain == 'rec':
            rec_full_pred = self.inter_pred([rec_full, src_full, user_emb],
                                            flat_items_emb,
                                            domain='rec')
            rec_wo_cross_pred = self.inter_pred(
                [rec_same_only, src_full, user_emb],
                flat_items_emb,
                domain='rec')
            rec_wo_same_pred = self.inter_pred(
                [rec_cross_only, src_full, user_emb],
                flat_items_emb,
                domain='rec')

            src_full_pred = self.inter_pred([rec_full, src_full, user_emb],
                                            flat_items_emb,
                                            domain='rec')
            src_wo_cross_pred = self.inter_pred(
                [rec_full, src_same_only, user_emb],
                flat_items_emb,
                domain='rec')
            src_wo_same_pred = self.inter_pred(
                [rec_full, src_cross_only, user_emb],
                flat_items_emb,
                domain='rec')
        else:
            rec_full_pred = self.inter_pred([rec_full, src_full, user_emb],
                                            flat_items_emb,
                                            domain='src',
                                            query_emb=repeated_query_emb)
            rec_wo_cross_pred = self.inter_pred(
                [rec_same_only, src_full, user_emb],
                flat_items_emb,
                domain='src',
                query_emb=repeated_query_emb)
            rec_wo_same_pred = self.inter_pred(
                [rec_cross_only, src_full, user_emb],
                flat_items_emb,
                domain='src',
                query_emb=repeated_query_emb)

            src_full_pred = self.inter_pred([rec_full, src_full, user_emb],
                                            flat_items_emb,
                                            domain='src',
                                            query_emb=repeated_query_emb)
            src_wo_cross_pred = self.inter_pred(
                [rec_full, src_same_only, user_emb],
                flat_items_emb,
                domain='src',
                query_emb=repeated_query_emb)
            src_wo_same_pred = self.inter_pred(
                [rec_full, src_cross_only, user_emb],
                flat_items_emb,
                domain='src',
                query_emb=repeated_query_emb)

        rec_same_gate, rec_cross_gate, rec_gate_probs = self.compute_cross_supplement_gates(
            rec_full_pred, rec_wo_cross_pred, rec_wo_same_pred)
        src_same_gate, src_cross_gate, src_gate_probs = self.compute_counterfactual_gates(
            src_full_pred, src_wo_cross_pred, src_wo_same_pred)
        rec_same_delta = F.relu(rec_full_pred - rec_wo_same_pred).squeeze(-1)
        rec_cross_delta = F.relu(rec_full_pred - rec_wo_cross_pred).squeeze(-1)
        src_same_delta = F.relu(src_full_pred - src_wo_same_pred).squeeze(-1)
        src_cross_delta = F.relu(src_full_pred - src_wo_cross_pred).squeeze(-1)
        rec_consistency = 0.5 * (
            F.relu(rec_wo_cross_pred - rec_full_pred).mean() +
            F.relu(rec_wo_same_pred - rec_full_pred).mean())
        src_consistency = 0.5 * (
            F.relu(src_wo_cross_pred - src_full_pred).mean() +
            F.relu(src_wo_same_pred - src_full_pred).mean())

        rec_fusion_decoded, rec_attention_peak, rec_temp_mean, rec_temp_std = self.apply_intent_attention(
            query_seq=rec2rec,
            same_seq=rec2rec,
            cross_seq=src2rec,
            same_mask=rec_his_mask,
            cross_mask=src_his_mask,
            same_item_ids=rec_his_ids,
            cross_item_ids=src_proxy_item_ids,
            target_emb=attention_target_emb,
            target_item_ids=rec_target_item_ids,
            same_mu=rec_mu,
            same_sigma=rec_sigma,
            cross_mu=src_mu,
            cross_sigma=src_sigma,
            same_gate=rec_same_gate,
            cross_gate=rec_cross_gate,
            query_explore=rec_explore,
            query_exploit=rec_exploit)

        src_fusion_decoded, src_attention_peak, src_temp_mean, src_temp_std = self.apply_intent_attention(
            query_seq=src2src,
            same_seq=src2src,
            cross_seq=rec2src,
            same_mask=src_his_mask,
            cross_mask=rec_his_mask,
            same_item_ids=src_proxy_item_ids,
            cross_item_ids=rec_his_ids,
            target_emb=attention_target_emb,
            target_item_ids=src_target_item_ids,
            same_mu=src_mu,
            same_sigma=src_sigma,
            cross_mu=rec_mu,
            cross_sigma=rec_sigma,
            same_gate=src_same_gate,
            cross_gate=src_cross_gate,
            query_explore=src_explore,
            query_exploit=src_exploit)

        path_strengths = all_his.new_zeros((rec_same_gate.size(0), 4),
                                           dtype=rec_same_gate.dtype,
                                           device=rec_same_gate.device)
        path_strengths[:, 0] = src_same_gate
        path_strengths[:, 1] = src_cross_gate
        path_strengths[:, 2] = rec_same_gate
        path_strengths[:, 3] = rec_cross_gate
        path_dist = path_strengths / path_strengths.sum(dim=-1,
                                                        keepdim=True).clamp_min(1e-8)
        rec_gate_entropy = -(rec_gate_probs.clamp_min(1e-8) *
                             rec_gate_probs.clamp_min(1e-8).log()).sum(
                                 dim=-1).mean()
        src_gate_entropy = -(src_gate_probs.clamp_min(1e-8) *
                             src_gate_probs.clamp_min(1e-8).log()).sum(
                                 dim=-1).mean()
        regularization['cf_sparsity_reg'] = 0.5 * (rec_gate_entropy +
                                                   src_gate_entropy)
        regularization['path_competition_reg'] = -(
            path_dist.clamp_min(1e-8) * path_dist.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        regularization['cf_mask_mean'] = 0.5 * (
            rec_cross_gate.mean() + src_cross_gate.mean())
        regularization['cf_necessity_mean'] = rec_cross_gate.mean()
        regularization['cf_potential_mean'] = src_cross_gate.mean()
        regularization['cf_self_mean'] = 0.5 * (
            rec_same_gate.mean() + src_same_gate.mean())
        regularization['path_s2s'] = path_strengths[:, 0].mean()
        regularization['path_r2s'] = path_strengths[:, 1].mean()
        regularization['path_r2r'] = path_strengths[:, 2].mean()
        regularization['path_s2r'] = path_strengths[:, 3].mean()
        regularization['rec_same_delta_mean'] = rec_same_delta.mean()
        regularization['rec_cross_delta_mean'] = rec_cross_delta.mean()
        regularization['src_same_delta_mean'] = src_same_delta.mean()
        regularization['src_cross_delta_mean'] = src_cross_delta.mean()
        regularization['rec_cross_gate_mean'] = rec_cross_gate.mean()
        regularization['src_cross_gate_mean'] = src_cross_gate.mean()
        regularization['cf_consistency_reg'] = 0.5 * (rec_consistency +
                                                      src_consistency)
        regularization['attention_peak'] = 0.5 * (
            rec_attention_peak + src_attention_peak)
        regularization['attention_temp_mean'] = 0.5 * (rec_temp_mean +
                                                       src_temp_mean)
        regularization['attention_temp_std'] = 0.5 * (rec_temp_std +
                                                      src_temp_std)

        rec_fusion = rec_same_only + rec_cross_gate.unsqueeze(-1) * (
            rec_cross_only - rec_same_only)
        src_fusion = self.src_his_attn_pooling(src_fusion_decoded, flat_items_emb,
                                               src_his_mask)

        user_feats = [rec_fusion, src_fusion, user_emb]

        return user_feats, q_i_align_used, his_cl_used, regularization

    def inter_pred(self, user_feats, item_emb, domain, query_emb=None):
        assert domain in ["rec", "src"]

        rec_interest, src_interest, user_emb = user_feats

        if domain == "rec":
            item_emb = item_emb.reshape(-1, item_emb.size(-1))

            output = self.ple_layer(
                torch.cat([
                    rec_interest, src_interest, item_emb, user_emb,
                    self.rec_query.expand(item_emb.shape[0], -1)
                ], -1))[0]

            return self.rec_fc_layer(output)

        elif domain == "src":
            if item_emb.dim() == 3:
                [query_emb], item_emb = self.repeat_feat([query_emb], item_emb)

            output = self.ple_layer(
                torch.cat([
                    rec_interest, src_interest, item_emb, user_emb, query_emb
                ], -1))[1]
            return self.src_fc_layer(output)

    def rec_loss(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs[
            'user'], inputs['all_his'], inputs['all_his_type'], inputs[
                'item'], inputs['neg_items']

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        items_emb = self.session_embedding.get_item_emb(items)
        batch_size = items_emb.size(0)

        user_feats, q_i_align_used, his_cl_used, regularization = self.forward(
            user, all_his, all_his_type, items, items_emb, domain='rec')

        logits = self.inter_pred(user_feats, items_emb, domain="rec").reshape(
            (batch_size, -1))
        labels = torch.zeros_like(logits, dtype=torch.float32)
        labels[:, 0] = 1.0

        logits = logits.reshape((-1, ))
        labels = labels.reshape((-1, ))

        total_loss = self.loss_fn(logits, labels)
        loss_dict = {}
        loss_dict['click_loss'] = total_loss.clone()

        if self.q_i_cl_weight > 0:
            align_neg_item, align_neg_query = inputs['align_neg_item'], inputs[
                'align_neg_query']
            query_emb, click_item_mask, q_click_item_emb = q_i_align_used

            align_neg_items_emb = self.session_embedding.get_item_emb(
                align_neg_item)
            align_neg_querys_emb = self.session_embedding.get_query_emb(
                align_neg_query)
            align_loss = self.feature_alignment(
                [align_neg_items_emb, align_neg_querys_emb], query_emb,
                click_item_mask, q_click_item_emb)
            loss_dict['q_i_cl_loss'] = align_loss.clone()

            total_loss += self.q_i_cl_weight * align_loss

        if self.his_cl_weight > 0:
            src2rec, rec2rec, rec_his_mask,\
                rec2src, src2src, src_his_mask = his_cl_used
            rec_his_cl_loss = self.rec_his_cl(src2rec, rec2rec, rec_his_mask)

            src_his_cl_loss = self.src_his_cl(rec2src, src2src, src_his_mask)

            his_cl_loss = rec_his_cl_loss + src_his_cl_loss
            loss_dict['his_cl_loss'] = his_cl_loss.clone()

            total_loss += self.his_cl_weight * his_cl_loss

        loss_dict['uncertainty_reg'] = regularization['uncertainty_reg'].clone()
        loss_dict['cf_sparsity_reg'] = regularization['cf_sparsity_reg'].clone()
        loss_dict['path_competition_reg'] = regularization[
            'path_competition_reg'].clone()
        loss_dict['intent_separation_reg'] = regularization[
            'intent_separation_reg'].clone()
        loss_dict['intent_collapse_reg'] = regularization[
            'intent_collapse_reg'].clone()
        loss_dict['intent_proto_sim_mean'] = regularization[
            'intent_proto_sim_mean'].clone()
        loss_dict['intent_proto_sim_max'] = regularization[
            'intent_proto_sim_max'].clone()
        loss_dict['cf_consistency_reg'] = regularization[
            'cf_consistency_reg'].clone()
        loss_dict['intent_mu_norm'] = regularization['intent_mu_norm'].clone()
        loss_dict['intent_assign_entropy'] = regularization[
            'intent_assign_entropy'].clone()
        loss_dict['intent_usage_max'] = regularization[
            'intent_usage_max'].clone()
        loss_dict['intent_residual_mean'] = regularization[
            'intent_residual_mean'].clone()
        loss_dict['transition_explore_mean'] = regularization[
            'transition_explore_mean'].clone()
        loss_dict['transition_exploit_mean'] = regularization[
            'transition_exploit_mean'].clone()
        loss_dict['uncertainty_mean'] = regularization[
            'uncertainty_mean'].clone()
        loss_dict['uncertainty_std'] = regularization['uncertainty_std'].clone()
        loss_dict['cf_mask_mean'] = regularization['cf_mask_mean'].clone()
        loss_dict['cf_necessity_mean'] = regularization[
            'cf_necessity_mean'].clone()
        loss_dict['cf_potential_mean'] = regularization[
            'cf_potential_mean'].clone()
        loss_dict['cf_self_mean'] = regularization['cf_self_mean'].clone()
        loss_dict['path_s2s'] = regularization['path_s2s'].clone()
        loss_dict['path_r2s'] = regularization['path_r2s'].clone()
        loss_dict['path_r2r'] = regularization['path_r2r'].clone()
        loss_dict['path_s2r'] = regularization['path_s2r'].clone()
        loss_dict['rec_same_delta_mean'] = regularization[
            'rec_same_delta_mean'].clone()
        loss_dict['rec_cross_delta_mean'] = regularization[
            'rec_cross_delta_mean'].clone()
        loss_dict['src_same_delta_mean'] = regularization[
            'src_same_delta_mean'].clone()
        loss_dict['src_cross_delta_mean'] = regularization[
            'src_cross_delta_mean'].clone()
        loss_dict['rec_cross_gate_mean'] = regularization[
            'rec_cross_gate_mean'].clone()
        loss_dict['src_cross_gate_mean'] = regularization[
            'src_cross_gate_mean'].clone()
        loss_dict['attention_peak'] = regularization['attention_peak'].clone()
        loss_dict['attention_temp_mean'] = regularization[
            'attention_temp_mean'].clone()
        loss_dict['attention_temp_std'] = regularization[
            'attention_temp_std'].clone()
        total_loss += self.uncertainty_reg_weight * regularization[
            'uncertainty_reg']
        total_loss += self.cf_sparsity_weight * regularization[
            'cf_sparsity_reg']
        total_loss += self.path_competition_weight * regularization[
            'path_competition_reg']
        total_loss += self.intent_separation_weight * regularization[
            'intent_separation_reg']
        total_loss += self.intent_collapse_weight * regularization[
            'intent_collapse_reg']
        total_loss += self.cf_consistency_weight * regularization[
            'cf_consistency_reg']

        loss_dict['total_loss'] = total_loss

        return loss_dict

    def rec_predict(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs[
            'user'], inputs['all_his'], inputs['all_his_type'], inputs[
                'item'], inputs['neg_items']

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        items_emb = self.session_embedding.get_item_emb(items)
        batch_size = items_emb.size(0)

        user_feats, q_i_align_used, his_cl_used, _ = self.forward(
            user, all_his, all_his_type, items, items_emb, domain='rec')

        logits = self.inter_pred(user_feats, items_emb, domain="rec").reshape(
            (batch_size, -1))
        return logits

    def src_loss(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs[
            'user'], inputs['all_his'], inputs['all_his_type'], inputs[
                'item'], inputs['neg_items']

        query = inputs['query']
        query_emb = self.session_embedding.get_query_emb(query)

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        items_emb = self.session_embedding.get_item_emb(items)
        batch_size = items_emb.size(0)

        user_feats, q_i_align_used, his_cl_used, regularization = self.forward(
            user,
            all_his,
            all_his_type,
            items,
            items_emb,
            domain='src',
            query_emb=query_emb)

        logits = self.inter_pred(user_feats,
                                 items_emb,
                                 domain="src",
                                 query_emb=query_emb).reshape((batch_size, -1))
        labels = torch.zeros_like(logits, dtype=torch.float32)
        labels[:, 0] = 1.0

        logits = logits.reshape((-1, ))
        labels = labels.reshape((-1, ))

        total_loss = self.loss_fn(logits, labels)
        loss_dict = {}
        loss_dict['click_loss'] = total_loss.clone()

        if self.q_i_cl_weight > 0:
            align_neg_item, align_neg_query = inputs['align_neg_item'], inputs[
                'align_neg_query']
            query_emb, click_item_mask, q_click_item_emb = q_i_align_used

            align_neg_items_emb = self.session_embedding.get_item_emb(
                align_neg_item)
            align_neg_querys_emb = self.session_embedding.get_query_emb(
                align_neg_query)
            align_loss = self.feature_alignment(
                [align_neg_items_emb, align_neg_querys_emb], query_emb,
                click_item_mask, q_click_item_emb)
            loss_dict['q_i_cl_loss'] = align_loss.clone()

            total_loss += self.q_i_cl_weight * align_loss

        if self.his_cl_weight > 0:
            src2rec, rec2rec, rec_his_mask,\
                rec2src, src2src, src_his_mask = his_cl_used

            rec_his_cl_loss = self.rec_his_cl(src2rec, rec2rec, rec_his_mask)

            src_his_cl_loss = self.src_his_cl(rec2src, src2src, src_his_mask)

            his_cl_loss = rec_his_cl_loss + src_his_cl_loss
            loss_dict['his_cl_loss'] = his_cl_loss.clone()

            total_loss += self.his_cl_weight * his_cl_loss

        loss_dict['uncertainty_reg'] = regularization['uncertainty_reg'].clone()
        loss_dict['cf_sparsity_reg'] = regularization['cf_sparsity_reg'].clone()
        loss_dict['path_competition_reg'] = regularization[
            'path_competition_reg'].clone()
        loss_dict['intent_separation_reg'] = regularization[
            'intent_separation_reg'].clone()
        loss_dict['intent_collapse_reg'] = regularization[
            'intent_collapse_reg'].clone()
        loss_dict['intent_proto_sim_mean'] = regularization[
            'intent_proto_sim_mean'].clone()
        loss_dict['intent_proto_sim_max'] = regularization[
            'intent_proto_sim_max'].clone()
        loss_dict['cf_consistency_reg'] = regularization[
            'cf_consistency_reg'].clone()
        loss_dict['intent_mu_norm'] = regularization['intent_mu_norm'].clone()
        loss_dict['intent_assign_entropy'] = regularization[
            'intent_assign_entropy'].clone()
        loss_dict['intent_usage_max'] = regularization[
            'intent_usage_max'].clone()
        loss_dict['intent_residual_mean'] = regularization[
            'intent_residual_mean'].clone()
        loss_dict['transition_explore_mean'] = regularization[
            'transition_explore_mean'].clone()
        loss_dict['transition_exploit_mean'] = regularization[
            'transition_exploit_mean'].clone()
        loss_dict['uncertainty_mean'] = regularization[
            'uncertainty_mean'].clone()
        loss_dict['uncertainty_std'] = regularization['uncertainty_std'].clone()
        loss_dict['cf_mask_mean'] = regularization['cf_mask_mean'].clone()
        loss_dict['cf_necessity_mean'] = regularization[
            'cf_necessity_mean'].clone()
        loss_dict['cf_potential_mean'] = regularization[
            'cf_potential_mean'].clone()
        loss_dict['cf_self_mean'] = regularization['cf_self_mean'].clone()
        loss_dict['path_s2s'] = regularization['path_s2s'].clone()
        loss_dict['path_r2s'] = regularization['path_r2s'].clone()
        loss_dict['path_r2r'] = regularization['path_r2r'].clone()
        loss_dict['path_s2r'] = regularization['path_s2r'].clone()
        loss_dict['rec_same_delta_mean'] = regularization[
            'rec_same_delta_mean'].clone()
        loss_dict['rec_cross_delta_mean'] = regularization[
            'rec_cross_delta_mean'].clone()
        loss_dict['src_same_delta_mean'] = regularization[
            'src_same_delta_mean'].clone()
        loss_dict['src_cross_delta_mean'] = regularization[
            'src_cross_delta_mean'].clone()
        loss_dict['rec_cross_gate_mean'] = regularization[
            'rec_cross_gate_mean'].clone()
        loss_dict['src_cross_gate_mean'] = regularization[
            'src_cross_gate_mean'].clone()
        loss_dict['attention_peak'] = regularization['attention_peak'].clone()
        loss_dict['attention_temp_mean'] = regularization[
            'attention_temp_mean'].clone()
        loss_dict['attention_temp_std'] = regularization[
            'attention_temp_std'].clone()
        total_loss += self.uncertainty_reg_weight * regularization[
            'uncertainty_reg']
        total_loss += self.cf_sparsity_weight * regularization[
            'cf_sparsity_reg']
        total_loss += self.path_competition_weight * regularization[
            'path_competition_reg']
        total_loss += self.intent_separation_weight * regularization[
            'intent_separation_reg']
        total_loss += self.intent_collapse_weight * regularization[
            'intent_collapse_reg']
        total_loss += self.cf_consistency_weight * regularization[
            'cf_consistency_reg']

        loss_dict['total_loss'] = total_loss

        return loss_dict

    def src_predict(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs[
            'user'], inputs['all_his'], inputs['all_his_type'], inputs[
                'item'], inputs['neg_items']

        query = inputs['query']
        query_emb = self.session_embedding.get_query_emb(query)

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        items_emb = self.session_embedding.get_item_emb(items)
        batch_size = items_emb.size(0)

        user_feats, q_i_align_used, his_cl_used, _ = self.forward(
            user,
            all_his,
            all_his_type,
            items,
            items_emb,
            domain='src',
            query_emb=query_emb)

        logits = self.inter_pred(user_feats,
                                 items_emb,
                                 domain="src",
                                 query_emb=query_emb).reshape((batch_size, -1))
        return logits


class LatentIntentDiscovery(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 num_intents: int,
                 num_heads: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        if emb_dim % num_heads != 0:
            num_heads = 1
        self.num_intents = num_intents
        self.intent_slots = nn.Parameter(torch.randn(num_intents, emb_dim))
        nn.init.xavier_normal_(self.intent_slots)
        self.slot_attention = nn.MultiheadAttention(embed_dim=emb_dim,
                                                    num_heads=num_heads,
                                                    dropout=dropout,
                                                    batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, behavior_seq: torch.Tensor, pad_mask: torch.Tensor):
        batch_size = behavior_seq.size(0)
        slots = self.intent_slots.unsqueeze(0).expand(batch_size, -1, -1)
        safe_mask = pad_mask
        if pad_mask.any():
            safe_mask = pad_mask.clone()
            empty_rows = safe_mask.all(dim=1)
            if empty_rows.any():
                safe_mask[empty_rows, 0] = False
        intents, _ = self.slot_attention(query=slots,
                                         key=behavior_seq,
                                         value=behavior_seq,
                                         key_padding_mask=safe_mask)
        return self.norm(intents + slots)


class Target_Attention(nn.Module):
    def __init__(self, hid_dim1, hid_dim2):
        super().__init__()

        self.W = nn.Parameter(torch.randn((1, hid_dim1, hid_dim2)))
        nn.init.xavier_normal_(self.W)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seq_emb, target, mask):
        score = torch.matmul(seq_emb, self.W)
        score = torch.matmul(score, target.unsqueeze(-1))

        all_score = score.masked_fill(mask.unsqueeze(-1), -1e16)
        all_weight = self.softmax(all_score.transpose(-2, -1))
        all_weight = all_weight.masked_fill(mask.unsqueeze(1), 0.0)
        norm = all_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        all_weight = all_weight / norm
        all_vec = torch.matmul(all_weight, seq_emb).squeeze(1)
        valid_mask = (~mask).any(dim=1, keepdim=True).float()
        all_vec = all_vec * valid_mask

        return all_vec


class TransAlign(nn.Module):
    def __init__(self, batch_size, hidden_dim, device, infoNCE_temp) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.device = device

        self.infoNCE_temp = nn.Parameter(torch.ones([]) * infoNCE_temp)
        self.weight_matrix = nn.Parameter(torch.randn(
            (hidden_dim, hidden_dim)))
        nn.init.xavier_normal_(self.weight_matrix)

        self.cl_loss_func = nn.CrossEntropyLoss()
        self.mask_default = self.mask_correlated_samples(self.batch_size)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool, device=self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, same_his: torch.Tensor, diff_his: torch.Tensor,
                his_mask: torch.Tensor):
        same_his_emb = same_his.masked_fill(his_mask.unsqueeze(2), 0)
        same_his_sum = same_his_emb.sum(dim=1)
        same_his_mean = same_his_sum / \
            (~his_mask).sum(dim=1, keepdim=True).clamp_min(1)

        diff_his_emb = diff_his.masked_fill(his_mask.unsqueeze(2), 0)
        diff_his_sum = diff_his_emb.sum(dim=1)
        diff_his_mean = diff_his_sum / \
            (~his_mask).sum(dim=1, keepdim=True).clamp_min(1)

        batch_size = same_his_mean.size(0)
        N = 2 * batch_size

        z = torch.cat([same_his_mean.squeeze(),
                       diff_his_mean.squeeze()],
                      dim=0)
        sim = torch.mm(torch.mm(z, self.weight_matrix), z.T)
        sim = torch.tanh(sim) / self.infoNCE_temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        info_nce_loss = self.cl_loss_func(logits, labels)

        return info_nce_loss


class IntentAwareSelfAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout,
                 intent_assign_bias_weight, intent_mu_bias_weight,
                 uncertainty_bias_weight, explore_temp_scale,
                 exploit_temp_scale, attention_temp_min,
                 attention_temp_max) -> None:
        super().__init__()
        if emb_size % num_heads != 0:
            num_heads = 1
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.intent_assign_bias_weight = intent_assign_bias_weight
        self.intent_mu_bias_weight = intent_mu_bias_weight
        self.uncertainty_bias_weight = uncertainty_bias_weight
        self.explore_temp_scale = explore_temp_scale
        self.exploit_temp_scale = exploit_temp_scale
        self.attention_temp_min = attention_temp_min
        self.attention_temp_max = attention_temp_max

        self.q_proj = nn.Linear(emb_size, emb_size)
        self.k_proj = nn.Linear(emb_size, emb_size)
        self.v_proj = nn.Linear(emb_size, emb_size)
        self.out_proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def build_intent_bias(self, intent_mu, intent_sigma, intent_assign):
        bias = None
        if intent_assign is not None and self.intent_assign_bias_weight != 0:
            assign_sim = torch.matmul(intent_assign,
                                      intent_assign.transpose(-1, -2))
            bias = self.intent_assign_bias_weight * assign_sim

        if intent_mu is not None and self.intent_mu_bias_weight != 0:
            mu_dist = (intent_mu.unsqueeze(2) -
                       intent_mu.unsqueeze(1)).pow(2).mean(dim=-1)
            mu_bias = -self.intent_mu_bias_weight * mu_dist
            bias = mu_bias if bias is None else bias + mu_bias

        if intent_sigma is not None and self.uncertainty_bias_weight != 0:
            uncertainty = intent_sigma.mean(dim=-1).unsqueeze(1)
            uncertainty_bias = -self.uncertainty_bias_weight * uncertainty
            bias = uncertainty_bias if bias is None else bias + uncertainty_bias
        return bias

    def forward(self,
                his_emb,
                src_key_padding_mask,
                src_mask=None,
                intent_mu=None,
                intent_sigma=None,
                intent_assign=None,
                explore=None,
                exploit=None):
        batch_size, seq_len, _ = his_emb.size()
        query = self.q_proj(his_emb).reshape(
            batch_size, seq_len, self.num_heads,
            self.head_dim).transpose(1, 2)
        key = self.k_proj(his_emb).reshape(batch_size, seq_len, self.num_heads,
                                           self.head_dim).transpose(1, 2)
        value = self.v_proj(his_emb).reshape(
            batch_size, seq_len, self.num_heads,
            self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(query,
                                   key.transpose(-1, -2)) / math.sqrt(
                                       self.head_dim)
        intent_bias = self.build_intent_bias(intent_mu, intent_sigma,
                                             intent_assign)
        if intent_bias is not None:
            attn_logits = attn_logits + intent_bias.unsqueeze(1)

        attn_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
        if src_mask is not None:
            attn_mask = attn_mask | src_mask.unsqueeze(1)
        attn_logits = attn_logits.masked_fill(attn_mask, -1e16)

        if explore is not None and exploit is not None:
            temperature = 1.0 + self.explore_temp_scale * explore - \
                self.exploit_temp_scale * exploit
            temperature = temperature.clamp(min=self.attention_temp_min,
                                            max=self.attention_temp_max)
            temperature = temperature.masked_fill(src_key_padding_mask, 1.0)
            attn_logits = attn_logits / temperature.unsqueeze(1).unsqueeze(-1)

        attn_probs = torch.softmax(attn_logits, dim=-1)
        attn_probs = attn_probs.masked_fill(attn_mask, 0.0)
        attn_probs = attn_probs / attn_probs.sum(dim=-1,
                                                 keepdim=True).clamp_min(1e-8)
        attn_probs = self.dropout(attn_probs)
        output = torch.matmul(attn_probs, value)
        output = output.transpose(1, 2).reshape(batch_size, seq_len,
                                               self.emb_size)
        output = self.out_proj(output)
        return output.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)


class IntentAwareTransformerLayer(nn.Module):
    def __init__(self, emb_size, num_heads, dropout,
                 intent_assign_bias_weight, intent_mu_bias_weight,
                 uncertainty_bias_weight, explore_temp_scale,
                 exploit_temp_scale, attention_temp_min,
                 attention_temp_max) -> None:
        super().__init__()
        self.self_attn = IntentAwareSelfAttention(
            emb_size=emb_size,
            num_heads=num_heads,
            dropout=dropout,
            intent_assign_bias_weight=intent_assign_bias_weight,
            intent_mu_bias_weight=intent_mu_bias_weight,
            uncertainty_bias_weight=uncertainty_bias_weight,
            explore_temp_scale=explore_temp_scale,
            exploit_temp_scale=exploit_temp_scale,
            attention_temp_min=attention_temp_min,
            attention_temp_max=attention_temp_max)
        self.linear1 = nn.Linear(emb_size, emb_size)
        self.linear2 = nn.Linear(emb_size, emb_size)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self,
                his_emb,
                src_key_padding_mask,
                src_mask=None,
                intent_mu=None,
                intent_sigma=None,
                intent_assign=None,
                explore=None,
                exploit=None):
        attn_output = self.self_attn(his_emb, src_key_padding_mask, src_mask,
                                     intent_mu, intent_sigma, intent_assign,
                                     explore, exploit)
        his_emb = self.norm1(his_emb + self.dropout(attn_output))
        ffn_output = self.linear2(self.dropout(self.activation(
            self.linear1(his_emb))))
        his_emb = self.norm2(his_emb + self.dropout(ffn_output))
        return his_emb.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)


class Transformer(nn.Module):
    def __init__(self, emb_size, num_heads, num_layers, dropout,
                 intent_assign_bias_weight, intent_mu_bias_weight,
                 uncertainty_bias_weight, explore_temp_scale,
                 exploit_temp_scale, attention_temp_min,
                 attention_temp_max) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            IntentAwareTransformerLayer(
                emb_size=emb_size,
                num_heads=num_heads,
                dropout=dropout,
                intent_assign_bias_weight=intent_assign_bias_weight,
                intent_mu_bias_weight=intent_mu_bias_weight,
                uncertainty_bias_weight=uncertainty_bias_weight,
                explore_temp_scale=explore_temp_scale,
                exploit_temp_scale=exploit_temp_scale,
                attention_temp_min=attention_temp_min,
                attention_temp_max=attention_temp_max)
            for _ in range(num_layers)
        ])

    def forward(self,
                his_emb: torch.Tensor,
                src_key_padding_mask: torch.Tensor,
                src_mask: torch.Tensor = None,
                intent_mu: torch.Tensor = None,
                intent_sigma: torch.Tensor = None,
                intent_assign: torch.Tensor = None,
                explore: torch.Tensor = None,
                exploit: torch.Tensor = None):
        his_encoded = his_emb
        for layer in self.layers:
            his_encoded = layer(his_encoded, src_key_padding_mask, src_mask,
                                intent_mu, intent_sigma, intent_assign,
                                explore, exploit)
        return his_encoded
