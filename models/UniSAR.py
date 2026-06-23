import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from utils import const

from .BaseModel import BaseModel
from .layers import FullyConnectedLayer, feature_align, PositionalEmbedding, PLE_layer


class UniSAR(BaseModel):
    BELIEF_MEAN_KEYS = [
        'state_uncertainty_mean',
        'belief_entropy_mean',
        'belief_confidence_mean',
        'belief_variance_mean',
        'state_uncertainty_std',
        'state_uncertainty_early_mean',
        'state_uncertainty_mid_mean',
        'state_uncertainty_late_mean',
        'state_temp_mean',
        'state_temp_std',
        'state_temp_early_mean',
        'state_temp_mid_mean',
        'state_temp_late_mean',
        'belief_mass_mean',
    ]
    BELIEF_MAX_KEYS = ['state_temp_max', 'belief_mass_max']
    BELIEF_MIN_KEYS = ['state_temp_min', 'belief_mass_min']
    DIAGNOSTIC_KEYS = [
        'intent_assignment_reg',
        'intent_proto_sim_mean',
        'intent_proto_sim_max',
        'intent_proto_sim_min',
        'intent_proto_margin_violation',
        'cf_consistency_reg',
        'intent_assign_entropy',
        'intent_usage_entropy',
        'intent_confidence',
        'intent_usage_max',
        'intent_residual_mean',
        'state_uncertainty_mean',
        'belief_entropy_mean',
        'belief_confidence_mean',
        'belief_variance_mean',
        'state_uncertainty_std',
        'state_uncertainty_early_mean',
        'state_uncertainty_mid_mean',
        'state_uncertainty_late_mean',
        'state_temp_mean',
        'state_temp_std',
        'state_temp_early_mean',
        'state_temp_mid_mean',
        'state_temp_late_mean',
        'state_temp_max',
        'state_temp_min',
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
    ]

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
        parser.add_argument('--use_counterfactual', type=int, default=1)
        parser.add_argument('--intent_num', type=int, default=4)
        parser.add_argument('--intent_heads', type=int, default=2)
        parser.add_argument('--intent_dropout', type=float, default=0.1)
        parser.add_argument('--intent_assignment_weight',
                            type=float,
                            default=0.01)
        parser.add_argument('--intent_entropy_target',
                            type=float,
                            default=0.5)
        parser.add_argument('--intent_confidence_target',
                            type=float,
                            default=0.55)
        parser.add_argument('--intent_diversity_margin',
                            type=float,
                            default=0.2)
        parser.add_argument('--intent_var_min', type=float, default=1e-4)
        parser.add_argument('--rec_cross_alpha', type=float, default=0.05)
        parser.add_argument('--belief_init_var', type=float, default=1.0)
        parser.add_argument('--belief_init_mass', type=float, default=1.0)
        parser.add_argument('--belief_prior_weight', type=float, default=1.0)
        parser.add_argument('--intent_bias_scale', type=float, default=1.0)
        parser.add_argument('--state_temp_scale', type=float, default=0.5)
        parser.add_argument('--state_temp_max', type=float, default=1.5)
        return BaseModel.parse_model_args(parser)

    def __init__(self, args):
        super().__init__(args)
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.batch_size = args.batch_size
        self.intent_temp = args.intent_temp
        self.cf_gate_scale = args.cf_gate_scale
        self.cf_consistency_weight = args.cf_consistency_weight
        self.use_counterfactual = bool(args.use_counterfactual)
        self.intent_num = args.intent_num
        self.intent_assignment_weight = args.intent_assignment_weight
        self.intent_entropy_target = args.intent_entropy_target
        self.intent_confidence_target = args.intent_confidence_target
        self.intent_diversity_margin = args.intent_diversity_margin
        self.intent_var_min = args.intent_var_min
        self.rec_cross_alpha = args.rec_cross_alpha
        self.belief_init_var = args.belief_init_var
        self.belief_init_mass = args.belief_init_mass
        self.belief_prior_weight = args.belief_prior_weight
        self.intent_bias_scale = args.intent_bias_scale
        self.state_temp_scale = args.state_temp_scale
        self.state_temp_max = args.state_temp_max
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
                                           intent_bias_scale=self.intent_bias_scale,
                                           state_temp_scale=self.state_temp_scale,
                                           state_temp_max=self.state_temp_max)
        self.src_transformer = Transformer(emb_size=self.item_size,
                                           num_heads=self.num_heads,
                                           num_layers=self.num_layers,
                                           dropout=self.dropout,
                                           intent_bias_scale=self.intent_bias_scale,
                                           state_temp_scale=self.state_temp_scale,
                                           state_temp_max=self.state_temp_max)
        self.global_transformer = Transformer(emb_size=self.item_size,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_layers,
                                              dropout=self.dropout,
                                              intent_bias_scale=self.intent_bias_scale,
                                              state_temp_scale=self.state_temp_scale,
                                              state_temp_max=self.state_temp_max)

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

        if self.use_counterfactual:
            self.original_decoder_layer = None
            self.original_rec_cross_fusion = None
            self.original_src_cross_fusion = None
        else:
            self.original_decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.item_size,
                nhead=self.num_heads,
                dim_feedforward=self.item_size,
                dropout=self.dropout,
                batch_first=True)
            self.original_rec_cross_fusion = nn.TransformerDecoder(
                self.original_decoder_layer, num_layers=self.num_layers)
            self.original_src_cross_fusion = nn.TransformerDecoder(
                self.original_decoder_layer, num_layers=self.num_layers)

        self.intent_discovery = LatentIntentDiscovery(
            emb_dim=self.item_size,
            num_intents=self.intent_num,
            num_heads=args.intent_heads,
            dropout=args.intent_dropout)
        if self.use_counterfactual:
            mix_init = min(max(args.rec_cross_alpha, 1e-4), 1.0 - 1e-4)
            mix_logit = math.log(mix_init / (1.0 - mix_init))
            self.rec_src_mix = nn.Parameter(torch.tensor(float(mix_logit)))
            self.src_cross_mix = nn.Parameter(torch.tensor(float(mix_logit)))
        else:
            self.rec_src_mix = None
            self.src_cross_mix = None
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

    def compute_segment_means(self, values, valid_mask, prefix):
        seq_len = values.size(1)
        boundaries = [0, seq_len // 3, (2 * seq_len) // 3, seq_len]
        segment_names = ['early', 'mid', 'late']
        diagnostics = {}
        for idx, name in enumerate(segment_names):
            start, end = boundaries[idx], boundaries[idx + 1]
            if end <= start:
                diagnostics[f'{prefix}_{name}_mean'] = values.new_tensor(0.0)
                continue
            segment_values = values[:, start:end]
            segment_mask = valid_mask[:, start:end]
            diagnostics[f'{prefix}_{name}_mean'] = self.safe_masked_mean(
                segment_values, segment_mask)
        return diagnostics

    def compute_intent_state(self, all_his_emb, all_his_mask):
        intents = self.intent_discovery(all_his_emb, all_his_mask)
        assign_logits = torch.matmul(all_his_emb, intents.transpose(-1, -2))
        assign_logits = assign_logits / max(self.intent_temp, 1e-6)
        prior_assign = torch.softmax(assign_logits, dim=-1)
        prior_assign = prior_assign.masked_fill(all_his_mask.unsqueeze(-1), 0.0)

        intent_mu = torch.matmul(prior_assign, intents)

        residual = (all_his_emb - intent_mu).pow(2)
        assign_prob = prior_assign.clamp_min(1e-8)
        entropy = -(assign_prob * assign_prob.log()).sum(dim=-1,
                                                         keepdim=True)
        if self.intent_num > 1:
            entropy = entropy / math.log(self.intent_num)
        valid_mask = ~all_his_mask
        token_entropy = self.safe_masked_mean(entropy.squeeze(-1), valid_mask)
        token_confidence = self.safe_masked_mean(
            prior_assign.max(dim=-1).values, valid_mask)
        intent_reg, proto_sim_mean, proto_sim_max, proto_sim_min, \
            proto_margin_violation, usage_entropy = \
            self.compute_intent_regularization(intents, prior_assign, valid_mask,
                                               token_entropy,
                                               token_confidence)
        diagnostics = {
            'intent_assignment_reg':
            intent_reg,
            'intent_proto_sim_mean':
            proto_sim_mean,
            'intent_proto_sim_max':
            proto_sim_max,
            'intent_proto_sim_min':
            proto_sim_min,
            'intent_proto_margin_violation':
            proto_margin_violation,
            'intent_assign_entropy':
            token_entropy,
            'intent_usage_entropy':
            usage_entropy,
            'intent_confidence':
            token_confidence,
            'intent_usage_max':
            self.compute_intent_usage(prior_assign, valid_mask).max(),
            'intent_residual_mean':
            self.safe_masked_mean(residual.mean(dim=-1), valid_mask)
        }
        return intents, prior_assign, diagnostics

    def compute_belief_dynamics(self, seq_emb, intents, prior_assign, mask):
        batch_size, seq_len, emb_dim = seq_emb.size()
        posterior_trace = seq_emb.new_zeros(batch_size, seq_len,
                                            self.intent_num)
        state_uncertainty = seq_emb.new_zeros(batch_size, seq_len)
        entropy_trace = seq_emb.new_zeros(batch_size, seq_len)
        confidence_trace = seq_emb.new_zeros(batch_size, seq_len)
        var_trace = seq_emb.new_zeros(batch_size, seq_len)
        temp_trace = seq_emb.new_zeros(batch_size, seq_len)

        mu = intents.clone()
        var = seq_emb.new_full((batch_size, self.intent_num, emb_dim),
                               self.belief_init_var).clamp_min(
                                   self.intent_var_min)
        mass = seq_emb.new_full((batch_size, self.intent_num),
                                self.belief_init_mass)
        valid = ~mask

        for t in range(seq_len):
            valid_t = valid[:, t]
            if not valid_t.any():
                continue

            token_state = seq_emb[:, t, :].unsqueeze(1)
            delta = token_state - mu
            assignment_cost = (
                delta.pow(2) / var.clamp_min(self.intent_var_min)).mean(dim=-1)
            log_prior = torch.log(prior_assign[:, t, :].clamp_min(1e-8))
            scores = -0.5 * assignment_cost + \
                self.belief_prior_weight * log_prior
            posterior = torch.softmax(scores, dim=-1)
            posterior = posterior * valid_t.unsqueeze(-1).float()
            posterior = posterior / posterior.sum(dim=-1,
                                                  keepdim=True).clamp_min(1e-8)
            posterior_trace[:, t, :] = posterior

            entropy_unc = -(posterior.clamp_min(1e-8) *
                            posterior.clamp_min(1e-8).log()).sum(dim=-1)
            if self.intent_num > 1:
                entropy_unc = entropy_unc / math.log(self.intent_num)
            entropy_unc = entropy_unc.clamp(0.0, 1.0)
            confidence = (1.0 - entropy_unc).clamp(0.0, 1.0)
            state_temp = (1.0 + self.state_temp_scale * entropy_unc).clamp(
                min=1.0, max=self.state_temp_max)

            state_uncertainty[:, t] = entropy_unc * valid_t.float()
            entropy_trace[:, t] = entropy_unc * valid_t.float()
            confidence_trace[:, t] = confidence * valid_t.float()
            var_trace[:, t] = (posterior * var.mean(dim=-1)).sum(dim=-1) * \
                valid_t.float()
            temp_trace[:, t] = state_temp * valid_t.float()

            posterior_update = posterior.unsqueeze(-1)
            token_state_sq = token_state.pow(2)
            prev_mass = mass
            prev_mu = mu
            prev_var = var
            mass = mass + posterior
            mass_safe = mass.unsqueeze(-1).clamp_min(1e-8)
            weighted_token = posterior_update * token_state
            weighted_token_sq = posterior_update * token_state_sq
            mu = (prev_mass.unsqueeze(-1) * prev_mu +
                  weighted_token) / mass_safe
            second_old = prev_var + prev_mu.pow(2)
            second = (prev_mass.unsqueeze(-1) * second_old +
                      weighted_token_sq) / mass_safe
            var = (second - mu.pow(2)).clamp_min(self.intent_var_min)

        diagnostics = {
            'state_uncertainty_mean':
            self.safe_masked_mean(state_uncertainty, valid),
            'state_uncertainty_std':
            self.safe_masked_std(state_uncertainty, valid),
            'belief_entropy_mean':
            self.safe_masked_mean(entropy_trace, valid),
            'belief_confidence_mean':
            self.safe_masked_mean(confidence_trace, valid),
            'belief_variance_mean':
            self.safe_masked_mean(var_trace, valid),
            'state_temp_mean':
            self.safe_masked_mean(temp_trace, valid),
            'state_temp_std':
            self.safe_masked_std(temp_trace, valid),
            'state_temp_max':
            temp_trace.masked_select(valid).max()
            if valid.any() else seq_emb.new_tensor(1.0),
            'state_temp_min':
            temp_trace.masked_select(valid).min()
            if valid.any() else seq_emb.new_tensor(1.0),
            'belief_mass_mean':
            mass.mean(),
            'belief_mass_max':
            mass.max(),
            'belief_mass_min':
            mass.min()
        }
        diagnostics.update(
            self.compute_segment_means(state_uncertainty, valid,
                                       'state_uncertainty'))
        diagnostics.update(
            self.compute_segment_means(temp_trace, valid, 'state_temp'))
        posterior_trace = posterior_trace.masked_fill(mask.unsqueeze(-1), 0.0)
        return posterior_trace, state_uncertainty.masked_fill(mask, 0.0), diagnostics

    def compute_intent_usage(self, assign, valid_mask):
        valid_count = valid_mask.float().sum()
        if valid_count <= 0:
            return assign.new_full((self.intent_num, ), 1.0 / self.intent_num)
        token_weights = valid_mask.float().unsqueeze(-1)
        usage = (assign * token_weights).sum(dim=(0, 1))
        usage = usage / valid_count.clamp_min(1.0)
        return usage / usage.sum().clamp_min(1e-8)

    def compute_intent_regularization(self, intents, assign, valid_mask,
                                      token_entropy, token_confidence):
        zero = intents.new_tensor(0.0)
        if self.intent_num <= 1:
            return zero, zero, zero, zero, zero, zero
        norm_intents = F.normalize(intents, dim=-1)
        proto_sim = torch.matmul(norm_intents,
                                 norm_intents.transpose(-1, -2))
        off_diag_mask = ~torch.eye(self.intent_num,
                                   dtype=torch.bool,
                                   device=intents.device).unsqueeze(0)
        off_diag_sim = proto_sim.masked_select(off_diag_mask)
        proto_sim_mean = off_diag_sim.mean()
        proto_sim_max = off_diag_sim.max()
        proto_sim_min = off_diag_sim.min()
        proto_violation = F.relu(proto_sim - self.intent_diversity_margin)
        proto_loss = proto_violation.masked_select(off_diag_mask).mean()
        proto_margin_violation = proto_violation.masked_select(
            off_diag_mask).mean()

        valid_count = valid_mask.float().sum()
        if valid_count <= 0:
            return proto_loss, proto_sim_mean, proto_sim_max, proto_sim_min, \
                proto_margin_violation, zero
        usage = self.compute_intent_usage(assign, valid_mask)
        usage_entropy = -(usage.clamp_min(1e-8) *
                          usage.clamp_min(1e-8).log()).sum()
        if self.intent_num > 1:
            usage_entropy = usage_entropy / math.log(self.intent_num)
        usage_loss = 1.0 - usage_entropy
        entropy_target = float(min(max(self.intent_entropy_target, 0.0), 1.0))
        confidence_target = float(
            min(max(self.intent_confidence_target, 0.0), 1.0))
        assignment_loss = F.relu(token_entropy - entropy_target) + \
            F.relu(confidence_target - token_confidence)
        return assignment_loss + proto_loss + usage_loss, proto_sim_mean, \
            proto_sim_max, proto_sim_min, proto_margin_violation, \
            usage_entropy

    def aggregate_belief_diagnostics(self, diagnostics):
        aggregate = {}
        for key in self.BELIEF_MEAN_KEYS:
            aggregate[key] = sum(d[key] for d in diagnostics) / len(diagnostics)
        for key in self.BELIEF_MAX_KEYS:
            aggregate[key] = torch.stack([d[key] for d in diagnostics]).max()
        for key in self.BELIEF_MIN_KEYS:
            aggregate[key] = torch.stack([d[key] for d in diagnostics]).min()
        return aggregate

    def build_regularization(self, intent_diagnostics, belief_diagnostics,
                             device):
        zero = torch.tensor(0.0, device=device)
        regularization = {
            'intent_assignment_reg':
            intent_diagnostics['intent_assignment_reg'],
            'intent_proto_sim_mean':
            intent_diagnostics['intent_proto_sim_mean'],
            'intent_proto_sim_max':
            intent_diagnostics['intent_proto_sim_max'],
            'intent_proto_sim_min':
            intent_diagnostics['intent_proto_sim_min'],
            'intent_proto_margin_violation':
            intent_diagnostics['intent_proto_margin_violation'],
            'intent_assign_entropy':
            intent_diagnostics['intent_assign_entropy'],
            'intent_usage_entropy':
            intent_diagnostics['intent_usage_entropy'],
            'intent_confidence':
            intent_diagnostics['intent_confidence'],
            'intent_usage_max':
            intent_diagnostics['intent_usage_max'],
            'intent_residual_mean':
            intent_diagnostics['intent_residual_mean'],
            'cf_consistency_reg':
            zero
        }
        regularization.update(self.aggregate_belief_diagnostics(
            belief_diagnostics))
        for key in [
                'cf_mask_mean', 'cf_necessity_mean', 'cf_potential_mean',
                'cf_self_mean', 'rec_mix_mean', 'src_mix_mean',
                'rec_same_delta_mean', 'rec_cross_delta_mean',
                'src_same_delta_mean', 'src_cross_delta_mean',
                'rec_cross_gate_mean', 'src_cross_gate_mean',
                'cross_mix_effective_mean'
        ]:
            regularization[key] = zero
        return regularization

    def compute_counterfactual_gates(self, full_pred, wo_cross_pred,
                                     wo_same_pred):
        cross_delta = F.relu(full_pred - wo_cross_pred).squeeze(-1)
        same_delta = F.relu(full_pred - wo_same_pred).squeeze(-1)
        gate_logits = self.cf_gate_scale * torch.stack([same_delta, cross_delta],
                                                       dim=-1)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        return gate_probs[:, 0], gate_probs[:, 1]

    def compute_cross_supplement_gates(self, full_pred, wo_cross_pred,
                                       wo_same_pred):
        cross_delta = F.relu(full_pred - wo_cross_pred).squeeze(-1)
        same_delta = F.relu(full_pred - wo_same_pred).squeeze(-1)
        gate_logits = self.cf_gate_scale * torch.stack([same_delta, cross_delta],
                                                       dim=-1)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        cross_gate = gate_probs[:, 1]
        same_gate = torch.ones_like(cross_gate)
        return same_gate, cross_gate

    def apply_original_cross_fusion(self, rec2rec, src2rec, rec_his_mask,
                                    src2src, rec2src, src_his_mask, user_emb,
                                    items_emb):
        rec_fusion_decoded = self.original_rec_cross_fusion(
            tgt=rec2rec,
            memory=src2rec,
            tgt_key_padding_mask=rec_his_mask,
            memory_key_padding_mask=rec_his_mask)
        src_fusion_decoded = self.original_src_cross_fusion(
            tgt=src2src,
            memory=rec2src,
            tgt_key_padding_mask=src_his_mask,
            memory_key_padding_mask=src_his_mask)

        if items_emb.dim() == 3:
            feature_list = [
                rec_fusion_decoded, rec_his_mask, src_fusion_decoded,
                src_his_mask, user_emb
            ]
            repeat_feature_list, items_emb = self.repeat_feat(
                feature_list, items_emb)
            rec_fusion_decoded, rec_his_mask, src_fusion_decoded, \
                src_his_mask, user_emb = repeat_feature_list

        rec_fusion = self.rec_his_attn_pooling(rec_fusion_decoded, items_emb,
                                               rec_his_mask)
        src_fusion = self.src_his_attn_pooling(src_fusion_decoded, items_emb,
                                               src_his_mask)
        return [rec_fusion, src_fusion, user_emb]

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

        all_intents, all_assign, intent_diagnostics = self.compute_intent_state(
            all_his_emb, all_his_mask)
        rec_his_emb, src_his_emb = self.split_rec_src(all_his_emb,
                                                      all_his_type)
        rec_intents, rec_prior_assign, _ = \
            self.compute_intent_state(rec_his_emb, rec_his_mask)
        src_intents, src_prior_assign, _ = \
            self.compute_intent_state(src_his_emb, src_his_mask)
        global_posterior, global_state_uncertainty, global_belief_diagnostics = \
            self.compute_belief_dynamics(all_his_emb, all_intents, all_assign,
                                         all_his_mask)
        rec_posterior, rec_state_uncertainty, rec_belief_diagnostics = \
            self.compute_belief_dynamics(rec_his_emb, rec_intents,
                                         rec_prior_assign, rec_his_mask)
        src_posterior, src_state_uncertainty, src_belief_diagnostics = \
            self.compute_belief_dynamics(src_his_emb, src_intents,
                                         src_prior_assign, src_his_mask)

        all_his_emb_w_pos = all_his_emb + self.global_pos_emb(all_his_emb)

        global_mask = all_his_type[:, :, None] == all_his_type[:, None, :]

        global_encoded = self.global_transformer(
            all_his_emb_w_pos,
            all_his_mask,
            global_mask,
            state_uncertainty=global_state_uncertainty,
            intent_assign=global_posterior)
        cross_valid = (~global_mask) & (~all_his_mask).unsqueeze(1)
        has_cross_source = cross_valid.any(dim=-1)
        global_encoded = global_encoded.masked_fill(
            (~has_cross_source).unsqueeze(-1), 0.0)
        src2rec, rec2src = self.split_rec_src(global_encoded, all_his_type)

        rec_his_emb_w_pos = rec_his_emb + self.rec_pos(rec_his_emb)
        src_his_emb_w_pos = src_his_emb + self.src_pos(src_his_emb)

        rec2rec = self.rec_transformer(rec_his_emb_w_pos,
                                       rec_his_mask,
                                       state_uncertainty=rec_state_uncertainty,
                                       intent_assign=rec_posterior)
        src2src = self.src_transformer(src_his_emb_w_pos,
                                       src_his_mask,
                                       state_uncertainty=src_state_uncertainty,
                                       intent_assign=src_posterior)

        his_cl_used = [
            src2rec, rec2rec, rec_his_mask, rec2src, src2src, src_his_mask
        ]

        regularization = self.build_regularization(
            intent_diagnostics, [
                global_belief_diagnostics, rec_belief_diagnostics,
                src_belief_diagnostics
            ], all_his.device)

        if not self.use_counterfactual:
            regularization['cf_mask_mean'] = regularization[
                'cf_mask_mean'].new_tensor(0.5)
            user_feats = self.apply_original_cross_fusion(
                rec2rec, src2rec, rec_his_mask, src2src, rec2src,
                src_his_mask, user_emb, items_emb)
            return user_feats, q_i_align_used, his_cl_used, regularization

        feature_list = [
            rec2rec, src2rec, rec_his_mask, src2src, rec2src, src_his_mask,
            user_emb
        ]
        if domain == 'src':
            assert query_emb is not None
            feature_list.append(query_emb)
        repeat_feature_list, flat_items_emb = self.repeat_feat(
            feature_list, items_emb)

        if domain == 'rec':
            rec2rec, src2rec, rec_his_mask, src2src, rec2src, src_his_mask, \
                user_emb = repeat_feature_list
        else:
            rec2rec, src2rec, rec_his_mask, src2src, rec2src, src_his_mask, \
                user_emb, repeated_query_emb = repeat_feature_list

        rec_full_seq = torch.cat([rec2rec, src2rec], dim=1)
        rec_full_mask = torch.cat([rec_his_mask, src_his_mask], dim=1)
        src_full_seq = torch.cat([src2src, rec2src], dim=1)
        src_full_mask = torch.cat([src_his_mask, rec_his_mask], dim=1)

        rec_same_only = self.rec_his_attn_pooling(rec2rec, flat_items_emb,
                                                  rec_his_mask)
        rec_cross_only = self.rec_his_attn_pooling(src2rec, flat_items_emb,
                                                   src_his_mask)
        rec_full = self.rec_his_attn_pooling(rec_full_seq, flat_items_emb,
                                             rec_full_mask)
        src_same_only = self.src_his_attn_pooling(src2src, flat_items_emb,
                                                  src_his_mask)
        src_cross_only = self.src_his_attn_pooling(rec2src, flat_items_emb,
                                                   rec_his_mask)
        src_full = self.src_his_attn_pooling(src_full_seq, flat_items_emb,
                                             src_full_mask)

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

        rec_same_gate, rec_cross_gate = self.compute_cross_supplement_gates(
            rec_full_pred, rec_wo_cross_pred, rec_wo_same_pred)
        src_same_gate, src_cross_gate = self.compute_counterfactual_gates(
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

        regularization['cf_mask_mean'] = 0.5 * (
            rec_cross_gate.mean() + src_cross_gate.mean())
        regularization['cf_necessity_mean'] = rec_cross_gate.mean()
        regularization['cf_potential_mean'] = src_cross_gate.mean()
        regularization['cf_self_mean'] = 0.5 * (
            rec_same_gate.mean() + src_same_gate.mean())
        regularization['rec_same_delta_mean'] = rec_same_delta.mean()
        regularization['rec_cross_delta_mean'] = rec_cross_delta.mean()
        regularization['src_same_delta_mean'] = src_same_delta.mean()
        regularization['src_cross_delta_mean'] = src_cross_delta.mean()
        regularization['rec_cross_gate_mean'] = rec_cross_gate.mean()
        regularization['src_cross_gate_mean'] = src_cross_gate.mean()
        regularization['rec_cross_gate_pos'] = rec_cross_gate.reshape(
            items_emb.size(0), -1)[:, 0]
        regularization['src_cross_gate_pos'] = src_cross_gate.reshape(
            items_emb.size(0), -1)[:, 0]
        regularization['rec_same_delta_pos'] = rec_same_delta.reshape(
            items_emb.size(0), -1)[:, 0]
        regularization['rec_cross_delta_pos'] = rec_cross_delta.reshape(
            items_emb.size(0), -1)[:, 0]
        regularization['src_same_delta_pos'] = src_same_delta.reshape(
            items_emb.size(0), -1)[:, 0]
        regularization['src_cross_delta_pos'] = src_cross_delta.reshape(
            items_emb.size(0), -1)[:, 0]
        regularization['cf_consistency_reg'] = 0.5 * (rec_consistency +
                                                      src_consistency)

        rec_mix = torch.sigmoid(self.rec_src_mix)
        src_mix = torch.sigmoid(self.src_cross_mix)
        rec_cross_candidate = rec_same_only + rec_cross_gate.unsqueeze(
            -1) * (rec_cross_only - rec_same_only)
        src_cross_candidate = src_same_only + src_cross_gate.unsqueeze(
            -1) * (src_cross_only - src_same_only)
        rec_fusion = (1.0 - rec_mix) * rec_same_only + rec_mix * \
            rec_cross_candidate
        src_fusion = (1.0 - src_mix) * src_same_only + src_mix * \
            src_cross_candidate
        regularization['rec_mix_mean'] = rec_mix
        regularization['src_mix_mean'] = src_mix
        regularization['cross_mix_effective_mean'] = 0.5 * (
            (rec_mix * rec_cross_gate).mean() +
            (src_mix * src_cross_gate).mean())

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

    def add_auxiliary_losses(self, inputs, q_i_align_used, his_cl_used,
                             loss_dict, total_loss):
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

        return total_loss

    def finalize_loss_dict(self, loss_dict, total_loss, regularization):
        for key in self.DIAGNOSTIC_KEYS:
            loss_dict[key] = regularization[key].clone()

        total_loss += self.intent_assignment_weight * regularization[
            'intent_assignment_reg']
        if self.use_counterfactual:
            total_loss += self.cf_consistency_weight * regularization[
                'cf_consistency_reg']
        loss_dict['total_loss'] = total_loss
        return loss_dict

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
        total_loss = self.add_auxiliary_losses(
            inputs, q_i_align_used, his_cl_used, loss_dict, total_loss)
        return self.finalize_loss_dict(loss_dict, total_loss, regularization)

    def rec_predict(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs[
            'user'], inputs['all_his'], inputs['all_his_type'], inputs[
                'item'], inputs['neg_items']

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        items_emb = self.session_embedding.get_item_emb(items)
        batch_size = items_emb.size(0)

        user_feats, _, _, _ = self.forward(
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
        total_loss = self.add_auxiliary_losses(
            inputs, q_i_align_used, his_cl_used, loss_dict, total_loss)
        return self.finalize_loss_dict(loss_dict, total_loss, regularization)

    def src_predict(self, inputs):
        user, all_his, all_his_type, pos_item, neg_items = inputs[
            'user'], inputs['all_his'], inputs['all_his_type'], inputs[
                'item'], inputs['neg_items']

        query = inputs['query']
        query_emb = self.session_embedding.get_query_emb(query)

        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)
        items_emb = self.session_embedding.get_item_emb(items)
        batch_size = items_emb.size(0)

        user_feats, _, _, _ = self.forward(
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

        z = torch.cat([same_his_mean, diff_his_mean], dim=0)
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
    def __init__(self, emb_size, num_heads, dropout, intent_bias_scale,
                 state_temp_scale, state_temp_max) -> None:
        super().__init__()
        if emb_size % num_heads != 0:
            num_heads = 1
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.intent_bias_scale = intent_bias_scale
        self.state_temp_scale = state_temp_scale
        self.state_temp_max = state_temp_max

        self.q_proj = nn.Linear(emb_size, emb_size)
        self.k_proj = nn.Linear(emb_size, emb_size)
        self.v_proj = nn.Linear(emb_size, emb_size)
        self.out_proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                his_emb,
                src_key_padding_mask,
                src_mask=None,
                state_uncertainty=None,
                intent_assign=None):
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

        if intent_assign is not None and self.intent_bias_scale != 0:
            intent_sim = torch.matmul(intent_assign,
                                      intent_assign.transpose(-1, -2))
            intent_center = 1.0 / max(intent_assign.size(-1), 1)
            intent_bias = intent_sim - intent_center
            if state_uncertainty is not None:
                state_confidence = (1.0 - state_uncertainty).clamp(0.0, 1.0)
                pair_confidence = torch.sqrt(
                    state_confidence.unsqueeze(-1) *
                    state_confidence.unsqueeze(1))
                intent_bias = intent_bias * pair_confidence
            intent_bias = intent_bias * self.intent_bias_scale
            attn_logits = attn_logits + intent_bias.unsqueeze(1)

        attn_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
        if src_mask is not None:
            attn_mask = attn_mask | src_mask.unsqueeze(1)
        fully_masked_rows = attn_mask.all(dim=-1, keepdim=True)
        attn_logits = attn_logits.masked_fill(attn_mask, -1e16)
        attn_logits = attn_logits.masked_fill(fully_masked_rows, 0.0)

        if state_uncertainty is not None:
            temperature = (1.0 + self.state_temp_scale *
                           state_uncertainty.clamp(0.0, 1.0)).clamp(
                               min=1.0, max=self.state_temp_max)
            temperature = temperature.masked_fill(src_key_padding_mask,
                                                  1.0)
            attn_logits = attn_logits / temperature.unsqueeze(1).unsqueeze(-1)
            attn_logits = attn_logits.masked_fill(fully_masked_rows, 0.0)

        attn_probs = torch.softmax(attn_logits, dim=-1)
        attn_probs = attn_probs.masked_fill(attn_mask, 0.0)
        attn_probs = self.dropout(attn_probs)
        output = torch.matmul(attn_probs, value)
        output = output.transpose(1, 2).reshape(batch_size, seq_len,
                                               self.emb_size)
        output = self.out_proj(output)
        return output.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)


class IntentAwareTransformerLayer(nn.Module):
    def __init__(self, emb_size, num_heads, dropout, intent_bias_scale,
                 state_temp_scale, state_temp_max) -> None:
        super().__init__()
        self.self_attn = IntentAwareSelfAttention(
            emb_size=emb_size,
            num_heads=num_heads,
            dropout=dropout,
            intent_bias_scale=intent_bias_scale,
            state_temp_scale=state_temp_scale,
            state_temp_max=state_temp_max)
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
                state_uncertainty=None,
                intent_assign=None):
        attn_output = self.self_attn(his_emb, src_key_padding_mask, src_mask,
                                     state_uncertainty, intent_assign)
        his_emb = self.norm1(his_emb + self.dropout(attn_output))
        ffn_output = self.linear2(self.dropout(self.activation(
            self.linear1(his_emb))))
        his_emb = self.norm2(his_emb + self.dropout(ffn_output))
        return his_emb.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)


class Transformer(nn.Module):
    def __init__(self, emb_size, num_heads, num_layers, dropout,
                 intent_bias_scale, state_temp_scale, state_temp_max) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            IntentAwareTransformerLayer(
                emb_size=emb_size,
                num_heads=num_heads,
                dropout=dropout,
                intent_bias_scale=intent_bias_scale,
                state_temp_scale=state_temp_scale,
                state_temp_max=state_temp_max)
            for _ in range(num_layers)
        ])

    def forward(self,
                his_emb: torch.Tensor,
                src_key_padding_mask: torch.Tensor,
                src_mask: torch.Tensor = None,
                state_uncertainty: torch.Tensor = None,
                intent_assign: torch.Tensor = None):
        his_encoded = his_emb
        for layer in self.layers:
            his_encoded = layer(his_encoded, src_key_padding_mask, src_mask,
                                state_uncertainty, intent_assign)
        return his_encoded
