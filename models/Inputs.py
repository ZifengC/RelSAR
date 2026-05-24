import torch
import torch.nn as nn

from utils import const


def _sanitize_index(index: torch.Tensor, vocab_size: int):
    if torch.is_floating_point(index):
        index = torch.nan_to_num(index, nan=0.0, posinf=0.0, neginf=0.0)
    index = index.long()
    if vocab_size is not None and vocab_size > 0:
        index = index.clamp(min=0, max=vocab_size - 1)
    else:
        index = index.clamp_min(0)
    return index


def _repair_embedding_weight(emb: nn.Embedding):
    if torch.isfinite(emb.weight).all():
        return False
    with torch.no_grad():
        emb.weight.data = torch.nan_to_num(emb.weight.data,
                                           nan=0.0,
                                           posinf=0.0,
                                           neginf=0.0)
        if emb.weight.data.size(0) > 0:
            emb.weight.data[0, :] = 0
    return True


def _repair_linear_weight(linear: nn.Linear):
    changed = False
    if not torch.isfinite(linear.weight).all():
        with torch.no_grad():
            linear.weight.data = torch.nan_to_num(linear.weight.data,
                                                  nan=0.0,
                                                  posinf=0.0,
                                                  neginf=0.0)
        changed = True
    if linear.bias is not None and not torch.isfinite(linear.bias).all():
        with torch.no_grad():
            linear.bias.data = torch.nan_to_num(linear.bias.data,
                                                nan=0.0,
                                                posinf=0.0,
                                                neginf=0.0)
        changed = True
    return changed


def _apply_blacklist(index: torch.Tensor, blacklist: set):
    if not blacklist:
        return index
    blocked = torch.tensor(list(blacklist), device=index.device, dtype=index.dtype)
    if blocked.numel() == 0:
        return index
    mask = torch.isin(index, blocked)
    if mask.any():
        index = index.clone()
        index[mask] = 0
    return index


class UserFeat(nn.Module):
    def __init__(self, map_vocab=None) -> None:
        super().__init__()
        self.map_vocab = map_vocab

        self.attr_ls = const.user_feature_list
        self.size = 0
        for attr in self.attr_ls:
            setattr(
                self, f'{attr}_emb',
                nn.Embedding(num_embeddings=getattr(const, f'{attr}_num'),
                             embedding_dim=getattr(const, f'{attr}_dim')))
            nn.init.xavier_normal_(getattr(self, f'{attr}_emb').weight.data)
            self.size += getattr(const, f'{attr}_dim')

        self.user_trans = nn.Linear(self.size, const.final_emb_size)

    def forward(self, sample):
        feats_ls = []
        for attr in self.attr_ls:
            if attr == 'user_id':
                index = sample
            else:
                index = self.map_vocab[attr][sample]

            index = _sanitize_index(index, getattr(const, f'{attr}_num'))
            emb_layer = getattr(self, f'{attr}_emb')
            _repair_embedding_weight(emb_layer)
            emb = emb_layer(index)
            if not torch.isfinite(emb).all():
                emb = torch.zeros_like(emb)
            feats_ls.append(emb)

        _repair_linear_weight(self.user_trans)
        out = torch.tanh(self.user_trans(torch.cat(feats_ls, dim=-1))).clone()
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class ItemFeat(nn.Module):
    def __init__(self, query_feat, map_vocab=None):
        super().__init__()

        self.map_vocab = map_vocab

        self.attr_ls = const.item_feature_list
        self.size = 0
        for attr in self.attr_ls:
            if attr in const.item_text_feature:
                setattr(self, f'{attr}_emb', query_feat)
                self.caption_id_emb = query_feat
                self.size += query_feat.size
            else:
                setattr(
                    self, f'{attr}_emb',
                    nn.Embedding(num_embeddings=getattr(const, f'{attr}_num'),
                                 embedding_dim=getattr(const, f'{attr}_dim'),
                                 padding_idx=0))
                nn.init.xavier_normal_(
                    getattr(self, f'{attr}_emb').weight.data)
                getattr(self, f'{attr}_emb').weight.data[0, :] = 0
                self.size += getattr(const, f'{attr}_dim')

        self.item_trans = nn.Linear(self.size, const.final_emb_size)
        self.bad_item_ids = set()

    def forward(self, sample):
        new_sample = sample.reshape((-1, ))
        result_emb = torch.zeros((new_sample.shape[0], const.final_emb_size),
                                 device=sample.device)
        sub_mask = new_sample != 0
        if sub_mask.sum() > 0:
            sub_sample = new_sample[sub_mask]

            feats_ls = []
            for attr in self.attr_ls:
                if attr == 'item_id':
                    index = sub_sample
                else:
                    index = self.map_vocab[attr][sub_sample]

                index = _sanitize_index(index, getattr(const, f'{attr}_num', None))
                if attr == 'item_id':
                    index = _apply_blacklist(index, self.bad_item_ids)

                emb_module = getattr(self, f'{attr}_emb')
                if isinstance(emb_module, nn.Embedding):
                    _repair_embedding_weight(emb_module)
                emb = emb_module(index)
                if not torch.isfinite(emb).all():
                    if attr == 'item_id':
                        bad_rows = ~torch.isfinite(emb).all(dim=-1)
                        if bad_rows.any():
                            bad_ids = index[bad_rows].detach().unique().tolist()
                            self.bad_item_ids.update(int(v) for v in bad_ids if int(v) > 0)
                    emb = torch.zeros_like(emb)
                feats_ls.append(emb)
            raw_feat = torch.cat(feats_ls, dim=-1)
            raw_feat = torch.nan_to_num(raw_feat, nan=0.0, posinf=0.0, neginf=0.0)
            _repair_linear_weight(self.item_trans)
            sub_sample_emb = torch.tanh(self.item_trans(raw_feat)).clone()
            sub_sample_emb = torch.nan_to_num(sub_sample_emb,
                                              nan=0.0,
                                              posinf=0.0,
                                              neginf=0.0)
            valid_feat_mask = raw_feat.abs().sum(dim=-1, keepdim=True) > 0
            sub_sample_emb = sub_sample_emb * valid_feat_mask.float()
            result_emb[sub_mask] = sub_sample_emb
        result_emb = result_emb.reshape((*sample.shape, const.final_emb_size))
        return result_emb


class QueryEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(num_embeddings=const.word_id_num,
                                           embedding_dim=const.word_id_dim,
                                           padding_idx=0)
        nn.init.xavier_normal_(self.word_embedding.weight.data)
        self.bad_word_ids = set()

    def forward(self, seqs):
        seqs = _sanitize_index(seqs, const.word_id_num)
        seqs = _apply_blacklist(seqs, self.bad_word_ids)
        _repair_embedding_weight(self.word_embedding)
        seqs_mask = (seqs == 0)
        output = self.word_embedding(seqs)
        if not torch.isfinite(output).all():
            bad_pos = ~torch.isfinite(output).all(dim=-1)
            if bad_pos.any():
                bad_ids = seqs[bad_pos].detach().unique().tolist()
                self.bad_word_ids.update(int(v) for v in bad_ids if int(v) > 0)
            output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
            output = output.masked_fill(bad_pos.unsqueeze(-1), 0.0)
        else:
            output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

        seqs_len = (~seqs_mask).sum(1, keepdim=True)
        output = output.masked_fill(seqs_mask.unsqueeze(2), 0)
        sum_emb = output.sum(dim=1)
        mean_emb = sum_emb / seqs_len.clamp_min(1)

        mean_emb = mean_emb.masked_fill(seqs_len == 0, 0)

        return mean_emb.squeeze()


class QueryFeat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_encoder = QueryEncoder()

        self.size = const.word_id_dim
        self.query_trans = nn.Linear(self.size, const.final_emb_size)
        self.size = const.final_emb_size

    def forward(self, sample: torch.Tensor):
        flat_sample = sample.reshape((-1, const.max_query_word_len))
        query_emb: torch.Tensor = self.query_encoder(flat_sample)
        query_emb = query_emb.reshape((*sample.shape[:-1], -1))
        _repair_linear_weight(self.query_trans)
        query_out = torch.tanh(self.query_trans(query_emb)).clone()
        query_out = torch.nan_to_num(query_out, nan=0.0, posinf=0.0, neginf=0.0)
        valid_query_mask = flat_sample.ne(0).any(dim=-1).reshape(
            (*sample.shape[:-1], 1))
        query_out = query_out * valid_query_mask.float()

        return query_out


class SrcSessionFeat(nn.Module):
    def __init__(self,
                 query_feat,
                 item_feat,
                 user_feat,
                 map_vocab=None) -> None:
        super().__init__()
        self.query_feat = query_feat
        self.item_feat = item_feat
        self.user_feat = user_feat

        self.map_vocab = map_vocab

    def get_user_emb(self, sample):
        return self.user_feat(sample)

    def get_item_emb(self, sample):
        return self.item_feat(sample)

    def get_query_emb(self, sample):
        return self.query_feat(sample)

    def forward(self, sample):
        new_sample = sample.reshape((-1, ))
        sub_mask = new_sample != 0

        result_query_emb = torch.zeros(
            (new_sample.shape[0], const.final_emb_size), device=sample.device)
        result_item_emb = torch.zeros(
            (new_sample.shape[0], const.max_session_item_len,
             const.final_emb_size),
            device=sample.device)
        result_item_mask = torch.zeros(
            (new_sample.shape[0], const.max_session_item_len),
            device=sample.device).bool()

        if sub_mask.sum() > 0:
            sub_sample = new_sample[sub_mask]
            sub_query_id = self.map_vocab['keyword'][sub_sample]
            sub_click_item_ls = self.map_vocab['pos_items'][sub_sample]
            sub_query_emb = self.get_query_emb(sub_query_id)
            sub_click_item_mask = torch.where(sub_click_item_ls == 0, 0,
                                              1).bool()
            sub_click_item_emb = self.get_item_emb(sub_click_item_ls)

            result_query_emb[sub_mask] = sub_query_emb
            result_item_emb[sub_mask] = sub_click_item_emb
            result_item_mask[sub_mask] = sub_click_item_mask

        result_query_emb = result_query_emb.reshape(
            (*sample.shape, const.final_emb_size))
        result_item_emb = result_item_emb.reshape(
            (*sample.shape, const.max_session_item_len, const.final_emb_size))
        result_item_mask = result_item_mask.reshape(
            (*sample.shape, const.max_session_item_len))

        return [result_query_emb, result_item_emb, result_item_mask]
