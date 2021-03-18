import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class FFN(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(FFN, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class ASAS(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(ASAS, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        edim = args.edim
        droprate = args.droprate
        self.edim = args.edim
        self.dev = torch.device(args.device)
        self.args = args

        # Self-Attention Block, encode user behavior sequences
        num_heads = 1
        self.item_attn_layernorm = nn.LayerNorm(edim, eps=1e-8)
        self.item_attn_layer = nn.MultiheadAttention(edim, num_heads, droprate)
        self.item_ffn_layernorm = nn.LayerNorm(edim, eps=1e-8)
        self.item_ffn = FFN(edim, args.droprate)
        self.item_last_layernorm = nn.LayerNorm(edim, eps=1e-8)

        # Embedding Layer
        self.user_embs = nn.Embedding(user_num, edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        nn.init.normal_(self.user_embs.weight, std=0.01)
        nn.init.normal_(self.item_embs.weight, std=0.01)

        if args.use_pos_emb:
            self.posn_embs = nn.Embedding(args.seq_maxlen, edim, padding_idx=0)
            nn.init.normal_(self.posn_embs.weight, std=0.01)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(args.droprate)

    def seq2feat(self, seq_iid, pert=None):
        timeline_mask = torch.BoolTensor(seq_iid == 0).to(self.dev)  # mask the padding item
        seqs = self.item_embs(seq_iid.to(self.dev)) * (self.item_embs.embedding_dim ** 0.5)  # Rescale emb
        if self.args.use_pos_emb:
            positions = np.tile(np.array(range(seq_iid.shape[1]), dtype=np.int64), [seq_iid.shape[0], 1])
            seqs += self.posn_embs(torch.LongTensor(positions).to(self.dev))
        if pert is not None:
            seqs += pert

        seqs = self.dropout(seqs)

        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim
        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        seqs = torch.transpose(seqs, 0, 1)  # seqlen x B x d
        query = self.item_attn_layernorm(seqs)
        mha_outputs, _ = self.item_attn_layer(query, seqs, seqs, attn_mask=attention_mask)

        seqs = query + mha_outputs
        seqs = torch.transpose(seqs, 0, 1)  # B x seqlen x d
        seqs = self.item_ffn_layernorm(seqs)
        seqs = self.item_ffn(seqs)
        seqs *= ~timeline_mask.unsqueeze(-1)  # B x seqlen x d
        seqs = self.item_last_layernorm(seqs)

        return seqs

    def pred(self, hu, hi):
        return (hu * hi).sum(dim=-1)

    def forward(self, batch, pert=None):
        uid, seq, pos, neg, nbr, nbr_iid = batch
        hu = self.seq2feat(seq, pert)  # B x sl x d
        pos_hi = self.dropout(self.item_embs(pos.to(self.dev)))
        neg_hi = self.dropout(self.item_embs(neg.to(self.dev)))
        if pert is not None:
            pos_logits = self.pred(hu, pos_hi + pert)
            neg_logits = self.pred(hu, neg_hi + pert)
        else:
            pos_logits = self.pred(hu, pos_hi)
            neg_logits = self.pred(hu, neg_hi)

        return pos_logits, neg_logits

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for i, eval_batch in enumerate(eval_loader):
                uid, seq, nbr, nbr_iid, eval_iid = eval_batch
                seq = seq.long()
                eval_iid = eval_iid.long()
                hi = self.item_embs(eval_iid.to(self.dev))  # B x item_len x d
                hu = self.seq2feat(seq)[:, -1, :]
                hu = hu.unsqueeze(1).expand_as(hi)

                batch_score = self.pred(hu, hi)  # B x item_len
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores


class BPRMF(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(BPRMF, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = torch.device(args.device)
        self.args = args

        # Embedding Layer
        self.user_embs = nn.Embedding(user_num, args.edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, args.edim, padding_idx=0)
        nn.init.normal_(self.user_embs.weight, std=0.01)
        nn.init.normal_(self.item_embs.weight, std=0.01)

        self.dropout = nn.Dropout(args.droprate)

    def pred(self, hu, hi):
        return (hu * hi).sum(dim=-1)

    def forward(self, batch):
        uid, seq, pos, neg, nbr, nbr_iid = batch

        # Fuse Layer
        uid = uid.unsqueeze(1).expand_as(seq)
        hu = self.dropout(self.user_embs(uid.to(self.dev)))
        pos_hi = self.dropout(self.item_embs(pos.to(self.dev)))
        neg_hi = self.dropout(self.item_embs(neg.to(self.dev)))
        pos_logits = self.pred(hu, pos_hi)
        neg_logits = self.pred(hu, neg_hi)

        return pos_logits, neg_logits, hu, pos_hi, neg_hi

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for i, eval_batch in enumerate(eval_loader):
                uid, seq, nbr, nbr_iid, eval_iid = eval_batch
                hi = self.item_embs(eval_iid.long().to(self.dev))  # B x item_len x d
                uid = uid.long().unsqueeze(1).expand_as(eval_iid)
                hu = self.user_embs(uid.to(self.dev))
                batch_score = self.pred(hu, hi)  # B x item_len
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores


class DGRec(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(DGRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        edim = args.edim
        self.edim = args.edim
        self.dev = torch.device(args.device)
        self.args = args

        self.rnn = nn.LSTM(input_size=edim, hidden_size=edim, num_layers=1, batch_first=True)
        self.nbr_lin = nn.Linear(edim + edim, edim, bias=False)
        self.gat_lin1 = nn.Linear(edim, edim, bias=False)
        self.gat_lin2 = nn.Linear(edim, edim, bias=False)
        self.gat_lin3 = nn.Linear(edim, edim, bias=False)
        self.fuse_lin = nn.Linear(edim + edim, edim, bias=False)

        self.user_embs = nn.Embedding(user_num, edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        nn.init.uniform_(self.user_embs.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(args.droprate)

    def pred(self, hu, hi):
        return (hu * hi).sum(dim=-1)

    def dgrec_forward(self, uid, seq, nbr, nbr_iid):
        # get mask
        batch_size, nbr_maxlen, seq_maxlen = nbr_iid.shape
        nbr_mask = torch.BoolTensor(nbr == 0).to(self.dev)  # B x nl
        nbr_seq_mask = torch.BoolTensor(nbr_iid == 0).to(self.dev)  # B x nl x sl

        # user short term feat
        user_seq_emb = self.item_embs(seq.to(self.dev))  # B x sl x d
        user_seq_feat, _ = self.rnn(user_seq_emb)  # B x sl x d
        user_seq_feat = user_seq_feat[:, -1, :]  # B x d

        # nbr short term feat
        nbr_seq_emb = self.item_embs(nbr_iid.to(self.dev))  # B x nl x sl x d
        nbr_seq_emb *= ~nbr_seq_mask.unsqueeze(-1)
        nbr_seq_emb = nbr_seq_emb.view(batch_size * nbr_maxlen, seq_maxlen, self.edim)
        nbr_seq_feat, _ = self.rnn(nbr_seq_emb)  # (B*nl) x sl x d
        nbr_seq_feat = nbr_seq_feat[:, -1, :]  # (B*nl) x d
        nbr_seq_feat = nbr_seq_feat.view(batch_size, nbr_maxlen, self.edim)  # B x nl x d

        # nbr long term feat
        nbr_emb = self.user_embs(nbr.to(self.dev))  # B x nl x d

        # nbr final feat
        nbr_feat = self.dropout(self.act(self.nbr_lin(
            torch.cat([nbr_seq_feat, nbr_emb], dim=-1))))  # B x nl x d

        # 3 layer dot GAT
        hu = user_seq_feat.unsqueeze(1)  # B x 1 x d
        attn = (hu * nbr_feat).sum(dim=-1, keepdims=True)  # B x nl x 1
        attn = attn + (-1e9) * nbr_mask.unsqueeze(-1)
        attn = F.softmax(attn, dim=1)
        hu = (attn * nbr_feat).sum(dim=1)  # B x d
        hu = self.dropout(self.act(self.gat_lin1(hu)))

        hu = hu.unsqueeze(1)  # B x 1 x d
        attn = (hu * nbr_feat).sum(dim=-1, keepdims=True)  # B x nl x 1
        attn = attn + (-1e9) * nbr_mask.unsqueeze(-1)
        attn = F.softmax(attn, dim=1)
        hu = (attn * nbr_feat).sum(dim=1)  # B x d
        hu = self.dropout(self.act(self.gat_lin2(hu)))

        hu = hu.unsqueeze(1)  # B x 1 x d
        attn = (hu * nbr_feat).sum(dim=-1, keepdims=True)  # B x nl x 1
        attn = attn + (-1e9) * nbr_mask.unsqueeze(-1)
        attn = F.softmax(attn, dim=1)
        hu = (attn * nbr_feat).sum(dim=1)  # B x d
        hu = self.dropout(self.act(self.gat_lin3(hu)))

        # user final feat
        hu = self.fuse_lin(torch.cat([hu, user_seq_feat], dim=-1))  # B x d

        return hu

    def forward(self, batch):
        uid, seq, pos, neg, nbr, nbr_iid = batch

        hu = self.dgrec_forward(uid, seq, nbr, nbr_iid)

        pos_hi = self.dropout(self.item_embs(pos.to(self.dev)))
        neg_hi = self.dropout(self.item_embs(neg.to(self.dev)))
        hu = hu.unsqueeze(1).expand_as(pos_hi)

        pos_logits = self.pred(hu, pos_hi)
        neg_logits = self.pred(hu, neg_hi)

        return pos_logits, neg_logits

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for i, eval_batch in enumerate(eval_loader):
                uid, seq, nbr, nbr_iid, eval_iid = eval_batch
                uid = uid.long()
                seq = seq.long()
                nbr = nbr.long()
                nbr_iid = nbr_iid.long()
                eval_iid = eval_iid.long()

                hi = self.item_embs(eval_iid.to(self.dev))  # B x item_len x d
                hu = self.dgrec_forward(uid, seq, nbr, nbr_iid)
                hu = hu.unsqueeze(1).expand_as(hi)

                batch_score = self.pred(hu, hi)  # B x item_len
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores

    def get_parameters(self):
        param_list = [
            {'params': self.rnn.parameters()},
            {'params': self.nbr_lin.parameters()},
            {'params': self.gat_lin1.parameters()},
            {'params': self.gat_lin2.parameters()},
            {'params': self.gat_lin3.parameters()},
            {'params': self.fuse_lin.parameters()},
            {'params': self.user_embs.parameters(), 'weight_decay': 0},
            {'params': self.item_embs.parameters(), 'weight_decay': 0},
        ]

        return param_list


class AttnLayer(torch.nn.Module):
    def __init__(self, edim, droprate):
        super(AttnLayer, self).__init__()
        self.attn0 = nn.Linear(edim + edim, edim)
        self.attn1 = nn.Linear(edim, edim)
        self.attn2 = nn.Linear(edim, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(droprate)

    def forward(self, items, user, items_mask):
        # items: B x l x d
        # user:  B x 1 x d
        # items_mask: B x l x 1

        user = user.expand_as(items)
        h = torch.cat([items, user], dim=-1)
        h = self.dropout(self.act(self.attn1(
            self.dropout(self.act(self.attn0(
                h))))))

        h = self.attn2(h) - 1e9 * items_mask
        a = torch.softmax(h, dim=1)  # B x l x 1
        attn_items = (a * items).sum(dim=1)  # B x d
        return attn_items


class GRec(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(GRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        edim = args.edim
        droprate = args.droprate
        self.edim = args.edim
        self.dev = torch.device(args.device)
        self.args = args

        # Fuse Layer
        self.item_attn = AttnLayer(edim, droprate)
        self.user_attn = AttnLayer(edim, droprate)
        self.seq_nbr_lin = nn.Linear(edim + edim, edim)
        self.user_mlp = nn.Sequential(
            nn.Linear(edim, edim),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(edim, edim),
        )

        self.item_mlp = nn.Sequential(
            nn.Linear(edim, edim),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(edim, edim),
        )

        self.pred_mlp = nn.Sequential(
            nn.Linear(edim + edim, edim),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(edim, edim),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(edim, 1),
        )

        # Embedding Layer
        self.user_embs = nn.Embedding(user_num, edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        nn.init.normal_(self.user_embs.weight, std=0.01)
        nn.init.normal_(self.item_embs.weight, std=0.01)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(args.droprate)

    def seq2feat(self, uid, seq_iid):
        item_mask = torch.BoolTensor(seq_iid == 0).to(self.dev)  # B x sl
        item_mask = item_mask.unsqueeze(-1)  # B x sl x 1
        items_emb = self.item_embs(seq_iid.to(self.dev))  # B x sl x d
        uid = uid.to(self.dev).unsqueeze(-1)  # B x 1
        user_emb = self.dropout(self.user_embs(uid))  # B x 1 x d
        seq_feat = self.item_attn(items_emb, user_emb, item_mask)
        return seq_feat

    def nbr2feat(self, uid, nbr):
        nbr_mask = torch.BoolTensor(nbr == 0).to(self.dev)  # B x nl
        nbr_mask = nbr_mask.unsqueeze(-1)  # B x nl x 1
        uid = uid.to(self.dev).unsqueeze(-1)  # B x 1
        nbr = nbr.to(self.dev)  # B x nl
        user_emb = self.dropout(self.user_embs(uid))  # B x  1 x d
        nbrs_emb = self.dropout(self.user_embs(nbr))  # B x nl x d
        nbr_feat = self.user_attn(nbrs_emb, user_emb, nbr_mask)
        return nbr_feat

    def pred(self, hu, hi):
        batch_size, seq_maxlen, edim = hi.shape
        hu = self.user_mlp(hu)
        hu = hu.unsqueeze(1).expand_as(hi)
        hi = self.item_mlp(hi)
        logits = self.pred_mlp(torch.cat([hu, hi], dim=-1))
        logits = logits.view(batch_size, seq_maxlen)
        return logits

    def forward(self, batch):
        uid, seq, pos, neg, nbr, nbr_iid = batch
        seq_feat = self.seq2feat(uid, seq)  # B x d
        nbr_feat = self.nbr2feat(uid, nbr)  # B x d

        # Fuse Layer
        hu = self.seq_nbr_lin(torch.cat([seq_feat, nbr_feat], dim=-1))
        pos_hi = self.dropout(self.item_embs(pos.to(self.dev)))
        neg_hi = self.dropout(self.item_embs(neg.to(self.dev)))
        pos_logits = self.pred(hu, pos_hi)
        neg_logits = self.pred(hu, neg_hi)
        return pos_logits, neg_logits

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for eval_batch in tqdm(eval_loader):
                uid, seq, nbr, nbr_iid, eval_iid = eval_batch
                uid = uid.long()
                seq = seq.long()
                nbr = nbr.long()
                eval_iid = eval_iid.long()

                hi = self.item_embs(eval_iid.to(self.dev))  # B x item_len x d
                seq_feat = self.seq2feat(uid, seq)  # B x d
                nbr_feat = self.nbr2feat(uid, nbr)  # B x d
                hu = self.seq_nbr_lin(torch.cat([seq_feat, nbr_feat], dim=-1))
                batch_score = self.pred(hu, hi)  # B x item_len
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores


class LightGCN(nn.Module):
    def __init__(self,
                 edim,
                 n_layers,
                 user_num,
                 item_num,
                 ui_graph,
                 args):
        super(LightGCN, self).__init__()
        self.n_layers = n_layers
        self.num_users = user_num
        self.num_items = item_num
        self.args = args
        self.dev = torch.device(args.device)
        self.Graph = ui_graph.to(self.dev)

        self.embedding_user = torch.nn.Embedding(self.num_users, edim)
        self.embedding_item = torch.nn.Embedding(self.num_items, edim)
        nn.init.uniform_(self.embedding_user.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.embedding_item.weight, a=-0.5 / item_num, b=0.5 / item_num)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [torch.cat([users_emb, items_emb])]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, batch):
        uid, pos, neg = batch
        indices = torch.where(pos != 0)
        uid = uid.unsqueeze(1).expand_as(pos)

        users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0 = \
            self.getEmbedding(
                uid.long().to(self.dev),
                pos.long().to(self.dev),
                neg.long().to(self.dev))

        pos_scores = (users_emb * pos_emb).sum(dim=-1)
        neg_scores = (users_emb * neg_emb).sum(dim=-1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores[indices] - pos_scores[indices]))
        loss += self.args.emb_reg * 0.5 * (
                userEmb0.norm(2, dim=-1).pow(2).mean() +
                posEmb0.norm(2, dim=-1).pow(2).mean() +
                negEmb0.norm(2, dim=-1).pow(2).mean())

        return loss

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            all_users, all_items = self.computer()
            for eval_batch in tqdm(eval_loader):
                uid, eval_iid = eval_batch
                users_emb = all_users[uid.long().to(self.dev)]  # B x d
                items_emb = all_items[eval_iid.long().to(self.dev)]  # B x item_len x d
                users_emb = users_emb.unsqueeze(1).expand_as(items_emb)
                batch_score = (users_emb * items_emb).sum(dim=-1)
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores


class NeuMF(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(NeuMF, self).__init__()
        self.edim = args.edim
        edim = args.edim
        self.dev = torch.device(args.device)
        self.embedding_user_mlp = torch.nn.Embedding(user_num, edim, padding_idx=0)
        self.embedding_item_mlp = torch.nn.Embedding(item_num, edim, padding_idx=0)
        self.embedding_user_mf = torch.nn.Embedding(user_num, edim, padding_idx=0)
        self.embedding_item_mf = torch.nn.Embedding(item_num, edim, padding_idx=0)
        torch.nn.init.uniform_(self.embedding_user_mlp.weight, a=-0.5 / user_num, b=0.5 / user_num)
        torch.nn.init.uniform_(self.embedding_item_mlp.weight, a=-0.5 / item_num, b=0.5 / item_num)
        torch.nn.init.uniform_(self.embedding_user_mf.weight, a=-0.5 / user_num, b=0.5 / user_num)
        torch.nn.init.uniform_(self.embedding_item_mf.weight, a=-0.5 / item_num, b=0.5 / item_num)

        self.mlp_lin0 = nn.Linear(edim + edim, edim)
        self.mlp_lin1 = nn.Linear(edim, edim)
        self.mlp_lin2 = nn.Linear(edim, edim)
        self.out_lin = nn.Linear(edim + edim, 1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(args.droprate)

    def forward(self, batch):
        uid, seq, pos, neg, nbr, nbr_iid = batch
        user_indices = uid
        item_indices = pos
        neg_item_indices = neg

        item_indices = item_indices.to(self.dev).long()
        neg_item_indices = neg_item_indices.to(self.dev).long()
        user_indices = user_indices.to(self.dev).long().unsqueeze(1).expand_as(item_indices)

        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        pos_item_embedding_mlp = self.embedding_item_mlp(item_indices)
        neg_item_embedding_mlp = self.embedding_item_mlp(neg_item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        pos_item_embedding_mf = self.embedding_item_mf(item_indices)
        neg_item_embedding_mf = self.embedding_item_mf(neg_item_indices)

        pos_mlp_vector = torch.cat([user_embedding_mlp, pos_item_embedding_mlp], dim=-1)
        pos_mlp_vector = \
            self.dropout(self.act(self.mlp_lin2(
                self.dropout(self.act(self.mlp_lin1(
                    self.dropout(self.act(self.mlp_lin0(
                        pos_mlp_vector)))))))))

        pos_mf_vector = self.dropout(torch.mul(user_embedding_mf, pos_item_embedding_mf))
        pos_vector = torch.cat([pos_mlp_vector, pos_mf_vector], dim=-1)
        pos_logits = self.out_lin(pos_vector)

        neg_mlp_vector = torch.cat([user_embedding_mlp, neg_item_embedding_mlp], dim=-1)
        neg_mlp_vector = \
            self.dropout(self.act(self.mlp_lin2(
                self.dropout(self.act(self.mlp_lin1(
                    self.dropout(self.act(self.mlp_lin0(
                        neg_mlp_vector)))))))))

        neg_mf_vector = self.dropout(torch.mul(user_embedding_mf, neg_item_embedding_mf))
        neg_vector = torch.cat([neg_mlp_vector, neg_mf_vector], dim=-1)
        neg_logits = self.out_lin(neg_vector)

        return pos_logits, neg_logits

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for eval_batch in eval_loader:
                uid, seq, nbr, nbr_iid, eval_iid = eval_batch
                user_indices = uid
                item_indices = eval_iid

                item_indices = item_indices.to(self.dev).long()
                user_indices = user_indices.to(self.dev).long().unsqueeze(1).expand_as(item_indices)

                item_embedding_mlp = self.embedding_item_mlp(item_indices)
                user_embedding_mlp = self.embedding_user_mlp(user_indices)
                item_embedding_mf = self.embedding_item_mf(item_indices)
                user_embedding_mf = self.embedding_user_mf(user_indices)

                mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
                mlp_vector = \
                    self.dropout(self.act(self.mlp_lin2(
                        self.dropout(self.act(self.mlp_lin1(
                            self.dropout(self.act(self.mlp_lin0(
                                mlp_vector)))))))))

                mf_vector = self.dropout(torch.mul(user_embedding_mf, item_embedding_mf))
                vector = torch.cat([mlp_vector, mf_vector], dim=-1)
                batch_score = self.out_lin(vector).squeeze()
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores


class SAS(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(SAS, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        edim = args.edim
        droprate = args.droprate
        self.edim = args.edim
        self.dev = torch.device(args.device)
        self.args = args

        # Self-Attention Block, encode user behavior sequences
        num_heads = 1
        self.item_attn_layernorm = nn.LayerNorm(edim, eps=1e-8)
        self.item_attn_layer = nn.MultiheadAttention(edim, num_heads, droprate)
        self.item_ffn_layernorm = nn.LayerNorm(edim, eps=1e-8)
        self.item_ffn = FFN(edim, args.droprate)
        self.item_last_layernorm = nn.LayerNorm(edim, eps=1e-8)

        # Embedding Layer
        self.user_embs = nn.Embedding(user_num, edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        nn.init.normal_(self.user_embs.weight, std=0.01)
        nn.init.normal_(self.item_embs.weight, std=0.01)

        if args.use_pos_emb:
            self.posn_embs = nn.Embedding(args.seq_maxlen, edim, padding_idx=0)
            nn.init.normal_(self.posn_embs.weight, std=0.01)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(args.droprate)
        self.trained_user_feat = None

    def seq2feat(self, seq_iid):
        timeline_mask = torch.BoolTensor(seq_iid == 0).to(self.dev)  # mask the padding item
        seqs = self.item_embs(seq_iid.to(self.dev)) * (self.item_embs.embedding_dim ** 0.5)  # Rescale emb
        if self.args.use_pos_emb:
            positions = np.tile(np.array(range(seq_iid.shape[1]), dtype=np.int64), [seq_iid.shape[0], 1])
            seqs += self.posn_embs(torch.LongTensor(positions).to(self.dev))

        seqs = self.dropout(seqs)

        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim
        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        seqs = torch.transpose(seqs, 0, 1)  # seqlen x B x d
        query = self.item_attn_layernorm(seqs)
        mha_outputs, _ = self.item_attn_layer(query, seqs, seqs, attn_mask=attention_mask)

        seqs = query + mha_outputs
        seqs = torch.transpose(seqs, 0, 1)  # B x seqlen x d
        seqs = self.item_ffn_layernorm(seqs)
        seqs = self.item_ffn(seqs)
        seqs *= ~timeline_mask.unsqueeze(-1)  # B x seqlen x d
        seqs = self.item_last_layernorm(seqs)

        return seqs

    def pred(self, hu, hi):
        return (hu * hi).sum(dim=-1)

    def forward(self, batch):
        uid, seq, pos, neg, nbr, nbr_iid = batch
        hu = self.seq2feat(seq)  # B x sl x d
        pos_hi = self.dropout(self.item_embs(pos.to(self.dev)))
        neg_hi = self.dropout(self.item_embs(neg.to(self.dev)))
        pos_logits = self.pred(hu, pos_hi)
        neg_logits = self.pred(hu, neg_hi)
        return pos_logits, neg_logits

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for i, eval_batch in enumerate(eval_loader):
                uid, seq, nbr, nbr_iid, eval_iid = eval_batch
                seq = seq.long()
                eval_iid = eval_iid.long()
                hi = self.item_embs(eval_iid.to(self.dev))  # B x item_len x d
                hu = self.seq2feat(seq)[:, -1, :]
                hu = hu.unsqueeze(1).expand_as(hi)

                batch_score = self.pred(hu, hi)  # B x item_len
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores


class SocialMF(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(SocialMF, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.edim = args.edim
        self.dev = torch.device(args.device)
        self.args = args

        # Embedding Layer
        self.user_embs = nn.Embedding(user_num, args.edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, args.edim, padding_idx=0)
        nn.init.uniform_(self.user_embs.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)

    def pred(self, hu, hi):
        return (hu * hi).sum(dim=-1)

    def forward(self, batch):
        uid, seq, pos, neg, nbr, nbr_iid = batch
        uid = uid.unsqueeze(1).expand_as(seq)
        hu = self.user_embs(uid.to(self.dev))
        pos_hi = self.item_embs(pos.to(self.dev))
        neg_hi = self.item_embs(neg.to(self.dev))
        pos_logits = self.pred(hu, pos_hi)
        neg_logits = self.pred(hu, neg_hi)

        batch_size, nbr_maxlen = nbr.shape
        nbr_mask = torch.BoolTensor(nbr == 0).to(self.dev)  # B x nl
        nbr_len = (nbr_maxlen - nbr_mask.sum(1))  # B
        nbr_emb = self.user_embs(nbr.to(self.dev))  # B x nl x d
        nbr_emb *= ~nbr_mask.unsqueeze(-1)  # B x nl x d
        nbr_len = nbr_len.view(batch_size, 1, 1)  # B x 1  x 1
        nbr_emb = nbr_emb.sum(dim=1, keepdim=True) / nbr_len  # B x 1  x d
        nbr_emb = nbr_emb.expand_as(hu)

        return pos_logits, neg_logits, hu, pos_hi, neg_hi, nbr_emb

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for i, eval_batch in enumerate(eval_loader):
                uid, seq, nbr, nbr_iid, eval_iid = eval_batch
                uid = uid.long()
                eval_iid = eval_iid.long()
                hi = self.item_embs(eval_iid.to(self.dev))  # B x item_len x d
                uid = uid.unsqueeze(1).expand_as(eval_iid)
                user_emb = self.user_embs(uid.to(self.dev))
                hu = user_emb
                batch_score = self.pred(hu, hi)  # B x item_len
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores


class SocRec(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(SocRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.edim = args.edim
        self.dev = torch.device(args.device)
        self.args = args

        # Embedding Layer
        self.user_embs = nn.Embedding(user_num, args.edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, args.edim, padding_idx=0)
        nn.init.uniform_(self.user_embs.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)

    def pred(self, hu, hi):
        return (hu * hi).sum(dim=-1)

    def forward(self, batch):
        uid, seq, pos, neg, nbr, nbr_iid = batch
        user_emb = self.user_embs(uid.to(self.dev))
        pos_hi = self.item_embs(pos.to(self.dev))
        neg_hi = self.item_embs(neg.to(self.dev))
        hu = user_emb.unsqueeze(1).expand_as(pos_hi)
        pos_logits = self.pred(hu, pos_hi)
        neg_logits = self.pred(hu, neg_hi)

        nbr_emb = self.user_embs(nbr.to(self.dev))  # B x nl x d
        user_emb = self.user_embs(uid.to(self.dev))
        hu = user_emb.unsqueeze(1).expand_as(nbr_emb)
        nbr_logits = self.pred(hu, nbr_emb)

        return pos_logits, neg_logits, nbr_logits, user_emb, pos_hi, neg_hi, nbr_emb

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for i, eval_batch in enumerate(eval_loader):
                uid, seq, nbr, nbr_iid, eval_iid = eval_batch
                uid = uid.long()
                eval_iid = eval_iid.long()
                hi = self.item_embs(eval_iid.to(self.dev))  # B x item_len x d
                uid = uid.unsqueeze(1).expand_as(eval_iid)
                hu = self.user_embs(uid.to(self.dev))
                batch_score = self.pred(hu, hi)  # B x item_len
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores


class TransRec(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(TransRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = torch.device(args.device)
        self.args = args

        self.user_embs = nn.Embedding(user_num, args.edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, args.edim, padding_idx=0)
        self.item_beta = nn.Embedding(item_num, 1, padding_idx=0)
        self.trans = nn.Parameter(torch.zeros((args.edim,)))

        nn.init.uniform_(self.user_embs.weight, a=-6 / args.edim, b=6 / args.edim)
        nn.init.uniform_(self.item_embs.weight, a=-6 / args.edim, b=6 / args.edim)
        nn.init.uniform_(self.item_beta.weight, a=-6 / args.edim, b=6 / args.edim)
        nn.init.uniform_(self.trans.data, a=-6 / args.edim, b=6 / args.edim)

    def l2_distance(self, x, y):
        return (x - y).square().sum(dim=-1, keepdims=True)

    def clip_by_norm(self, tensor, clip_norm, dim=None):
        clip_norm = torch.FloatTensor([clip_norm]).to(tensor.device)
        l2sum = tensor.pow(2).sum(dim=dim, keepdims=True)
        pred = l2sum > 0
        l2sum_safe = torch.where(pred, l2sum, torch.ones_like(l2sum))
        l2norm = torch.where(pred, l2sum_safe.sqrt(), l2sum)
        values_clip = tensor * clip_norm / torch.max(l2norm, clip_norm)
        return values_clip

    def forward(self, batch):
        uid, seq, pos, neg, nbr, nbr_iid = batch

        user_emb = self.clip_by_norm(self.user_embs(uid.to(self.dev)), clip_norm=1, dim=-1)  # B x d
        seq_emb = self.clip_by_norm(self.item_embs(seq.to(self.dev)), clip_norm=1, dim=-1)  # B x sl x d
        pos_emb = self.clip_by_norm(self.item_embs(pos.to(self.dev)), clip_norm=1, dim=-1)
        neg_emb = self.clip_by_norm(self.item_embs(neg.to(self.dev)), clip_norm=1, dim=-1)

        user_emb = user_emb.unsqueeze(1).expand_as(seq_emb)  # B x sl x d
        h_out = user_emb + self.trans + seq_emb

        pos_bias = self.item_beta(pos.to(self.dev))  # B x sl x 1
        neg_bias = self.item_beta(neg.to(self.dev))  # B x sl x 1

        pos_logits = pos_bias - self.l2_distance(h_out, pos_emb)  # B x sl x 1
        neg_logits = neg_bias - self.l2_distance(h_out, neg_emb)  # B x sl x 1

        return pos_logits, neg_logits

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for i, eval_batch in enumerate(eval_loader):
                uid, seq, nbr, nbr_iid, eval_iid = eval_batch
                user_emb = self.user_embs(uid.long().to(self.dev))  # B x d
                last_emb = self.item_embs(seq[:, -1].long().to(self.dev))  # B x d
                item_embs = self.item_embs(eval_iid.long().to(self.dev))  # B x item_len x d
                item_bias = self.item_beta(eval_iid.long().to(self.dev))  # B x item_len x 1
                h_out = user_emb + self.trans + last_emb
                h_out = h_out.unsqueeze(1).expand_as(item_embs)
                batch_score = item_bias - self.l2_distance(h_out, item_embs)
                all_scores.append(batch_score.squeeze())

            all_scores = torch.cat(all_scores, dim=0)

            return all_scores
