#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 上午11:12
# @Author  : PeiP Liu
# @FileName: viterbi.py
# @Software: PyCharm
import torch

class ViterbiDetector(object):
    def __init__(
            self,
            id2label,
            transition_matrix,
            ignore_token_label_id=torch.nn.CrossEntropyLoss().ignore_index,
    ):
        self.id2label = id2label
        self.n_labels = len(id2label)
        self.transitions = transition_matrix
        self.ignore_token_label_id = ignore_token_label_id

    def forward(self, logprobs, attention_mask, label_ids):
        # logprobs: (batch_size, max_seq, n_labels)
        batch_size, max_seq_len, n_labels = logprobs.size()
        attention_mask = attention_mask[:, :max_seq_len]
        label_ids = label_ids[:, :max_seq_len]

        active_tokens = (attention_mask == 1) & (label_ids != self.ignore_token_label_id)
        if n_labels != self.n_labels:
            raise ValueError("Labels do not match!")

        label_seqs = []
        for idx in range(batch_size):
            logprob_i = logprobs[idx, :, :][active_tokens[idx]] # (seq_len(activate), n_labels)

            back_pointters = []

            forward_var = logprob_i[0] # n_labels

            for j in range(1, len(logprob_i)): # for tag_feat in feat
                next_label_var = forward_var + self.transitions
                viterbivars_t, bptrs_t = torch.max(next_label_var, dim=1)

                logp_j = logprob_i[j]
                forward_var = viterbivars_t + logp_j
                bptrs_t = bptrs_t.cpu().numpy().tolist()
                back_pointters.append(bptrs_t)

            path_score, best_label_id = torch.max(forward_var, dim=-1)
            best_label_id = best_label_id.item()
            best_path = [best_label_id]

            for bptrs_t in reversed(back_pointters):
                best_label_id = bptrs_t[best_label_id]
                best_path.append(best_label_id)

            if len(best_path) != len(logprob_i):
                raise ValueError("Number of labels doesn't match!")

            best_path.reverse()
            label_seqs.append(best_path)

        return label_seqs
