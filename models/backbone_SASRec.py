import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import *

import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class Item_Embedding(nn.Module):
    def __init__(self, emb_pipline, **key_words):

        super(Item_Embedding, self).__init__()

        data_statis = pd.read_pickle(
            os.path.join(key_words["language_embs_path"], 'data_statis.df'))
        self.state_size = data_statis['seq_size'][0]
        self.item_num = data_statis['item_num'][0]

        self.construct_item_embeddings(emb_pipline, **key_words)
        print("Item_Embedding initialized")

    def construct_item_embeddings(self, emb_pipline, **key_words):

        if emb_pipline == "ID":

            self.init_ID_embedding(key_words["hidden_dim"], key_words["ID_embs_init_type"])

        elif emb_pipline == "SI":

            self.init_ID_embedding(key_words["hidden_dim"], "language_embeddings", **key_words)

        elif emb_pipline == "SR":

            self.init_ID_embedding(key_words["hidden_dim"], key_words["ID_embs_init_type"], **key_words)
            language_embs = self.load_language_embeddings(key_words["language_embs_path"],
                                                          key_words["language_model_type"],
                                                          key_words["language_embs_scale"])

            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs, dtype=torch.float32),
                freeze=True,
            )

        elif emb_pipline == "Dual_view":

            self.init_ID_embedding(key_words["hidden_dim"], "language_embeddings", **key_words)
            language_embs = self.load_language_embeddings(key_words["language_embs_path"],
                                                          key_words["language_model_type"],
                                                          key_words["language_embs_scale"])
            padding_emb = np.random.rand(language_embs.shape[1])
            language_embs = np.vstack([language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs, dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
            )

        elif emb_pipline == "AP":

            language_embs = self.load_language_embeddings(key_words["language_embs_path"],
                                                          key_words["language_model_type"],
                                                          key_words["language_embs_scale"])
            padding_emb = np.random.rand(language_embs.shape[1])
            language_embs = np.vstack([language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs, dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
            )

        elif emb_pipline == "WAP":

            key_words["item_frequency_flag"] = False
            key_words['standardization'] = True
            language_embs = self.semantic_space_decomposion(None, **key_words)
            padding_emb = np.random.rand(language_embs.shape[1])
            language_embs = np.vstack([language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs, dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
            )

        elif emb_pipline == "AF":

            cliped_language_embs = self.semantic_space_decomposion(key_words["hidden_dim"], **key_words)
            padding_emb = np.random.rand(cliped_language_embs.shape[1])
            cliped_language_embs = np.vstack(
                [cliped_language_embs, padding_emb])

            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(cliped_language_embs, dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
            )

            self.init_ID_embedding(self.nullity, key_words["ID_embs_init_type"])
            print("language_embeddings and ID_embeddings finished!")

    def load_language_embeddings(self, directory, language_model_type, scale):

        language_embs = pd.read_pickle(os.path.join(directory,
                                                    language_model_type + '_emb.pickle'))
        self.item_num = len(language_embs)
        self.language_dim = len(language_embs[0])

        return np.stack(language_embs) * scale

    def init_ID_embedding(self, ID_dim, init_type, **key_words):

        if init_type == "language_embeddings":

            language_embs = self.load_language_embeddings(key_words["language_embs_path"],
                                                          key_words["language_model_type"],
                                                          key_words["language_embs_scale"])
            if self.language_dim == ID_dim:

                padding_emb = np.random.rand(language_embs.shape[1])
                language_embs = np.vstack([language_embs, padding_emb])

                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.item_num
                )
            else:

                clipped_language_embs = self.semantic_space_decomposion(ID_dim, **key_words)
                padding_emb = np.random.rand(clipped_language_embs.shape[1])
                clipped_language_embs = np.vstack([clipped_language_embs, padding_emb])

                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(clipped_language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.item_num
                )
        else:

            self.ID_embeddings = nn.Embedding(
                num_embeddings=self.item_num + 1,
                embedding_dim=ID_dim,

            )

            if init_type == "uniform":
                nn.init.uniform_(self.ID_embeddings.weight, a=0.0, b=1.0)
            elif init_type == "normal":
                nn.init.normal_(self.ID_embeddings.weight, 0, 1)
            elif init_type == "zeros":
                nn.init.zeros_(self.ID_embeddings.weight)
            elif init_type == "ortho":
                nn.init.orthogonal_(self.ID_embeddings.weight, gain=1.0)
            elif init_type == "xavier":
                nn.init.xavier_uniform_(self.ID_embeddings.weight, gain=1.0)
            elif init_type == "sparse":
                nn.init.sparse_(self.ID_embeddings.weight, 0.01, std=1)
            else:
                raise NotImplementedError("This kind of init for ID embeddings is not implemented yet.")

    def semantic_space_decomposion(self, clipped_dim, **key_words):

        language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"],
                                                      key_words["language_embs_scale"])

        if not key_words["item_frequency_flag"]:
            self.language_mean = np.mean(language_embs, axis=0)
            cov = np.cov(language_embs - self.language_mean, rowvar=False)
        else:

            items_pop = np.load(os.path.join(key_words["language_embs_path"], 'items_pop.npy'))
            items_freq_scale = 1.0 / items_pop.sum()
            items_freq = (items_pop * items_freq_scale).reshape(-1, 1)
            self.language_mean = np.sum(language_embs * items_freq, axis=0)
            cov = np.cov((language_embs - self.language_mean) * np.sqrt(items_freq), rowvar=False)

        U, S, _ = np.linalg.svd(cov, full_matrices=False)

        if key_words["null_thres"] is not None:

            indices_null = np.where(S <= key_words["null_thres"])[0]
            self.nullity = len(indices_null)
        elif key_words["null_dim"] is not None:

            self.nullity = key_words["null_dim"]

        if clipped_dim is None:
            clipped_dim = self.language_dim
        if key_words["cover"]:

            clipped_dim = clipped_dim - self.nullity

        Projection_matrix = U[..., :clipped_dim]

        if key_words['standardization']:

            Diagnals = np.sqrt(1 / S)[:clipped_dim]

            Projection_matrix = Projection_matrix.dot(np.diag(Diagnals))

        clipped_language_embs = (language_embs - self.language_mean).dot(
            Projection_matrix)
        return clipped_language_embs

class SASRec_backbone(nn.Module):
    def __init__(self, device, **key_words):

        super(SASRec_backbone, self).__init__()

        data_statis = pd.read_pickle(
            os.path.join(key_words["language_embs_path"], 'data_statis.df'))
        self.seq_len = data_statis['seq_size'][0]
        self.item_num = data_statis['item_num'][0]

        self.dropout = key_words["dropout_rate"]
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.hidden_dim = key_words["hidden_dim"]

        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.seq_len,
            embedding_dim=self.hidden_dim
        )

        self.emb_dropout = nn.Dropout(self.dropout)
        self.ln_1 = nn.LayerNorm(self.hidden_dim)
        self.ln_2 = nn.LayerNorm(self.hidden_dim)
        self.ln_3 = nn.LayerNorm(self.hidden_dim)

        self.mh_attn = MultiHeadAttention(self.hidden_dim, self.hidden_dim, key_words["num_heads"], self.dropout)

        self.feed_forward = PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, self.dropout)

    def embed_ID(self, x):

        pass

    def return_item_emb(self, ):

        pass

    def forward(self, sequences):

        inputs_emb = self.embed_ID(sequences)

        inputs_emb += self.positional_embeddings(torch.arange(self.seq_len).to(self.device))

        seq = self.emb_dropout(inputs_emb)

        mask = torch.ne(sequences, self.item_num).float().unsqueeze(-1).to(self.device)

        seq *= mask

        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))

        ff_out *= mask
        ff_out = self.ln_3(ff_out)

        logits = ff_out[:, -1].squeeze()
        return logits

    def predict(self, sequences):

        state_hidden = self.forward(sequences)
        item_embs = self.return_item_emb()
        scores = torch.matmul(state_hidden, item_embs[:-1].transpose(0, 1))
        return scores

    def calculate_ce_loss(self, sequences, target):
        seq_output = self.forward(sequences)
        item_embs = self.return_item_emb()
        logits = torch.matmul(seq_output, item_embs[:-1].transpose(0, 1))
        loss = self.ce_loss(logits, target)
        return loss

    def calculate_bce_loss(self, sequences, target, neg_ratio, emb_type="both"):
        batch_size = target.shape[0]
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
        target_neg = neg_samples.to(target.device)

        pos_embs = self.embed_ID(target)
        neg_embs = self.embed_ID(target_neg)

        log_feats = self.forward(sequences)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats.unsqueeze(1) * neg_embs).sum(dim=-1)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape,
                                                                                               device=self.device)
        loss = self.bce_loss(pos_logits, pos_labels)
        loss += self.bce_loss(neg_logits, neg_labels)

        return loss

    def calculate_infonce_loss(self, sequences, target, neg_ratio, temperature, emb_type="both"):

        batch_size = target.shape[0]

        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))

        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()

        expanded_sequences = sequences.view(batch_size, -1, 1).expand(batch_size, sequences.shape[1], neg_ratio).cpu()

        mask = neg_samples == expanded_target

        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target

        target_neg = neg_samples.to(
            target.device)

        pos_embs = self.embed_ID(target)

        neg_embs = self.embed_ID(target_neg)

        log_feats = self.forward(sequences)

        log_feats = F.normalize(log_feats, p=2, dim=-1)

        pos_embs = F.normalize(pos_embs, p=2, dim=-1)

        neg_embs = F.normalize(neg_embs, p=2, dim=-1)

        pos_logits = (log_feats * pos_embs).sum(dim=-1, keepdim=True)

        neg_logits = torch.bmm(neg_embs, log_feats.unsqueeze(-1)).squeeze(-1)

        logits = torch.cat([pos_logits, neg_logits], dim=-1)

        logits /= temperature

        labels = torch.zeros(batch_size, dtype=torch.long,
                             device=logits.device)

        loss = F.cross_entropy(logits, labels)

        return loss
class ProAlign(SASRec_backbone):

    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)

        self.key_words = key_words

        self.item_embeddings = nn.Embedding(
            num_embeddings=self.item_num + 1,
            embedding_dim=self.hidden_dim,
            padding_idx=self.item_num
        )

        self.num_prototypes = key_words.get('num_prototypes', 64)
        self.temperature = key_words.get('proto_temperature', 0.1)

        self.no_prototype = key_words.get('no_prototype', False)
        if self.no_prototype:
            print("[ProAlign] ABLATION MODE: Prototype mechanism DISABLED (w/o Prototype)")
        self.alpha = key_words.get('alpha', 0.1)
        self.beta_proto = key_words.get('beta_proto', 0.01)
        self.llm_dim = key_words.get('llm_dim', 3072)
        self.fusion_mode = key_words.get('fusion_mode', 'concat')
        self.semantic_weight = key_words.get('semantic_weight', 0.5)

        self.num_heads_proto = key_words.get('num_heads_proto', 1)
        self.head_dim = self.hidden_dim // self.num_heads_proto
        assert self.hidden_dim % self.num_heads_proto == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads_proto ({self.num_heads_proto})"

        self.prototypes = nn.Parameter(torch.zeros(self.num_prototypes, self.hidden_dim))

        self.adapter = nn.Sequential(
            nn.Linear(self.llm_dim, self.llm_dim // 4),
            nn.ReLU(),
            nn.Linear(self.llm_dim // 4, self.hidden_dim)
        )

        self.gate = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        self.macro_scale = nn.Parameter(torch.tensor(1.0))

        self.use_slsi = key_words.get('use_slsi', False)
        self.slsi_weight = key_words.get('slsi_weight', 0.3)
        self.slsi_context_aware = key_words.get('slsi_context_aware', False)

        self.use_attn_fusion = key_words.get('use_attn_fusion', True)
        if self.use_attn_fusion:
            self.proto_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=4,
                batch_first=True
            )

        self.warmup_epochs = key_words.get('warmup_epochs', 5)
        self.current_epoch = 0

        self.hard_neg_indices = None
        self.hard_neg_top_k = key_words.get('hard_neg_top_k', 10)
        self.item_intent_emb_for_align = None

        self.user_intent_emb = None
        self.item_intent_emb = None
        self.item_emb_reduced = None
        self.prototype_initialized = False

        self._inference_mode = False
        self._item_proto_cache = None
        self._fused_item_cache = None

        self.apply(self._init_bsarec_weights)
        self._init_special_params()

    def _init_proalign_weights(self):
        nn.init.normal_(self.item_embeddings.weight, 0, 0.02)
        for module in self.adapter:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.gate:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _init_bsarec_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)

        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)

    def _init_special_params(self):
        with torch.no_grad():
            if torch.all(self.prototypes == 0):
                nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

        with torch.no_grad():
            self.item_embeddings.weight[self.item_num].fill_(0.0)

    def load_intent_embeddings(self, user_intent_path, item_intent_path):
        if os.path.exists(user_intent_path):
            with open(user_intent_path, 'rb') as f:
                user_intent = pickle.load(f)
            self.user_intent_emb = torch.tensor(user_intent, dtype=torch.float32)
            print(f"[ProAlign] Loaded user intent embedding: {self.user_intent_emb.shape}")
        else:
            print(f"[ProAlign] Warning: User intent file not found: {user_intent_path}")

        if os.path.exists(item_intent_path):
            with open(item_intent_path, 'rb') as f:
                item_intent = pickle.load(f)
            self.item_intent_emb = item_intent
            print(f"[ProAlign] Loaded item intent embedding: {self.item_intent_emb.shape}")
        else:
            print(f"[ProAlign] Warning: Item intent file not found: {item_intent_path}")

    def initialize_item_embeddings(self):

        if self.item_intent_emb is None:
            print("[ProAlign] Warning: No item intent found, using random init.")
            return

        print("[ProAlign] Initializing ID Embeddings from LLM semantics...")

        if self.item_emb_reduced is None:
            pca = PCA(n_components=self.hidden_dim)
            self.item_emb_reduced = pca.fit_transform(self.item_intent_emb)
            print(f"  PCA: {self.item_intent_emb.shape} -> {self.item_emb_reduced.shape}")

        with torch.no_grad():
            pretrained_weight = torch.tensor(self.item_emb_reduced, dtype=torch.float32)
            pretrained_weight = F.normalize(pretrained_weight, p=2, dim=-1)

            padding = torch.zeros(1, self.hidden_dim)

            new_weight = torch.cat([pretrained_weight, padding], dim=0)

            if new_weight.shape[0] == self.item_embeddings.weight.shape[0]:
                self.item_embeddings.weight.data.copy_(new_weight)
                print('ID Embedding，not LLM Embedding', self.item_embeddings.weight.requires_grad)
                print(f"   ID Embeddings initialized! Shape: {new_weight.shape}")
            else:
                print(f"   Shape mismatch: {new_weight.shape} vs {self.item_embeddings.weight.shape}")

    def initialize_prototypes(self):
        if self.prototype_initialized:
            return

        if self.item_emb_reduced is None:
            if self.item_intent_emb is None:
                print("[ProAlign] Warning: No item intent, using random prototypes.")
                nn.init.normal_(self.prototypes.data, 0, 0.02)
                self.prototype_initialized = True
                return
            pca = PCA(n_components=self.hidden_dim)
            self.item_emb_reduced = pca.fit_transform(self.item_intent_emb)

        print(f"[ProAlign] Initializing prototypes with K-Means (K={self.num_prototypes})...")

        kmeans = KMeans(n_clusters=self.num_prototypes,
                        random_state=42,
                        n_init=10
                        )
        kmeans.fit(self.item_emb_reduced)
        centroids = kmeans.cluster_centers_

        with torch.no_grad():
            centroids_tensor = torch.tensor(centroids, dtype=torch.float32)
            self.prototypes.data = F.normalize(centroids_tensor, p=2, dim=-1)

        freeze_proto = self.key_words.get('freeze_prototypes', True)
        if freeze_proto:
            self.prototypes.requires_grad = False
            print(f"  ✅ Prototypes initialized and FROZEN. Shape: {self.prototypes.shape}")
        else:
            self.prototypes.requires_grad = True
            print(f"  ✅ Prototypes initialized and TRAINABLE. Shape: {self.prototypes.shape}")

        self.item_intent_emb = None
        self.prototype_initialized = True

    def precompute_hard_negatives(self, top_k=10):
        if self.item_intent_emb is None:
            print("[ProAlign] Warning: item_intent_emb not loaded, skip hard negative precomputation")
            return

        item_emb = self.item_intent_emb
        if isinstance(item_emb, np.ndarray):
            item_emb = torch.tensor(item_emb, dtype=torch.float32)
        item_emb_norm = F.normalize(item_emb, p=2, dim=-1)

        V = item_emb.size(0)
        batch_size = 1000

        hard_neg_indices = []
        for i in range(0, V, batch_size):
            end_i = min(i + batch_size, V)
            batch_emb = item_emb_norm[i:end_i]
            sim_matrix = torch.matmul(batch_emb, item_emb_norm.t())

            for j in range(end_i - i):
                sim_matrix[j, i + j] = -1e9

            _, topk_indices = torch.topk(sim_matrix, top_k, dim=-1)
            hard_neg_indices.append(topk_indices)

        self.hard_neg_indices = torch.cat(hard_neg_indices, dim=0)
        self.hard_neg_top_k = top_k
        self.item_intent_emb_for_align = item_emb
        print(f"[ProAlign] Hard negatives precomputed: {V} items × Top-{top_k}")

    def _proto_address(self, x, normalize_proto=False):
        orig_shape = x.shape
        D = orig_shape[-1]
        x_flat = x.view(-1, D)
        N = x_flat.size(0)

        P = self.prototypes
        if normalize_proto:
            P = F.normalize(P, p=2, dim=-1)

        if self.num_heads_proto == 1:
            score = torch.matmul(x_flat, P.t()) / self.temperature
            pi = F.softmax(score, dim=-1)
            r = torch.matmul(pi, P)
        else:
            x_h = x_flat.view(N, self.num_heads_proto, self.head_dim)
            P_h = P.view(self.num_prototypes, self.num_heads_proto, self.head_dim)
            scores = torch.einsum('nhd,khd->nhk', x_h, P_h) / self.temperature
            pi = F.softmax(scores, dim=-1)
            r_h = torch.einsum('nhk,khd->nhd', pi, P_h)
            r = r_h.reshape(N, D)

        r = r * self.macro_scale
        return r.view(*orig_shape)

    def embed_ID(self, x):
        return self.item_embeddings(x)

    def return_item_emb(self):
        if self._inference_mode and self._fused_item_cache is not None:
            return self._fused_item_cache

        if self.no_prototype:
            item_emb = self.item_embeddings.weight
            if self.fusion_mode == 'add':
                return item_emb
            else:
                r_dummy = torch.zeros_like(item_emb)
                return torch.cat([item_emb, r_dummy], dim=-1)

        if self.fusion_mode == 'add':
            return self._get_fused_item_emb_add()
        else:
            return self._get_fused_item_emb_concat()

    def _get_fused_item_emb_add(self):
        item_emb = self.item_embeddings.weight

        V = item_emb.size(0)
        if self.num_heads_proto == 1:
            score_all = torch.matmul(item_emb, self.prototypes.t()) / self.temperature
            pi_all = F.softmax(score_all, dim=-1)
            r_all = torch.matmul(pi_all, self.prototypes)
        else:
            item_heads = item_emb.view(V, self.num_heads_proto, self.head_dim)
            proto_heads = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)
            scores = torch.einsum('vhd,khd->vhk', item_heads, proto_heads) / self.temperature
            pi = F.softmax(scores, dim=-1)
            r_heads = torch.einsum('vhk,khd->vhd', pi, proto_heads)
            r_all = r_heads.reshape(V, self.hidden_dim)
        r_all = r_all * self.macro_scale

        return item_emb + self.semantic_weight * r_all

    def _get_fused_item_emb_concat(self):
        item_emb = self.item_embeddings.weight

        V = item_emb.size(0)
        if self.num_heads_proto == 1:
            score_all = torch.matmul(item_emb, self.prototypes.t()) / self.temperature
            pi_all = F.softmax(score_all, dim=-1)
            r_all = torch.matmul(pi_all, self.prototypes)
        else:
            item_heads = item_emb.view(V, self.num_heads_proto, self.head_dim)
            proto_heads = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)
            scores = torch.einsum('vhd,khd->vhk', item_heads, proto_heads) / self.temperature
            pi = F.softmax(scores, dim=-1)
            r_heads = torch.einsum('vhk,khd->vhd', pi, proto_heads)
            r_all = r_heads.reshape(V, self.hidden_dim)
        r_all = r_all * self.macro_scale

        return torch.cat([item_emb, r_all], dim=-1)

    def forward(self, sequences):
        inputs_emb = self.embed_ID(sequences)

        if self.use_slsi and not self.no_prototype:
            B, S, D = inputs_emb.shape

            slsi_mask = torch.ne(sequences, self.item_num).float().unsqueeze(-1).to(self.device)

            if self.slsi_context_aware:

                inputs_emb_masked = inputs_emb * slsi_mask

                cumsum = torch.cumsum(inputs_emb_masked, dim=1)

                counts = torch.cumsum(slsi_mask, dim=1).clamp_min(1.0)
                context_repr = cumsum / counts

                if self.num_heads_proto == 1:
                    slsi_score = torch.matmul(context_repr, self.prototypes.t()) / self.temperature
                    slsi_pi = F.softmax(slsi_score, dim=-1)
                    r_seq = torch.matmul(slsi_pi, self.prototypes)
                else:
                    context_heads = context_repr.view(B, S, self.num_heads_proto, self.head_dim)
                    proto_heads = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)
                    slsi_scores = torch.einsum('bshd,khd->bshk', context_heads, proto_heads) / self.temperature
                    slsi_pi = F.softmax(slsi_scores, dim=-1)
                    r_seq_heads = torch.einsum('bshk,khd->bshd', slsi_pi, proto_heads)
                    r_seq = r_seq_heads.reshape(B, S, self.hidden_dim)
            else:
                if self.num_heads_proto == 1:
                    slsi_score = torch.matmul(inputs_emb, self.prototypes.t()) / self.temperature
                    slsi_pi = F.softmax(slsi_score, dim=-1)
                    r_seq = torch.matmul(slsi_pi, self.prototypes)
                else:
                    inputs_heads = inputs_emb.view(B, S, self.num_heads_proto, self.head_dim)
                    proto_heads = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)
                    slsi_scores = torch.einsum('bshd,khd->bshk', inputs_heads, proto_heads) / self.temperature
                    slsi_pi = F.softmax(slsi_scores, dim=-1)
                    r_seq_heads = torch.einsum('bshk,khd->bshd', slsi_pi, proto_heads)
                    r_seq = r_seq_heads.reshape(B, S, self.hidden_dim)

            r_seq = r_seq * slsi_mask

            inputs_emb = inputs_emb + self.slsi_weight * r_seq

        inputs_emb += self.positional_embeddings(torch.arange(self.seq_len).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(sequences, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        h_u = ff_out[:, -1, :]

        if self.no_prototype:
            if self.fusion_mode == 'add':
                H_final = h_u
            else:
                r_u_dummy = torch.zeros_like(h_u)
                H_final = torch.cat([h_u, r_u_dummy], dim=-1)
            return H_final

        B = h_u.size(0)

        if self.use_attn_fusion and hasattr(self, 'proto_attn'):
            query = h_u.unsqueeze(1)
            keys = self.prototypes.unsqueeze(0).expand(B, -1, -1)

            r_dynamic, _ = self.proto_attn(query, keys, keys)
            r_u = r_dynamic.squeeze(1)
        elif self.num_heads_proto == 1:
            score_stu = torch.matmul(h_u, self.prototypes.t()) / self.temperature
            pi_stu = F.softmax(score_stu, dim=-1)
            r_u = torch.matmul(pi_stu, self.prototypes)
        else:
            h_u_heads = h_u.view(B, self.num_heads_proto, self.head_dim)
            proto_heads = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)

            scores = torch.einsum('bhd,khd->bhk', h_u_heads, proto_heads) / self.temperature
            pi = F.softmax(scores, dim=-1)

            r_u_heads = torch.einsum('bhk,khd->bhd', pi, proto_heads)

            r_u = r_u_heads.reshape(B, self.hidden_dim)

        r_u = r_u * self.macro_scale

        if self.fusion_mode == 'add':
            H_final = h_u + self.semantic_weight * r_u
        else:
            concat_feat = torch.cat([h_u, r_u], dim=-1)
            g = self.gate(concat_feat)
            H_final = torch.cat([h_u, g * r_u], dim=-1)

        return H_final

    def predict(self, sequences):
        H_final = self.forward(sequences)
        item_embs = self.return_item_emb()

        if self.fusion_mode == 'add':
            item_embs = item_embs[:-1]
        else:
            item_embs = item_embs[:-1]

        scores = torch.matmul(H_final, item_embs.t())
        return scores

    def calculate_loss_with_align(self, sequences, target, user_ids, neg_ratio, temperature):
        H_final = self.forward(sequences)

        item_embs = self.return_item_emb()

        if self.fusion_mode == 'add':
            pos_embs = self._get_target_fused_emb_add(target)
        else:
            pos_embs = self._get_target_fused_emb_concat(target)

        batch_size = target.shape[0]
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
        neg_samples = neg_samples.to(target.device)

        if self.fusion_mode == 'add':
            neg_embs = self._get_target_fused_emb_add(neg_samples)
        else:
            neg_embs = self._get_target_fused_emb_concat(neg_samples)

        H_final_norm = F.normalize(H_final, p=2, dim=-1)
        pos_embs_norm = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs_norm = F.normalize(neg_embs, p=2, dim=-1)

        pos_logits = (H_final_norm * pos_embs_norm).sum(dim=-1, keepdim=True)
        neg_logits = torch.bmm(neg_embs_norm, H_final_norm.unsqueeze(-1)).squeeze(-1)
        logits = torch.cat([pos_logits, neg_logits], dim=-1) / temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        L_rec = F.cross_entropy(logits, labels)

        L_align = torch.tensor(0.0, device=sequences.device)
        align_mode = self.key_words.get('align_mode', 'infonce')
        cl_temperature = self.key_words.get('cl_temperature', 1.0)

        if self.user_intent_emb is not None and user_ids is not None:
            z_next = self.user_intent_emb[user_ids.cpu()].to(sequences.device)
            z_next_proj = self.adapter(z_next)

            h_u = H_final[:, :self.hidden_dim]

            h_u_norm = F.normalize(h_u, p=2, dim=-1)
            z_next_norm = F.normalize(z_next_proj, p=2, dim=-1)

            if align_mode == 'kl':
                proto_norm = F.normalize(self.prototypes, p=2, dim=-1)

                score_stu = torch.matmul(h_u_norm, proto_norm.t()) / self.temperature

                score_tea = torch.matmul(z_next_norm, proto_norm.t()) / self.temperature
                pi_tea = F.softmax(score_tea, dim=-1)

                log_pi_stu = F.log_softmax(score_stu, dim=-1)
                L_align = F.kl_div(log_pi_stu, pi_tea, reduction='batchmean')

            elif align_mode == 'infonce':

                batch_size_align = h_u_norm.size(0)

                sim_matrix = torch.matmul(h_u_norm, z_next_norm.t()) / cl_temperature

                labels_align = torch.arange(batch_size_align, device=sequences.device)

                loss_h2z = F.cross_entropy(sim_matrix, labels_align)
                loss_z2h = F.cross_entropy(sim_matrix.t(), labels_align)

                L_align = (loss_h2z + loss_z2h) / 2

        if self.no_prototype:
            L_cluster = torch.tensor(0.0, device=sequences.device)
        else:
            e_target = self.item_embeddings(target)
            e_target_norm = F.normalize(e_target, p=2, dim=-1)
            proto_norm = F.normalize(self.prototypes, p=2, dim=-1)

            B_proto = e_target.size(0)
            if self.num_heads_proto == 1:
                score_target = torch.matmul(e_target_norm, proto_norm.t()) / self.temperature
                pi_target = F.softmax(score_target, dim=-1)
                r_target = torch.matmul(pi_target, self.prototypes)
            else:
                e_target_heads = e_target_norm.view(B_proto, self.num_heads_proto, self.head_dim)
                proto_heads = proto_norm.view(self.num_prototypes, self.num_heads_proto, self.head_dim)
                scores = torch.einsum('bhd,khd->bhk', e_target_heads, proto_heads) / self.temperature
                pi_target = F.softmax(scores, dim=-1)
                proto_unnorm = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)
                r_heads = torch.einsum('bhk,khd->bhd', pi_target, proto_unnorm)
                r_target = r_heads.reshape(B_proto, self.hidden_dim)
            r_target = r_target * self.macro_scale

            L_cluster = F.mse_loss(e_target, r_target.detach())

        loss = L_rec + self.alpha * L_align + self.beta_proto * L_cluster

        return loss

    def _get_target_fused_emb_add(self, target):
        e_i = self.item_embeddings(target)

        if self.no_prototype:
            return e_i

        if self.num_heads_proto == 1:
            score = torch.matmul(e_i, self.prototypes.t()) / self.temperature
            pi = F.softmax(score, dim=-1)
            r_i = torch.matmul(pi, self.prototypes)
        else:
            orig_shape = e_i.shape[:-1]
            e_i_flat = e_i.view(-1, self.hidden_dim)
            N = e_i_flat.size(0)

            e_i_heads = e_i_flat.view(N, self.num_heads_proto, self.head_dim)
            proto_heads = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)
            scores = torch.einsum('nhd,khd->nhk', e_i_heads, proto_heads) / self.temperature
            pi = F.softmax(scores, dim=-1)
            r_heads = torch.einsum('nhk,khd->nhd', pi, proto_heads)
            r_i_flat = r_heads.reshape(N, self.hidden_dim)
            r_i = r_i_flat.view(*orig_shape, self.hidden_dim)
        r_i = r_i * self.macro_scale

        return e_i + self.semantic_weight * r_i

    def _get_target_fused_emb_concat(self, target):
        e_i = self.item_embeddings(target)

        if self.no_prototype:
            r_i_dummy = torch.zeros_like(e_i)
            return torch.cat([e_i, r_i_dummy], dim=-1)

        if self.num_heads_proto == 1:
            score = torch.matmul(e_i, self.prototypes.t()) / self.temperature
            pi = F.softmax(score, dim=-1)
            r_i = torch.matmul(pi, self.prototypes)
        else:
            orig_shape = e_i.shape[:-1]
            e_i_flat = e_i.view(-1, self.hidden_dim)
            N = e_i_flat.size(0)

            e_i_heads = e_i_flat.view(N, self.num_heads_proto, self.head_dim)
            proto_heads = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)
            scores = torch.einsum('nhd,khd->nhk', e_i_heads, proto_heads) / self.temperature
            pi = F.softmax(scores, dim=-1)
            r_heads = torch.einsum('nhk,khd->nhd', pi, proto_heads)
            r_i_flat = r_heads.reshape(N, self.hidden_dim)
            r_i = r_i_flat.view(*orig_shape, self.hidden_dim)
        r_i = r_i * self.macro_scale

        return torch.cat([e_i, r_i], dim=-1)

    def _compute_align_cluster_loss(self, sequences, target, h_u):
        L_align = torch.tensor(0.0, device=sequences.device)
        if self.user_intent_emb is not None:
            pass

        e_target = self.item_embeddings(target)
        e_target_norm = F.normalize(e_target, p=2, dim=-1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)

        B = e_target.size(0)
        if self.num_heads_proto == 1:
            score_target = torch.matmul(e_target_norm, proto_norm.t()) / self.temperature
            pi_target = F.softmax(score_target, dim=-1)
            r_target = torch.matmul(pi_target, self.prototypes)
        else:
            e_target_heads = e_target_norm.view(B, self.num_heads_proto, self.head_dim)
            proto_heads = proto_norm.view(self.num_prototypes, self.num_heads_proto, self.head_dim)
            scores = torch.einsum('bhd,khd->bhk', e_target_heads, proto_heads) / self.temperature
            pi_target = F.softmax(scores, dim=-1)
            proto_unnorm = self.prototypes.view(self.num_prototypes, self.num_heads_proto, self.head_dim)
            r_heads = torch.einsum('bhk,khd->bhd', pi_target, proto_unnorm)
            r_target = r_heads.reshape(B, self.hidden_dim)
        r_target = r_target * self.macro_scale
        L_cluster = F.mse_loss(e_target, r_target.detach())

        return L_align, L_cluster

    def calculate_ce_loss(self, sequences, target):
        H_final = self.forward(sequences)

        item_embs = self.return_item_emb()[:-1]
        logits = torch.matmul(H_final, item_embs.t())
        L_rec = self.ce_loss(logits, target)

        h_u = H_final[:, :self.hidden_dim] if self.fusion_mode == 'concat' else H_final

        L_align, L_cluster = self._compute_align_cluster_loss(sequences, target, h_u)

        loss = L_rec + self.alpha * L_align + self.beta_proto * L_cluster
        return loss

    def calculate_bce_loss(self, sequences, target, neg_ratio):
        H_final = self.forward(sequences)

        batch_size = target.shape[0]
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
        neg_samples = neg_samples.to(target.device)

        if self.fusion_mode == 'add':
            pos_embs = self._get_target_fused_emb_add(target)
            neg_embs = self._get_target_fused_emb_add(neg_samples)
        else:
            pos_embs = self._get_target_fused_emb_concat(target)
            neg_embs = self._get_target_fused_emb_concat(neg_samples)

        pos_logits = (H_final * pos_embs).sum(dim=-1)
        neg_logits = (H_final.unsqueeze(1) * neg_embs).sum(dim=-1)

        pos_labels = torch.ones(pos_logits.shape, device=self.device)
        neg_labels = torch.zeros(neg_logits.shape, device=self.device)

        L_rec = self.bce_loss(pos_logits, pos_labels) + self.bce_loss(neg_logits, neg_labels)

        h_u = H_final[:, :self.hidden_dim] if self.fusion_mode == 'concat' else H_final

        L_align, L_cluster = self._compute_align_cluster_loss(sequences, target, h_u)

        loss = L_rec + self.alpha * L_align + self.beta_proto * L_cluster
        return loss

    def calculate_infonce_loss(self, sequences, target, neg_ratio, temperature):
        H_final = self.forward(sequences)

        batch_size = target.shape[0]
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
        neg_samples = neg_samples.to(target.device)

        if self.fusion_mode == 'add':
            pos_embs = self._get_target_fused_emb_add(target)
            neg_embs = self._get_target_fused_emb_add(neg_samples)
        else:
            pos_embs = self._get_target_fused_emb_concat(target)
            neg_embs = self._get_target_fused_emb_concat(neg_samples)

        H_final_norm = F.normalize(H_final, p=2, dim=-1)
        pos_embs_norm = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs_norm = F.normalize(neg_embs, p=2, dim=-1)

        pos_logits = (H_final_norm * pos_embs_norm).sum(dim=-1, keepdim=True)
        neg_logits = torch.bmm(neg_embs_norm, H_final_norm.unsqueeze(-1)).squeeze(-1)

        logits = torch.cat([pos_logits, neg_logits], dim=-1) / temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        L_rec = F.cross_entropy(logits, labels)

        h_u = H_final[:, :self.hidden_dim] if self.fusion_mode == 'concat' else H_final

        L_align, L_cluster = self._compute_align_cluster_loss(sequences, target, h_u)

        loss = L_rec + self.alpha * L_align + self.beta_proto * L_cluster
        return loss

    def calculate_rasd_loss(self, sequences, sim_seqs, user_sim_func='cl'):
        B, K, S = sim_seqs.shape

        h_u = self.forward(sequences)

        sim_seqs_flat = sim_seqs.view(B * K, S)
        h_sim = self.forward(sim_seqs_flat)

        h_sim = h_sim.detach()

        h_sim = h_sim.view(B, K, -1)
        h_sim_avg = h_sim.mean(dim=1)

        if user_sim_func == 'cl':
            h_u_norm = F.normalize(h_u, p=2, dim=-1)
            h_sim_norm = F.normalize(h_sim_avg, p=2, dim=-1)
            rasd_loss = 1.0 - (h_u_norm * h_sim_norm).sum(dim=-1).mean()
        elif user_sim_func == 'kd':
            rasd_loss = F.mse_loss(h_u, h_sim_avg)
        else:
            raise ValueError(f"Unknown user_sim_func: {user_sim_func}")

        return rasd_loss

    def precompute_for_inference(self):
        if self.no_prototype:
            print("[ProAlign-Efficient] Prototype disabled, skip precompute")
            self._inference_mode = True
            return

        with torch.no_grad():
            item_emb = self.item_embeddings.weight
            V = item_emb.size(0)
            P = self.prototypes

            if self.num_heads_proto == 1:
                score_all = torch.matmul(item_emb, P.t()) / self.temperature
                pi_all = F.softmax(score_all, dim=-1)
                r_all = torch.matmul(pi_all, P)
            else:
                item_heads = item_emb.view(V, self.num_heads_proto, self.head_dim)
                proto_heads = P.view(self.num_prototypes, self.num_heads_proto, self.head_dim)
                scores = torch.einsum('vhd,khd->vhk', item_heads, proto_heads) / self.temperature
                pi = F.softmax(scores, dim=-1)
                r_heads = torch.einsum('vhk,khd->vhd', pi, proto_heads)
                r_all = r_heads.reshape(V, self.hidden_dim)

            r_all = r_all * self.macro_scale
            self._item_proto_cache = r_all

            if self.fusion_mode == 'add':
                self._fused_item_cache = item_emb + self.semantic_weight * r_all
            else:
                self._fused_item_cache = torch.cat([item_emb, r_all], dim=-1)

            self._inference_mode = True
            print(f"[ProAlign-Efficient] Inference cache ready: {V} items, mode={self.fusion_mode}")

    def clear_inference_cache(self):
        self._item_proto_cache = None
        self._fused_item_cache = None
        self._inference_mode = False

class PPDScheduler:

    def __init__(self, model, total_epochs, warmup_ratio=0.3, transition_ratio=0.7,
                 ema_decay=0.99, verbose=True):
        self.model = model
        self.total_epochs = total_epochs
        self.warmup_epochs = int(total_epochs * warmup_ratio)
        self.transition_epochs = int(total_epochs * transition_ratio)
        self.ema_decay = ema_decay
        self.verbose = verbose

        self.prototype_shadow = None

        self.current_phase = None

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            self._set_phase(1, epoch)
            self.model.prototypes.requires_grad = False

        elif epoch < self.transition_epochs:
            self._set_phase(2, epoch)
            self.model.prototypes.requires_grad = True

        else:
            self._set_phase(3, epoch)
            self.model.prototypes.requires_grad = True

            with torch.no_grad():
                if self.prototype_shadow is None:
                    self.prototype_shadow = self.model.prototypes.data.clone()
                else:
                    self.prototype_shadow = (
                            self.ema_decay * self.prototype_shadow +
                            (1 - self.ema_decay) * self.model.prototypes.data
                    )
                    self.model.prototypes.data = self.prototype_shadow.clone()

    def _set_phase(self, phase, epoch):
        if self.current_phase != phase:
            self.current_phase = phase
            if self.verbose:
                phase_names = {
                    1: "Phase 1: FROZEN (Warmup)",
                    2: "Phase 2: TRAINABLE (Transition)",
                    3: "Phase 3: TRAINABLE + EMA (Refinement)"
                }
                print(f"[PPD] Epoch {epoch}: Entering {phase_names[phase]}")

    def get_current_phase(self):
        return self.current_phase

    def get_phase_info(self):
        return {
            "warmup_epochs": self.warmup_epochs,
            "transition_epochs": self.transition_epochs,
            "total_epochs": self.total_epochs,
            "ema_decay": self.ema_decay
        }

