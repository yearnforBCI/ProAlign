import os
import time
import torch
import random
import numpy as np
import pandas as pd
import argparse
import logging
import pickle
from torch import nn
import torch.nn.functional as F
import datetime
from scipy.sparse import csr_matrix

from torch.utils.data import Dataset, DataLoader

from models.backbone_SASRec import ProAlign
from utils import evaluate, evaluate_diff

def generate_rating_matrix(df, num_users, num_items, padding_id):
   
    row, col, data = [], [], []
    
    for idx, r in df.iterrows():
        user_id = r['user_id']
        seq = r['seq']
        for item in seq:
            if item != padding_id:
                row.append(user_id)
                col.append(item)
                data.append(1)
    
    return csr_matrix((np.array(data), (np.array(row), np.array(col))), 
                       shape=(num_users, num_items))

class SeqDataset(Dataset):
    def __init__(self, data):
       
        self.seq_data = [torch.tensor(seq, dtype=torch.long) for seq in data['seq']]
        self.len_seq_data = [torch.tensor(len_seq, dtype=torch.long) for len_seq in data['len_seq']]
        self.next_data = [torch.tensor(next_val, dtype=torch.long) for next_val in data['next']]
        
        if 'user_id' in data.columns:
            self.user_id_data = [torch.tensor(uid, dtype=torch.long) for uid in data['user_id']]
            self.has_user_id = True
        else:
            self.user_id_data = None
            self.has_user_id = False

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        sample = {
            'seq': self.seq_data[idx],
            'len_seq': self.len_seq_data[idx],
            'next': self.next_data[idx]
        }
        if self.has_user_id:
            sample['user_id'] = self.user_id_data[idx]
        return sample

class SeqDatasetWithSimUser(SeqDataset):
    
    
    def __init__(self, data, sim_users, all_user_seqs, sim_user_num, seq_len, padding_id):
        super().__init__(data)
        self.sim_users = sim_users
        self.all_user_seqs = all_user_seqs
        self.sim_user_num = sim_user_num
        self.seq_len = seq_len
        self.padding_id = padding_id
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        
        if self.has_user_id:
            user_id = sample['user_id'].item()
        else:
            user_id = idx
        
        sim_user_ids = self.sim_users[user_id][:self.sim_user_num]
        sim_seqs = []
        for sim_uid in sim_user_ids:
            sim_seq = self._get_user_seq(sim_uid)
            sim_seqs.append(sim_seq)
        
        sample['sim_seqs'] = torch.tensor(np.array(sim_seqs), dtype=torch.long)
        return sample
    
    def _get_user_seq(self, user_id):
        user_seq = self.all_user_seqs[user_id]
        return np.array(user_seq, dtype=np.int32)

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

logging.getLogger().setLevel(logging.INFO)

def setup_seed(seed): 
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_local_time():
    
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur

def set_logger(log_path):
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="Run ProAlign.")
    parser.add_argument('--random_seed', type=int, default=22)
    
    parser.add_argument('--model_type', type=str, default="ProAlign")
    
    parser.add_argument('--train_name', default=None, type=str)
    parser.add_argument('--output_dir', default='./output/', type=str)
    
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--lr_delay_rate', type=float, default=0.99)
    parser.add_argument('--lr_delay_epoch', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=500, help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='Beauty', help='Dataset name')
    parser.add_argument('--cuda', type=int, default=0, help='cuda device.')
    parser.add_argument('--l2_decay', type=float, default=1e-6, help='l2 loss reg coef.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout')
    
    parser.add_argument('--loss_type', type=str, default="infoNCE")
    parser.add_argument('--neg_ratio', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=0.07)
    
    parser.add_argument('--use_rasd', type=str2bool, default=True)
    parser.add_argument('--alpha_rasd', type=float, default=0.1)
    parser.add_argument('--sim_user_num', type=int, default=10)
    parser.add_argument('--user_sim_func', type=str, default='kd', choices=['cl', 'kd'])
    
    parser.add_argument('--language_model_type', default="3large", type=str)
    parser.add_argument('--language_embs_scale', default=40, type=int)

    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Number of hidden factors, i.e., ID embedding size.')
    parser.add_argument('--ID_embs_init_type', type=str, default="normal")
    
    parser.add_argument('--efficient_inference', action='store_true')
    parser.add_argument('--num_prototypes', type=int, default=64)
    parser.add_argument('--proto_temperature', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta_proto', type=float, default=0.01)
    parser.add_argument('--llm_dim', type=int, default=3072)
    parser.add_argument('--fusion_mode', type=str, default='concat')
    parser.add_argument('--semantic_weight', type=float, default=0.5)
    parser.add_argument('--semantic_init', type=str2bool, default=True)
    parser.add_argument('--align_mode', type=str, default='infonce')
    parser.add_argument('--cl_temperature', type=float, default=1.0)
    parser.add_argument('--num_heads_proto', type=int, default=1)
    parser.add_argument('--use_slsi', type=str2bool, default=False)
    parser.add_argument('--slsi_weight', type=float, default=0.3)
    parser.add_argument('--slsi_context_aware', type=str2bool, default=False)
    parser.add_argument('--freeze_prototypes', type=str2bool, default=True)
    parser.add_argument('--hard_neg_top_k', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--use_attn_fusion', type=str2bool, default=True)
    parser.add_argument('--use_ppd', type=str2bool, default=False)
    parser.add_argument('--ppd_warmup_ratio', type=float, default=0.3)
    parser.add_argument('--ppd_transition_ratio', type=float, default=0.7)
    parser.add_argument('--ppd_ema_decay', type=float, default=0.99)
    parser.add_argument('--freeze_embedding_epochs', type=int, default=0)
    parser.add_argument('--use_user_intent', type=str2bool, default=True)
    parser.add_argument('--no_prototype', type=str2bool, default=False)
    
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    setup_seed(args.random_seed)
    
    if args.train_name is None:
        args.train_name = get_local_time()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    log_path = os.path.join(args.output_dir, args.train_name + '.log')
    logger = set_logger(log_path)
    logger.info(f"Experiment: {args.train_name}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    key_words = vars(args)
    
    if '/' in args.data or os.path.isabs(args.data):
        data_directory = args.data if args.data.startswith('./') or os.path.isabs(args.data) else './' + args.data
    else:
        data_directory = './data/' + args.data
    
    data_name = os.path.basename(data_directory.rstrip('/'))
    
    model_directory = './saved/' + data_name + '/'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    checkpoint_path = os.path.join(model_directory, args.train_name + '.pt')
    key_words["language_embs_path"] = data_directory

    ppd_scheduler = None
    
    model = ProAlign(device, **key_words).to(device)
    if args.efficient_inference:
        print("[ProAlign] Efficient inference mode enabled (will apply during evaluation)")
    user_intent_path = os.path.join(data_directory, 'usr_intent_emb.pkl')
    item_intent_path = os.path.join(data_directory, 'itm_intent_emb.pkl')
    model.load_intent_embeddings(user_intent_path, item_intent_path)
    
    if not args.use_user_intent:
        model.user_intent_emb = None
        logger.info("[ProAlign] Item-only mode: User intent embedding DISABLED (ablation study)")
        logger.info("[ProAlign] L_align will be 0, only using Item-side LLM info")
    else:
        logger.info("[ProAlign] Full mode: Using both Item + User side LLM info")
    
    if args.semantic_init:
        model.initialize_item_embeddings()
        logger.info("[ProAlign] ID Embeddings initialized from LLM semantics!")
    
    if hasattr(model, 'precompute_hard_negatives'):
        hard_neg_top_k = getattr(args, 'hard_neg_top_k', 10)
        model.precompute_hard_negatives(top_k=hard_neg_top_k)
        logger.info(f"[ProAlign] Semantic hard negatives precomputed (Top-{hard_neg_top_k})")
    
    model.initialize_prototypes()
    
    model.prototypes.data = model.prototypes.data.to(device)
    logger.info(f"[ProAlign] Prototypes moved to {device}")
    
    ppd_scheduler = None
    if args.use_ppd:
        from models.backbone_SASRec import PPDScheduler
        ppd_scheduler = PPDScheduler(
            model=model,
            total_epochs=args.epoch,
            warmup_ratio=args.ppd_warmup_ratio,
            transition_ratio=args.ppd_transition_ratio,
            ema_decay=args.ppd_ema_decay,
            verbose=True
        )
        logger.info(f"[PPD] Scheduler initialized: warmup={args.ppd_warmup_ratio}, transition={args.ppd_transition_ratio}, ema={args.ppd_ema_decay}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total Parameters: {total_params}")
    logger.info(f"Trainable Parameters: {trainable_params}")
    logger.info(str(args))

    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    train_data.reset_index(inplace=True, drop=True)
    
    if args.use_rasd:
        sim_user_path = os.path.join(data_directory, 'sim_user_100.pkl')
        if os.path.exists(sim_user_path):
            sim_users = pickle.load(open(sim_user_path, 'rb'))
            logger.info(f"[RASD] Loaded sim_user_100.pkl: shape={sim_users.shape}")
            
            all_user_seqs = [seq for seq in train_data['seq']]
            logger.info(f"[RASD] Built all_user_seqs: {len(all_user_seqs)} users")
            
            seq_len = len(train_data['seq'].iloc[0])
            data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
            item_num = data_statis['item_num'].iloc[0]
            logger.info(f"[RASD] seq_len={seq_len}, item_num={item_num}")
            
            train_dataset = SeqDatasetWithSimUser(
                data=train_data,
                sim_users=sim_users,
                all_user_seqs=all_user_seqs,
                sim_user_num=args.sim_user_num,
                seq_len=seq_len,
                padding_id=item_num
            )
            logger.info(f"[RASD] Using SeqDatasetWithSimUser with K={args.sim_user_num}")
        else:
            logger.warning(f"[RASD] sim_user_100.pkl not found, falling back to standard SeqDataset")
            train_dataset = SeqDataset(train_data)
    else:
        train_dataset = SeqDataset(train_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_data = pd.read_pickle(os.path.join(data_directory, 'val_data.df'))
    val_data.reset_index(inplace=True, drop=True)
    val_dataset = SeqDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    test_data = pd.read_pickle(os.path.join(data_directory, 'test_data.df'))
    test_data.reset_index(inplace=True, drop=True)
    test_dataset = SeqDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
    item_num = data_statis['item_num'].iloc[0]
    padding_id = item_num
    
    num_users = len(val_data)
    num_items_for_matrix = item_num
    
    valid_rating_matrix = generate_rating_matrix(val_data, num_users, num_items_for_matrix, padding_id)
    test_rating_matrix = generate_rating_matrix(test_data, num_users, num_items_for_matrix, padding_id)
    
    logger.info(f"[Rating Matrix] valid: {valid_rating_matrix.shape}, nnz={valid_rating_matrix.nnz}")
    logger.info(f"[Rating Matrix] test: {test_rating_matrix.shape}, nnz={test_rating_matrix.nnz}")

    best_ndcg20 = 0
    patience = 50
    counter = 0
    step = 0
    T = 0.0
    logger.info("Loading Data Done.")
    
    t0 = time.time()
    val_ndcg20 = evaluate(model, val_loader, device, rating_matrix=valid_rating_matrix)
    t1 = time.time() - t0
    logger.info(f"\n using {t1}s, Eval Time Cost {T}s.")

    for epoch in range(args.epoch):
        model.train()
        
        if args.model_type in ["ProAlign", "ProAlign_GRU", "ProAlign_BERT4Rec"] and args.use_ppd and ppd_scheduler is not None:
            ppd_scheduler.step(epoch)
        
        if args.model_type in ["ProAlign", "ProAlign_GRU", "ProAlign_BERT4Rec"]:
            if hasattr(model, 'current_epoch'):
                model.current_epoch = epoch
                if epoch == getattr(args, 'warmup_epochs', 5):
                    logger.info(f"[Curriculum] Epoch {epoch}: Hard negatives ACTIVATED! 🚀")
        
        if args.freeze_embedding_epochs > 0 and args.model_type in ["ProAlign", "ProAlign_GRU", "ProAlign_BERT4Rec"]:
            if epoch < args.freeze_embedding_epochs:
                if hasattr(model, 'ID_embeddings'):
                    for p in model.ID_embeddings.parameters():
                        p.requires_grad = False
                if epoch == 0:
                    logger.info(f"[Freeze Embedding] Epoch {epoch}: ID Embedding FROZEN (warmup {args.freeze_embedding_epochs} epochs)")
            else:
                if hasattr(model, 'ID_embeddings'):
                    for p in model.ID_embeddings.parameters():
                        p.requires_grad = True
                if epoch == args.freeze_embedding_epochs:
                    logger.info(f"[Freeze Embedding] Epoch {epoch}: ID Embedding UNFROZEN")
        
        for batch in train_loader:
            
            batch_size = len(batch['seq'])
            seq = batch['seq'].to(device)
            target = batch['next'].to(device)
            
            optimizer.zero_grad()
            
            if args.loss_type == "CE":
                loss = model.calculate_ce_loss(seq, target)
            elif args.loss_type == "BCE":
                loss = model.calculate_bce_loss(seq, target, args.neg_ratio)
            elif args.loss_type == "infoNCE":
                if args.model_type in ["ProAlign", "ProAlign_GRU", "ProAlign_BERT4Rec"]:
                    user_ids = batch.get('user_id', None)
                    if user_ids is not None:
                        user_ids = user_ids.to(device)
                    if step == 0:
                        logger.info(f"[DEBUG] user_ids is None: {user_ids is None}")
                        if user_ids is not None:
                            logger.info(f"[DEBUG] user_ids shape: {user_ids.shape}, first 5: {user_ids[:5]}")
                    loss = model.calculate_loss_with_align(seq, target, user_ids, args.neg_ratio, args.temperature)
            
            if args.model_type in ["ProAlign", "ProAlign_GRU", "ProAlign_BERT4Rec"] and args.use_rasd:
                sim_seqs = batch.get('sim_seqs', None)
                if sim_seqs is not None:
                    sim_seqs = sim_seqs.to(device)
                    rasd_loss = model.calculate_rasd_loss(seq, sim_seqs, args.user_sim_func)
                    loss = loss + args.alpha_rasd * rasd_loss
                    
                    if step == 0:
                        logger.info(f"[ProAlign RASD] sim_seqs shape: {sim_seqs.shape}")
                        logger.info(f"[ProAlign RASD] rasd_loss: {rasd_loss.item():.4f}")

            loss.backward()
            optimizer.step()
            step += 1
        
        logger.info(f"loss in epoch {epoch} iteration {step}: {loss.item()}")
        
        if (epoch+1) % 50 == 0:
            _ = evaluate(model, train_loader, device)
        
        if (epoch+1) % 1 == 0:
            model.eval()
            if args.efficient_inference and hasattr(model, 'precompute_for_inference'):
                model.precompute_for_inference()
            logger.info('-------------------------- EVALUATE PHRASE --------------------------')
            t0 = time.time()
            val_ndcg20 = evaluate(model, val_loader, device, rating_matrix=valid_rating_matrix)
            t1 = time.time() - t0
            logger.info(f"\n using {t1}s, Eval Time Cost {T}s.")

            model.train()
            if args.efficient_inference and hasattr(model, 'clear_inference_cache'):
                model.clear_inference_cache()
            tv_ndcg20 = val_ndcg20 
            
            if tv_ndcg20 > best_ndcg20:
                best_ndcg20 = tv_ndcg20
                counter = 0
                logger.info(f"\n best NDCG@20 is updated to {best_ndcg20} at epoch {epoch}")
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Model saved to {checkpoint_path}")
            else:
                counter += 1
                if counter >= patience:
                    logger.info("Early stopping")
                    break
            logger.info('----------------------------------------------------------------')
    
    logger.info(f"Loading best model from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
    items_emb = model.return_item_emb()
    
    model.eval()
    if args.efficient_inference and hasattr(model, 'precompute_for_inference'):
        model.precompute_for_inference()
    logger.info('-------------------------- TEST RESULTS --------------------------')
    _ = evaluate(model, test_loader, device, rating_matrix=test_rating_matrix)
    logger.info(f"Experiment: {args.train_name}")
    logger.info("Done.")