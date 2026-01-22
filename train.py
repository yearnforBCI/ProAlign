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
# ==================== [NEW] 时间戳和日志功能 ====================
import datetime
# ==================== [END NEW] ====================
# ==================== [NEW] 评测屏蔽矩阵（BSARec 风格）====================
from scipy.sparse import csr_matrix
# ==================== [END NEW] ====================

from torch.utils.data import Dataset, DataLoader

# from models.backbone_SASRec import SASRec,MoRec, WhitenRec, UniSRec, RLMRec, LLMESR, LLMInit, WhitenRec, AlphaFuse
# ==================== [NEW] 添加 ProAlign 导入 ====================
from models.backbone_SASRec import SASRec, MoRec, WhitenRec, UniSRec, RLMRec, LLMESR, LLMInit, AlphaFuse, ProAlign
from models.backbone_SASRec_efficiency import ProAlign as ProAlign_Efficient  # 效率优化版本
# ==================== [END NEW] ====================
# ==================== [NEW 2024-12-15] 添加 IRLLRec 导入 ====================
from models.backbone_SASRec import IRLLRec
# ==================== [END NEW] ====================
# ==================== [NEW] 添加 GRU4Rec 系列模型导入 ====================
from models.backbone_GRU4Rec import (
    GRU4Rec, MoRec_GRU, UniSRec_GRU, LLMInit_GRU, AlphaFuse_GRU,
    WhitenRec_GRU, RLMRec_GRU, LLMESR_GRU, ProAlign_GRU
)
# ==================== [END NEW] ====================
# ==================== [NEW 2024-12-15] 添加 IRLLRec_GRU 导入 ====================
from models.backbone_GRU4Rec import IRLLRec_GRU
# ==================== [END NEW] ====================
# ==================== [NEW 2024-12-15] 添加 BERT4Rec 系列模型导入 ====================
from models.backbone_BERT4Rec import (
    BERT4Rec, AlphaFuse_BERT4Rec, MoRec_BERT4Rec, WhitenRec_BERT4Rec,
    LLMInit_BERT4Rec, UniSRec_BERT4Rec, RLMRec_BERT4Rec, LLMESR_BERT4Rec,
    ProAlign_BERT4Rec, IRLLRec_BERT4Rec
)
# ==================== [END NEW] ====================
from utils import evaluate, evaluate_diff


# ==================== [NEW] 评测屏蔽矩阵生成函数 ====================
def generate_rating_matrix(df, num_users, num_items, padding_id):
    """
    从 DataFrame 的 seq 列生成评测用屏蔽矩阵
    
    与 BSARec 逻辑等价：
    - BSARec: 屏蔽 user_seq[:-2] (valid) 或 user_seq[:-1] (test)
    - AlphaFuse: val_data['seq'] 已经是 items[:-2]，test_data['seq'] 是 items[:-1]
    - 因此直接从 seq 列提取非 padding 物品即可
    
    Args:
        df: DataFrame，包含 'user_id' 和 'seq' 列
        num_users: 用户总数
        num_items: 物品总数 (item_num + 1, 包含 padding 列)
        padding_id: padding 填充值
    
    Returns:
        csr_matrix: 稀疏屏蔽矩阵 [num_users, num_items]
                    rating_matrix[u, i] = 1 表示用户 u 历史交互过物品 i
    """
    row, col, data = [], [], []
    
    for idx, r in df.iterrows():
        user_id = r['user_id']
        seq = r['seq']
        for item in seq:
            if item != padding_id:  # 跳过 padding
                row.append(user_id)
                col.append(item)
                data.append(1)
    
    return csr_matrix((np.array(data), (np.array(row), np.array(col))), 
                       shape=(num_users, num_items))
# ==================== [END NEW] ====================

#torch.autograd.set_detect_anomaly(True)

# 自定义序列数据集类，继承自 PyTorch 的 Dataset
class SeqDataset(Dataset):
    # ==================== [ORIGINAL CODE COMMENTED OUT] ====================
    # # 初始化函数，将 DataFrame 中的数据转换为 Tensor
    # def __init__(self, data):
    #     self.seq_data = [torch.tensor(seq, dtype=torch.long) for seq in data['seq']]  # 历史交互序列（物品ID列表）            (28478)
    #     self.len_seq_data = [torch.tensor(len_seq, dtype=torch.long) for len_seq in data['len_seq']]  # 每个序列的有效长度   (28478)
    #     self.next_data = [torch.tensor(next_val, dtype=torch.long) for next_val in data['next']]  # 下一个要预测的目标物品ID  (28478)
    # ==================== [END ORIGINAL] ====================
    
    # ==================== [NEW] 支持 user_id 列 ====================
    def __init__(self, data):
        """
        初始化函数，将 DataFrame 中的数据转换为 Tensor
        
        Args:
            data: DataFrame with columns ['seq', 'len_seq', 'next', 'user_id' (optional)]
        """
        self.seq_data = [torch.tensor(seq, dtype=torch.long) for seq in data['seq']]
        self.len_seq_data = [torch.tensor(len_seq, dtype=torch.long) for len_seq in data['len_seq']]
        self.next_data = [torch.tensor(next_val, dtype=torch.long) for next_val in data['next']]
        
        # [NEW] 可选读取 user_id 列（用于 ProAlign 的 L_align 损失）
        if 'user_id' in data.columns:
            self.user_id_data = [torch.tensor(uid, dtype=torch.long) for uid in data['user_id']]
            self.has_user_id = True
        else:
            self.user_id_data = None
            self.has_user_id = False
    # ==================== [END NEW] ====================

    # 返回数据集大小
    def __len__(self):
        return len(self.seq_data)

    # ==================== [ORIGINAL CODE COMMENTED OUT] ====================
    # def __getitem__(self, idx):
    #     # 根据索引返回单个样本，包含：历史序列、序列长度、目标物品
    #     return {'seq': self.seq_data[idx], 'len_seq': self.len_seq_data[idx], 'next': self.next_data[idx]}
    # ==================== [END ORIGINAL] ====================
    
    # ==================== [NEW] 支持返回 user_id ====================
    def __getitem__(self, idx):
        """
        根据索引返回单个样本
        
        Returns:
            dict: 包含 seq, len_seq, next, user_id (如果有)
        """
        sample = {
            'seq': self.seq_data[idx],
            'len_seq': self.len_seq_data[idx],
            'next': self.next_data[idx]
        }
        # [NEW] 如果有 user_id，添加到返回值
        if self.has_user_id:
            sample['user_id'] = self.user_id_data[idx]
        return sample
    # ==================== [END NEW] ====================


# ==================== [NEW] 支持 RASD 相似用户的数据集类 ====================
class SeqDatasetWithSimUser(SeqDataset):
    """
    支持 RASD (Retrieval Augmented Self-Distillation) 的序列数据集
    
    继承自 SeqDataset，额外加载相似用户数据用于自蒸馏
    """
    
    def __init__(self, data, sim_users, all_user_seqs, sim_user_num, seq_len, padding_id):
        """
        初始化支持 RASD 的数据集
        
        Args:
            data: DataFrame with columns ['seq', 'len_seq', 'next', 'user_id']
            sim_users: numpy array, shape (num_users, 100), 预计算的相似用户列表
            all_user_seqs: list of sequences, 所有用户的序列（用于获取相似用户的序列）
            sim_user_num: int, 每个用户使用的相似用户数量 K
            seq_len: int, 序列长度（用于 padding）
            padding_id: int, padding token ID
        """
        super().__init__(data)
        self.sim_users = sim_users
        self.all_user_seqs = all_user_seqs
        self.sim_user_num = sim_user_num
        self.seq_len = seq_len
        self.padding_id = padding_id
    
    def __getitem__(self, idx):
        """
        根据索引返回单个样本，包含相似用户的序列
        
        Returns:
            dict: 包含 seq, len_seq, next, user_id, sim_seqs
        """
        # 获取基础样本
        sample = super().__getitem__(idx)
        
        # 获取当前用户的 user_id
        if self.has_user_id:
            user_id = sample['user_id'].item()
        else:
            print('使用 DataFrame 行索引')
            user_id = idx  # 如果没有 user_id，使用 DataFrame 行索引
        
        # 获取相似用户的序列
        sim_user_ids = self.sim_users[user_id][:self.sim_user_num]  # Top-K 相似用户
        sim_seqs = []
        for sim_uid in sim_user_ids:
            sim_seq = self._get_user_seq(sim_uid)
            sim_seqs.append(sim_seq)
        
        # 转换为 tensor
        sample['sim_seqs'] = torch.tensor(np.array(sim_seqs), dtype=torch.long)  # [K, seq_len]
        
        return sample
    
    def _get_user_seq(self, user_id):
        """
        获取指定用户的序列
        
        注意：AlphaFuse 的 train_data['seq'] 已经是 padding 后的序列，
        直接返回即可，无需再次 padding。
        
        Args:
            user_id: 用户 ID（对应 train_data 的行索引）
        
        Returns:
            seq: numpy array, shape (seq_len,)，已经过左侧 padding
        """
        # [FIX] 直接返回已经 padding 过的序列
        # 原代码错误：又进行了一次 padding
        user_seq = self.all_user_seqs[user_id]
        
        # train_data['seq'] 已经是 list 格式，转换为 numpy array
        return np.array(user_seq, dtype=np.int32)
# ==================== [END NEW] ====================

# =============================================================================
# 辅助函数：将字符串转换为布尔值
# 用于命令行参数解析
# =============================================================================
def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

# =============================================================================
# 统计模型参数量
# 返回：总参数量 和 可训练参数量
# =============================================================================
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())  # 所有参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 需要梯度更新的参数
    return total_params, trainable_params

logging.getLogger().setLevel(logging.INFO)

# =============================================================================
# 设置随机种子，确保实验可复现
# 包括：PyTorch、CUDA、NumPy、Python random 的随机种子
# =============================================================================
def setup_seed(seed): 
     torch.manual_seed(seed)  # CPU 随机种子
     torch.cuda.manual_seed_all(seed)  # 所有 GPU 随机种子
     np.random.seed(seed)  # NumPy 随机种子
     random.seed(seed)  # Python 内置 random 随机种子
     torch.backends.cudnn.deterministic = True  # 确保 cuDNN 使用确定性算法

# ==================== [NEW] 时间戳生成函数（与 BSARec 一致）====================
def get_local_time():
    """获取当前时间戳，格式：Dec-06-2024_14-30-25"""
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur

def set_logger(log_path):
    """设置日志记录器，同时输出到文件和控制台"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除已有的 handlers（避免重复添加）
    if logger.handlers:
        logger.handlers.clear()
    
    # 文件 Handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
    # 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
# ==================== [END NEW] ====================

# =============================================================================
# 命令行参数解析函数
# 定义所有训练相关的超参数
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")
    parser.add_argument('--random_seed', type=int, default=22)  # 随机种子，用于复现实验
    
    # ==================== [NEW] 日志和模型保存名称（与 BSARec 一致）====================
    parser.add_argument('--train_name', default=None, type=str,
                        help='实验名称，用于 log 和 pt 文件命名。默认使用时间戳')
    parser.add_argument('--output_dir', default='./output/', type=str,
                        help='日志输出目录')
    # ==================== [END NEW] ====================
    
    ### ==================== 训练相关设置 ====================
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')  # 学习率
    parser.add_argument('--lr_delay_rate', type=float, default=0.99)  # 学习率衰减系数
    parser.add_argument('--lr_delay_epoch', type=int, default=100)  # 学习率衰减的epoch间隔
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of max epochs.')  # 最大训练轮数
    # parser.add_argument('--data', nargs='?', default='ATV',
    #                     help='yc, ks, zhihu')  # 数据集名称：ATV=Amazon Movies&TV, ATG=Toys&Games, ASO=Sports&Outdoors
    parser.add_argument('--data', nargs='?', default='ASO',
                        help='yc, ks, zhihu')  # 数据集名称：ATV=Amazon Movies&TV, ATG=Toys&Games, ASO=Sports&Outdoors
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')  # 使用的GPU设备编号
    parser.add_argument('--l2_decay', type=float, default=1e-6,
                        help='l2 loss reg coef.')  # L2正则化系数（权重衰减）
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')  # 批量大小
    
    ### ==================== SASRec 骨干网络设置 ====================
    parser.add_argument('--num_blocks', default=2, type=int)  # Transformer Block 数量
    parser.add_argument('--num_heads', default=1, type=int)  # 多头注意力的头数
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')  # Dropout 概率
    
    ### ==================== [NEW] GRU4Rec 骨干网络设置 ====================
    parser.add_argument('--gru_hidden_size', type=int, default=128,
                        help='GRU hidden state dimension')  # GRU 隐状态维度
    parser.add_argument('--num_gru_layers', type=int, default=2,
                        help='Number of GRU layers')  # GRU 层数
    ### ==================== [END NEW] ====================
    
    ### ==================== 损失函数相关参数 ====================
    parser.add_argument('--loss_type', type=str, default="infoNCE")  # 损失类型：CE/BCE/infoNCE
    # parser.add_argument('--neg_ratio', type=int, default=1,
    #                     help='#Negative:#Positive = neg_ratio.')  # 负采样比例（负样本:正样本）
    parser.add_argument('--neg_ratio', type=int, default=64,
                        help='#Negative:#Positive = neg_ratio.')  # 负采样比例（负样本:正样本）
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='tao.')  # InfoNCE 损失的温度参数
    parser.add_argument('--beta', type=float, default=0.1,
                        help='scale of additional loss of RLMRec and LLMESR')  # 额外损失项的权重（用于RLMRec和LLMESR）
    
    ### ==================== [NEW 2024-12-17] BERT4Rec MLM 训练参数 ====================
    parser.add_argument('--use_mlm', type=str2bool, default=False,
                        help='BERT4Rec: 是否使用 MLM Loss 训练（论文原始方式），默认 False 使用 InfoNCE')
    ### ==================== [END NEW] ====================
    
    ### ==================== [NEW] RASD 相关参数（LLM-ESR 原始方法）====================
    parser.add_argument('--use_rasd', type=str2bool, default=True,
                        help='LLMESR: 是否启用 RASD (Retrieval Augmented Self-Distillation)')
    parser.add_argument('--alpha_rasd', type=float, default=0.1,
                        help='LLMESR: RASD 对齐损失权重')
    parser.add_argument('--sim_user_num', type=int, default=10,
                        help='LLMESR: 每个用户使用的相似用户数量 K')
    parser.add_argument('--user_sim_func', type=str, default='kd',
                        choices=['cl', 'kd'],
                        help='LLMESR: 用户相似度函数类型 (cl=对比学习, kd=知识蒸馏/MSE), 原版 LLM-ESR 默认 kd')
    parser.add_argument('--mask_ratio', type=float, default=0.6,
                        help='BERT4Rec: 随机 Mask 的比例 (原版 LLM-ESR 默认 0.6)')
    ### ==================== [END NEW] ====================
    
    ### ==================== 语言嵌入（Language Embeddings）设置 ====================
    # parser.add_argument('--language_model_type', default="3small", type=str)  # 语言模型类型：3small=text-embedding-3-small, 3large=text-embedding-3-large
    parser.add_argument('--language_model_type', default="3large", type=str)
    parser.add_argument('--language_embs_scale', default=40, type=int)  # 语言嵌入的缩放因子

    ### ==================== ID 嵌入设置 ====================
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Number of hidden factors, i.e., ID embedding size.')  # 隐藏层维度（ID嵌入维度）= d_s + d_n
    parser.add_argument('--ID_embs_init_type', type=str, default="normal")  # ID嵌入初始化方式：normal/zeros/uniform/ortho/xavier
    
    ### ==================== [NEW 2024-12-15] RLMRec 用户侧 LLM 信息参数 ====================
    # ┌──────────────────────────────────────────────────────────────────────────────────┐
    # │  RLMRec 用户侧对齐 vs LLMESR RASD - 两种用户侧 LLM 信息使用方式                    │
    # ├──────────────────────────────────────────────────────────────────────────────────┤
    # │                                                                                  │
    # │  RLMRec (--use_user_llm)              LLMESR (--use_rasd)                        │
    # │  ────────────────────                 ─────────────────                          │
    # │  使用: usr_intent_emb.pkl            使用: sim_user_100.pkl                      │
    # │  形状: (N_users, 3072)               形状: (N_users, 100)                        │
    # │  方式: 直接语义对齐                   方式: 相似用户蒸馏                           │
    # │                                                                                  │
    # │  usr_intent_emb[user_id]             sim_users[user_id]                          │
    # │        ↓                                   ↓                                     │
    # │  MLP 映射 → hidden_dim               [相似用户 ID 列表]                           │
    # │        ↓                                   ↓                                     │
    # │  InfoNCE / Cosine 对齐               forward(sim_seqs) → 蒸馏                    │
    # │                                                                                  │
    # │  损失: L_user = InfoNCE(h_u, MLP(z_u))                                           │
    # │  损失: L_rasd = CL(h_u, mean(h_sim))                                             │
    # │                                                                                  │
    # └──────────────────────────────────────────────────────────────────────────────────┘
    parser.add_argument('--use_user_llm', type=str2bool, default=False,
                        help='RLMRec: 是否使用用户侧 LLM 信息 (usr_intent_emb.pkl)')
    parser.add_argument('--alpha_user', type=float, default=1.0,
                        help='RLMRec: 用户侧对齐损失权重')
    parser.add_argument('--user_align_mode', type=str, default='infonce',
                        choices=['infonce', 'cosine'],
                        help='RLMRec: 用户侧对齐模式 (infonce=InfoNCE对比学习, cosine=余弦相似度)')
    parser.add_argument('--user_align_temp', type=float, default=1.0,
                        help='RLMRec: 用户侧 InfoNCE 温度参数 (仅 user_align_mode=infonce 时有效)')
    ### ==================== [END NEW] ====================
    
    ### ==================== 模型选择 ====================
    # parser.add_argument('--model_type', type=str, default="UniSRec")  # 模型类型：SASRec/MoRec/WhitenRec/UniSRec/LLMInit/RLMRec/IRLLRec/LLMESR/AlphaFuse
    parser.add_argument('--model_type', type=str, default="AlphaFuse_BERT4Rec") # AlphaFuse AlphaFuse_GRU AlphaFuse_BERT4Rec   LLMESR  LLMESR_GRU  LLMESR_BERT4Rec  RLMRec  RLMRec_GRU  IRLLRec  IRLLRec_GRU
    parser.add_argument('--SR_aligement_type', type=str, default="con")  # RLMRec的对齐方式：con=对比/gen=生成
    
    ### ==================== AlphaFuse 特有参数 ====================
    #parser.add_argument('--emb_dim', type=int, default=None,)
    parser.add_argument('--null_thres', type=float, default=None,)  # 零空间阈值：特征值小于此值的维度被视为零空间
    parser.add_argument('--null_dim', type=int, default=64,)  # 零空间维度 d_n：ID嵌入的维度
    parser.add_argument('--item_frequency_flag', type=str2bool, default=False)  # 是否使用物品频率加权
    parser.add_argument('--standardization', type=str2bool, default=True)  # 是否进行白化标准化
    parser.add_argument('--cover', type=str2bool, default=False)  # 是否覆盖零空间（True=替换，False=相加）
    parser.add_argument('--ID_space', type=str, default="singular")  # ID嵌入空间类型：singular=奇异值空间/euclidean=欧氏空间
    parser.add_argument('--inject_space', type=str, default="singular")  # 注入空间类型
    #parser.add_argument('--emb_init_type', type=str, default="normal")
    #parser.add_argument('--emb_sim_type', type=str, default="both")
    
    ### ==================== [NEW] ProAlign 特有参数 ====================
    parser.add_argument('--efficient_inference', action='store_true',
                        help='ProAlign: 启用推理效率优化（预计算物品原型表示）')
    parser.add_argument('--num_prototypes', type=int, default=64,
                        help='ProAlign: 原型数量 K')
    parser.add_argument('--proto_temperature', type=float, default=0.1,
                        help='ProAlign: 原型 softmax 温度') # 温度越低 → 原型分布越尖锐→ 语义信号越清晰     高温: 这个物品 40% 护肤 + 35% 彩妆 + 25% 香水  (模糊)    低温 (τ=0.05)：物品"专注"于最相似的那个原型 → 语义清晰
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='ProAlign: 对齐损失权重')
    parser.add_argument('--beta_proto', type=float, default=0.01,
                        help='ProAlign: 聚类损失权重')
    parser.add_argument('--llm_dim', type=int, default=3072,
                        help='ProAlign: LLM 意图维度')
    parser.add_argument('--fusion_mode', type=str, default='concat',
                        help='ProAlign: 融合模式 (add/concat)')
    parser.add_argument('--semantic_weight', type=float, default=0.5,
                        help='ProAlign: 加法融合时的语义权重')
    parser.add_argument('--semantic_init', type=str2bool, default=True,
                        help='ProAlign: 是否使用语义初始化 ID Embedding')
    # [NEW] 对齐损失模式和对比学习温度
    parser.add_argument('--align_mode', type=str, default='infonce',
                        help='ProAlign: 对齐损失模式 (kl/infonce)')
    parser.add_argument('--cl_temperature', type=float, default=1.0,
                        help='ProAlign: InfoNCE 对比学习温度')
    # [NEW] 多头原型参数（解决语义中和问题）
    parser.add_argument('--num_heads_proto', type=int, default=1,
                        help='ProAlign: 原型注意力头数 (1=单头向后兼容, 4/8=多头解耦)')
    # ==================== [NEW-SLSI] 序列级语义注入参数 ====================
    parser.add_argument('--use_slsi', type=str2bool, default=False,
                        help='ProAlign: 是否启用序列级语义注入 (SLSI)')
    parser.add_argument('--slsi_weight', type=float, default=0.3,
                        help='ProAlign: SLSI 语义注入权重')
    # ==================== [NEW-SLSI-ContextAware] 上下文感知 SLSI ====================
    parser.add_argument('--slsi_context_aware', type=str2bool, default=False,
                        help='ProAlign: 是否启用上下文感知 SLSI (True=结合历史位置, False=独立位置)')
    # ==================== [END NEW-SLSI-ContextAware] ====================
    # ==================== [END NEW-SLSI] ====================
    # ==================== [NEW-BIALIGN] 双向对齐参数 ====================
    # parser.add_argument('--use_bidirectional_align', type=str2bool, default=False,
    #                     help='ProAlign: 是否启用双向对齐 (用户+物品)')
    # parser.add_argument('--item_align_weight', type=float, default=0.1,
    #                     help='ProAlign: 物品对齐损失权重')
    # ==================== [END NEW-BIALIGN] ====================
    # ==================== [NEW] 原型冻结控制参数 ====================
    parser.add_argument('--freeze_prototypes', type=str2bool, default=True,
                        help='ProAlign: 是否冻结原型矩阵 (True=冻结, False=可训练)')
    # ==================== [NEW 2024-12-25] 语义困难负样本 ====================
    parser.add_argument('--hard_neg_top_k', type=int, default=10,
                        help='ProAlign: 每个物品的 Top-K 语义困难负样本数量 (0=禁用)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='ProAlign: 前 N 个 epoch 不使用困难负样本 (课程学习)')
    parser.add_argument('--use_attn_fusion', type=str2bool, default=True,
                        help='ProAlign: 是否使用 MultiheadAttention 动态原型融合')
    # ==================== [NEW 2024-12-27] Masked Prototype Modeling (MPM) ====================
    # parser.add_argument('--use_mpm', type=str2bool, default=True,
    #                     help='ProAlign: 是否启用 Masked Prototype Modeling (h_mask 同时预测 ID + 原型)')
    # parser.add_argument('--mpm_weight', type=float, default=0.1,
    #                     help='ProAlign: MPM 损失权重')
    # ==================== [END NEW] ====================
    # ==================== [NEW] PPD (Progressive Prototype Distillation) 参数 ====================
    parser.add_argument('--use_ppd', type=str2bool, default=False,
                        help='ProAlign: 是否启用渐进式原型蒸馏')
    parser.add_argument('--ppd_warmup_ratio', type=float, default=0.3,
                        help='ProAlign: PPD Phase 1 (冻结) 占比')
    parser.add_argument('--ppd_transition_ratio', type=float, default=0.7,
                        help='ProAlign: PPD Phase 2 结束点占比')
    parser.add_argument('--ppd_ema_decay', type=float, default=0.99,
                        help='ProAlign: PPD Phase 3 EMA 衰减系数')
    # ==================== [NEW 2024-12-23] ID Embedding 冻结参数 ====================
    parser.add_argument('--freeze_embedding_epochs', type=int, default=0,
                        help='ProAlign: 前 N 个 epoch 冻结 ID Embedding，保护 LLM 语义初始化 (0=不冻结)')
    # ==================== [END NEW] ====================
    # ==================== [END NEW] PPD 参数 ====================
    # # [NEW] Forward Prediction 参数 (已注释，恢复原始状态)
    # parser.add_argument('--use_forward_predictor', type=str2bool, default=True,
    #                     help='ProAlign: 是否使用前向预测器 (True=Forward Prediction, False=Direct Alignment)')
    # ==================== [NEW 2024-12-15] Item-only 消融实验参数 ====================
    # 控制是否使用用户侧 LLM 信息（usr_intent_emb.pkl）
    # True  (默认) : ProAlign (Full) - 使用 Item + User 侧 LLM 信息
    # False        : ProAlign (Item-only) - 仅使用 Item 侧 LLM 信息（消融实验）
    parser.add_argument('--use_user_intent', type=str2bool, default=True,
                        help='ProAlign: 是否使用用户侧 LLM 意图 (True=Full, False=Item-only 消融)')
    # ==================== [END NEW] Item-only 消融实验参数 ====================
    
    # ==================== [NEW 2025-01-17] Prototype 消融实验参数 ====================
    # 控制是否使用原型机制
    # False (默认) : ProAlign (Full) - 正常使用原型机制
    # True         : ProAlign (w/o Prototype) - 禁用原型机制（消融实验）
    parser.add_argument('--no_prototype', type=str2bool, default=False,
                        help='ProAlign: 是否禁用原型机制 (True=w/o Prototype 消融, False=Full)')
    # ==================== [END NEW 2025-01-17] ====================
    ### ==================== [END NEW] ====================
    
    # ==================== [NEW 2024-12-15] IRLLRec 参数 ====================
    # IRLLRec: Intent Representation Learning with LLM for Recommendation (SIGIR 2025)
    # 核心参数：意图原型数量、多层次蒸馏权重和温度
    parser.add_argument('--intent_num', type=int, default=64,
                        help='IRLLRec: 意图原型数量 K')
    parser.add_argument('--kd_weight', type=float, default=0.01,
                        help='IRLLRec: Profile 蒸馏损失权重 (L_kd)')
    parser.add_argument('--kd_temperature', type=float, default=0.2,
                        help='IRLLRec: Profile 蒸馏 InfoNCE 温度')
    parser.add_argument('--kd_int_weight', type=float, default=0.02,
                        help='IRLLRec: Intent 蒸馏损失权重 (L_kd_int) ⭐核心')
    parser.add_argument('--kd_int_temperature', type=float, default=0.2,
                        help='IRLLRec: Intent 蒸馏 InfoNCE 温度')
    parser.add_argument('--kd_int_weight_2', type=float, default=1e-7,
                        help='IRLLRec: 加噪对比损失权重 (L_kd_int_2)')
    parser.add_argument('--kd_int_weight_3', type=float, default=1e-7,
                        help='IRLLRec: 动量蒸馏损失权重 (L_ITM)')
    parser.add_argument('--irllrec_momentum', type=float, default=0.999,
                        help='IRLLRec: 动量编码器 EMA 系数')
    # ==================== [END NEW] IRLLRec 参数 ====================
    
    return parser.parse_args()


# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == '__main__':

    # ---------------------- 初始化配置 ----------------------
    args = parse_args()  # 解析命令行参数
    setup_seed(args.random_seed)  # 设置随机种子确保可复现
    
    # ==================== [NEW] 设置 train_name 和日志（与 BSARec 一致）====================
    # 如果未指定 train_name，则使用时间戳
    if args.train_name is None:
        args.train_name = get_local_time()
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir): # './output/'
        os.makedirs(args.output_dir)
    
    # 设置日志文件路径
    log_path = os.path.join(args.output_dir, args.train_name + '.log') # './output/Dec-06-2025_20-14-50.log'
    logger = set_logger(log_path)
    logger.info(f"Experiment: {args.train_name}") # Experiment: Dec-06-2025_20-14-50
    # ==================== [END NEW] ====================
    
    # 配置 CUDA 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)  # 指定使用的GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置计算设备
    
    # 将参数转换为字典形式，方便传递给模型
    key_words = vars(args)
    
    # [MODIFIED 2024-01-14] 支持自定义数据路径（用于 AB 实验）
    # 如果 args.data 包含 '/'，视为完整路径；否则使用默认的 './data/' 前缀
    # 示例：
    #   --data Beauty              → ./data/Beauty
    #   --data ./AB/Beauty/Prospective  → ./AB/Beauty/Prospective
    #   --data AB/Beauty/Retrospective  → ./AB/Beauty/Retrospective
    if '/' in args.data or os.path.isabs(args.data):
        data_directory = args.data if args.data.startswith('./') or os.path.isabs(args.data) else './' + args.data
    else:
        data_directory = './data/' + args.data  # 数据目录路径 './data/ASO'   './data/Beauty'
    
    # model_directory 也需要适配新路径
    # 从 data_directory 提取数据集名称用于保存模型
    data_name = os.path.basename(data_directory.rstrip('/'))  # 'Beauty' 或 'Prospective'
    
    # model_directory = './saved/' +args.data  # 模型保存目录（原代码，已注释）
    # ==================== [NEW] 模型保存路径（与 BSARec 一致）====================
    model_directory = './saved/' + data_name + '/'  # 模型保存目录  './saved/ASO/'   './saved/Prospective/'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    checkpoint_path = os.path.join(model_directory, args.train_name + '.pt')  # 使用 train_name 命名  './saved/ASO/Dec-06-2025_20-14-50.pt'       './saved/Beauty/Dec-11-2025_18-40-12.pt'
    # ==================== [END NEW] ====================
    
    key_words["language_embs_path"] = data_directory  # 添加语言嵌入路径到参数字典  './data/ASO'     './data/Beauty'

    # ---------------------- 根据 model_type 初始化对应的模型 ----------------------
    # [NEW] PPD 调度器初始化为 None（防止 NameError）
    ppd_scheduler = None
    
    # SASRec: 纯ID嵌入的自注意力序列推荐模型（基线）
    if args.model_type == "SASRec":
        model = SASRec(device, **key_words).to(device)
    # MoRec: 使用语言嵌入 + MLP适配器
    elif args.model_type == "MoRec":
        model = MoRec(device, **key_words).to(device)
    # WhitenRec: 使用白化后的语言嵌入 + MLP适配器
    elif args.model_type == "WhitenRec":
        model = WhitenRec(device, **key_words).to(device)
    # UniSRec: 使用语言嵌入 + MoE（专家混合）适配器
    elif args.model_type == "UniSRec":
        model = UniSRec(device, **key_words).to(device)
    # LLMInit: 使用语言嵌入初始化ID嵌入（可微调）
    elif args.model_type == "LLMInit":
        model = LLMInit(device, **key_words).to(device)
    # RLMRec: ID嵌入 + 语义重建损失对齐
    elif args.model_type == "RLMRec":
        model = RLMRec(device, **key_words).to(device)
        # ==================== [NEW 2024-12-15] 加载用户侧 LLM 嵌入 ====================
        if args.use_user_llm:
            user_intent_path = os.path.join(data_directory, 'usr_intent_emb.pkl')
            model.load_user_intent_embedding(user_intent_path)
            logger.info(f"[RLMRec] User LLM mode: {args.user_align_mode}, weight: {args.alpha_user}, temp: {args.user_align_temp}")
        # ==================== [END NEW] ====================
    # LLMESR: 双视图模型，同时使用ID嵌入和语言嵌入
    elif args.model_type == "LLMESR":
        model = LLMESR(device, **key_words).to(device)
    # AlphaFuse: 本文提出的方法，在语言嵌入的零空间中注入ID信息
    elif args.model_type == "AlphaFuse":
        # **key_words 是 Python 的字典解包语法，*args：解包列表/元组 → 位置参数，**kwargs：解包字典 → 关键字参数
        model = AlphaFuse(device, **key_words).to(device) # 调用 AlphaFuse 的 init()，先调用 SASRec_backbone 的 init()，然后再初始化 AlphaFuse 中的属性
    
    # ==================== [NEW] ProAlign: 原型对齐的序列推荐模型 ====================
    elif args.model_type == "ProAlign":
        # 始终使用标准 ProAlign 进行训练，--efficient_inference 仅在评估时生效
        model = ProAlign(device, **key_words).to(device)
        if args.efficient_inference:
            print("[ProAlign] Efficient inference mode enabled (will apply during evaluation)")
        # 加载意图 embedding（如果文件存在）
        user_intent_path = os.path.join(data_directory, 'usr_intent_emb.pkl') # './data/Beauty/usr_intent_emb.pkl'
        item_intent_path = os.path.join(data_directory, 'itm_intent_emb.pkl') # './data/Beauty/itm_intent_emb.pkl'
        model.load_intent_embeddings(user_intent_path, item_intent_path)
        
        # ==================== [NEW 2024-12-15] Item-only 消融实验控制 ====================
        # 根据 --use_user_intent 参数决定是否使用用户侧 LLM 信息
        # False = Item-only 模式：清空 user_intent_emb，禁用用户侧对齐损失 L_align
        if not args.use_user_intent:
            model.user_intent_emb = None
            logger.info("[ProAlign] ⚠️ Item-only mode: User intent embedding DISABLED (ablation study)")
            logger.info("[ProAlign] → L_align will be 0, only using Item-side LLM info")
        else:
            logger.info("[ProAlign] Full mode: Using both Item + User side LLM info")
        # ==================== [END NEW] Item-only 消融实验控制 ====================
        
        # [FIXED] 正确的初始化顺序：
        # Step 1: 语义初始化 ID Embedding (必须在原型初始化之前！)
        # 因为 initialize_prototypes 会释放 item_intent_emb 以节省内存
        if args.semantic_init:
            model.initialize_item_embeddings()
            logger.info("[ProAlign] ID Embeddings initialized from LLM semantics!")
        
        # Step 2: 预计算语义困难负样本 (必须在 initialize_prototypes 之前！)
        # 因为 initialize_prototypes 会释放 item_intent_emb
        if hasattr(model, 'precompute_hard_negatives'):
            hard_neg_top_k = getattr(args, 'hard_neg_top_k', 10)
            model.precompute_hard_negatives(top_k=hard_neg_top_k)
            logger.info(f"[ProAlign] Semantic hard negatives precomputed (Top-{hard_neg_top_k})")
        
        # Step 3: 初始化原型矩阵 P（会释放 item_intent_emb）
        model.initialize_prototypes()
        
        # [NEW] 将 prototypes 移到正确的设备上（修复 CPU/CUDA 设备不匹配问题）
        model.prototypes.data = model.prototypes.data.to(device) # (64,128)
        logger.info(f"[ProAlign] Prototypes moved to {device}")
        
        # ==================== [NEW] PPD 调度器初始化 ====================
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
    # ==================== [END NEW] ====================
    
    # ==================== [NEW] GRU4Rec 系列模型 ====================
    # GRU4Rec: 纯 ID 嵌入的 GRU 序列推荐模型（基线）
    elif args.model_type == "GRU4Rec":
        model = GRU4Rec(device, **key_words).to(device)
    # MoRec_GRU: 语言嵌入 + MLP 适配器 + GRU 编码器
    elif args.model_type == "MoRec_GRU":
        model = MoRec_GRU(device, **key_words).to(device)
    # UniSRec_GRU: 语言嵌入 + MoE 适配器 + GRU 编码器
    elif args.model_type == "UniSRec_GRU":
        model = UniSRec_GRU(device, **key_words).to(device)
    # LLMInit_GRU: 语义初始化 ID 嵌入 + GRU 编码器
    elif args.model_type == "LLMInit_GRU":
        model = LLMInit_GRU(device, **key_words).to(device)
    # AlphaFuse_GRU: 零空间融合 + GRU 编码器（AlphaFuse 的 GRU 版本）
    elif args.model_type == "AlphaFuse_GRU":
        model = AlphaFuse_GRU(device, **key_words).to(device)
    # WhitenRec_GRU: 白化语言嵌入 + MLP 适配器 + GRU 编码器
    elif args.model_type == "WhitenRec_GRU":
        model = WhitenRec_GRU(device, **key_words).to(device)
    # RLMRec_GRU: 语义重建损失 + GRU 编码器
    elif args.model_type == "RLMRec_GRU":
        model = RLMRec_GRU(device, **key_words).to(device)
        # ==================== [NEW 2024-12-15] 加载用户侧 LLM 嵌入 ====================
        if args.use_user_llm:
            user_intent_path = os.path.join(data_directory, 'usr_intent_emb.pkl')
            model.load_user_intent_embedding(user_intent_path)
            logger.info(f"[RLMRec_GRU] User LLM mode: {args.user_align_mode}, weight: {args.alpha_user}, temp: {args.user_align_temp}")
        # ==================== [END NEW] ====================
    # LLMESR_GRU: 双视图模型 + GRU 编码器
    elif args.model_type == "LLMESR_GRU":
        model = LLMESR_GRU(device, **key_words).to(device)
    # ProAlign_GRU: 原型对齐的序列推荐模型 + GRU 编码器
    elif args.model_type == "ProAlign_GRU":
        model = ProAlign_GRU(device, **key_words).to(device)
        # 加载意图 embedding（与 ProAlign 相同的初始化流程）
        user_intent_path = os.path.join(data_directory, 'usr_intent_emb.pkl')
        item_intent_path = os.path.join(data_directory, 'itm_intent_emb.pkl')
        model.load_intent_embeddings(user_intent_path, item_intent_path)
        
        # ==================== [NEW 2024-12-15] Item-only 消融实验控制 ====================
        if not args.use_user_intent:
            model.user_intent_emb = None
            logger.info("[ProAlign_GRU] ⚠️ Item-only mode: User intent embedding DISABLED (ablation study)")
            logger.info("[ProAlign_GRU] → L_align will be 0, only using Item-side LLM info")
        else:
            logger.info("[ProAlign_GRU] Full mode: Using both Item + User side LLM info")
        # ==================== [END NEW] Item-only 消融实验控制 ====================
        
        # 语义初始化 ID Embedding
        if args.semantic_init:
            model.initialize_item_embeddings()
            logger.info("[ProAlign_GRU] ID Embeddings initialized from LLM semantics!")
        
        # 预计算语义困难负样本
        if hasattr(model, 'precompute_hard_negatives'):
            hard_neg_top_k = getattr(args, 'hard_neg_top_k', 10)
            model.precompute_hard_negatives(top_k=hard_neg_top_k)
            logger.info(f"[ProAlign_GRU] Semantic hard negatives precomputed (Top-{hard_neg_top_k})")
        
        # 初始化原型矩阵
        model.initialize_prototypes()
        model.prototypes.data = model.prototypes.data.to(device)
        logger.info(f"[ProAlign_GRU] Prototypes moved to {device}")
        
        # ==================== [FIX 2024-12-21] PPD 调度器初始化 ====================
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
            logger.info(f"[PPD] Scheduler initialized for ProAlign_GRU")
        # ==================== [END FIX] ====================
    # ==================== [END NEW] GRU4Rec 系列模型 ====================
    
    # ==================== [NEW 2024-12-15] IRLLRec 模型初始化 ====================
    # IRLLRec: Intent Representation Learning (SIGIR 2025)
    elif args.model_type == "IRLLRec":
        # 添加 IRLLRec 特有参数到 key_words
        key_words['intent_num'] = args.intent_num
        key_words['kd_weight'] = args.kd_weight
        key_words['kd_temperature'] = args.kd_temperature
        key_words['kd_int_weight'] = args.kd_int_weight
        key_words['kd_int_temperature'] = args.kd_int_temperature
        key_words['kd_int_weight_2'] = args.kd_int_weight_2
        key_words['kd_int_weight_3'] = args.kd_int_weight_3
        key_words['momentum'] = args.irllrec_momentum
        
        model = IRLLRec(device, **key_words).to(device)
        
        # 加载 LLM 嵌入文件
        # [COMMENTED] Profile 嵌入（粗粒度）- AlphaFuse 中没有这两个文件
        # usrprf_path = os.path.join(data_directory, 'usr_emb_np.pkl')
        # itmprf_path = os.path.join(data_directory, 'itm_emb_np.pkl')
        usrprf_path = None  # AlphaFuse 中不使用 Profile 嵌入
        itmprf_path = None  # AlphaFuse 中不使用 Profile 嵌入
        # Intent 嵌入（细粒度）- 优先使用 3072 维的文件，否则使用通用意图文件
        usrint_path = os.path.join(data_directory, 'user_intent_emb_3.pkl')
        if not os.path.exists(usrint_path):
            usrint_path = os.path.join(data_directory, 'usr_intent_emb.pkl')
        itmint_path = os.path.join(data_directory, 'item_intent_emb_3.pkl')
        if not os.path.exists(itmint_path):
            itmint_path = os.path.join(data_directory, 'itm_intent_emb.pkl')
        
        model.load_embeddings(usrprf_path, itmprf_path, usrint_path, itmint_path)
        logger.info(f"[IRLLRec] Initialized with intent_num={args.intent_num}")
        logger.info(f"[IRLLRec] kd_weight={args.kd_weight}, kd_int_weight={args.kd_int_weight}")
        logger.info(f"[IRLLRec] momentum={args.irllrec_momentum}")
    
    # IRLLRec_GRU: IRLLRec + GRU 编码器
    elif args.model_type == "IRLLRec_GRU":
        # 添加 IRLLRec 特有参数到 key_words
        key_words['intent_num'] = args.intent_num
        key_words['kd_weight'] = args.kd_weight
        key_words['kd_temperature'] = args.kd_temperature
        key_words['kd_int_weight'] = args.kd_int_weight
        key_words['kd_int_temperature'] = args.kd_int_temperature
        key_words['kd_int_weight_2'] = args.kd_int_weight_2
        key_words['kd_int_weight_3'] = args.kd_int_weight_3
        key_words['momentum'] = args.irllrec_momentum
        
        model = IRLLRec_GRU(device, **key_words).to(device)
        
        # 加载 LLM 嵌入文件
        # [COMMENTED] Profile 嵌入（粗粒度）- AlphaFuse 中没有这两个文件
        # usrprf_path = os.path.join(data_directory, 'usr_emb_np.pkl')
        # itmprf_path = os.path.join(data_directory, 'itm_emb_np.pkl')
        usrprf_path = None  # AlphaFuse 中不使用 Profile 嵌入
        itmprf_path = None  # AlphaFuse 中不使用 Profile 嵌入
        usrint_path = os.path.join(data_directory, 'user_intent_emb_3.pkl')
        if not os.path.exists(usrint_path):
            usrint_path = os.path.join(data_directory, 'usr_intent_emb.pkl')
        itmint_path = os.path.join(data_directory, 'item_intent_emb_3.pkl')
        if not os.path.exists(itmint_path):
            itmint_path = os.path.join(data_directory, 'itm_intent_emb.pkl')
        
        model.load_embeddings(usrprf_path, itmprf_path, usrint_path, itmint_path)
        logger.info(f"[IRLLRec_GRU] Initialized with intent_num={args.intent_num}")
        logger.info(f"[IRLLRec_GRU] kd_weight={args.kd_weight}, kd_int_weight={args.kd_int_weight}")
        logger.info(f"[IRLLRec_GRU] momentum={args.irllrec_momentum}")
    # ==================== [END NEW] IRLLRec 模型初始化 ====================
    
    # ==================== [NEW 2024-12-15] BERT4Rec 系列模型初始化 ====================
    # BERT4Rec: 纯 ID 嵌入的双向注意力序列推荐模型
    # 与 SASRec 的区别：使用双向注意力（可以看到左右两边）
    elif args.model_type == "BERT4Rec":
        model = BERT4Rec(device, **key_words).to(device)
        logger.info("[BERT4Rec] Initialized with bidirectional attention")
        logger.info(f"[BERT4Rec] Embedding size: item_num + 2 = {model.num_embeddings} (includes padding + [MASK])")
    
    # MoRec_BERT4Rec: 语言嵌入 + MLP 适配器 + 双向注意力
    elif args.model_type == "MoRec_BERT4Rec":
        model = MoRec_BERT4Rec(device, **key_words).to(device)
        logger.info("[MoRec_BERT4Rec] Initialized with language embeddings + MLP + bidirectional attention")
    
    # WhitenRec_BERT4Rec: 白化语言嵌入 + MLP 适配器 + 双向注意力
    elif args.model_type == "WhitenRec_BERT4Rec":
        model = WhitenRec_BERT4Rec(device, **key_words).to(device)
        logger.info("[WhitenRec_BERT4Rec] Initialized with whitened language embeddings + bidirectional attention")
    
    # UniSRec_BERT4Rec: 语言嵌入 + MoE 适配器 + 双向注意力
    elif args.model_type == "UniSRec_BERT4Rec":
        model = UniSRec_BERT4Rec(device, **key_words).to(device)
        logger.info("[UniSRec_BERT4Rec] Initialized with MoE adapter + bidirectional attention")
    
    # LLMInit_BERT4Rec: 语义初始化 ID 嵌入 + 双向注意力
    elif args.model_type == "LLMInit_BERT4Rec":
        model = LLMInit_BERT4Rec(device, **key_words).to(device)
        logger.info("[LLMInit_BERT4Rec] Initialized with semantic-initialized ID embeddings + bidirectional attention")
    
    # RLMRec_BERT4Rec: ID 嵌入 + 语义重建损失 + 双向注意力
    elif args.model_type == "RLMRec_BERT4Rec":
        model = RLMRec_BERT4Rec(device, **key_words).to(device)
        logger.info("[RLMRec_BERT4Rec] Initialized with semantic reconstruction + bidirectional attention")
        # ==================== [FIX 2024-12-16] 加载用户侧 LLM 嵌入 ====================
        if args.use_user_llm:
            user_intent_path = os.path.join(data_directory, 'usr_intent_emb.pkl')
            model.load_user_intent_embedding(user_intent_path)
            logger.info(f"[RLMRec_BERT4Rec] User LLM mode: {args.user_align_mode}, weight: {args.alpha_user}, temp: {args.user_align_temp}")
        # ==================== [END FIX] ====================
    
    # LLMESR_BERT4Rec: 双视图模型 + 双向注意力
    elif args.model_type == "LLMESR_BERT4Rec":
        model = LLMESR_BERT4Rec(device, **key_words).to(device)
        logger.info("[LLMESR_BERT4Rec] Initialized with dual-view + bidirectional attention")
    
    # AlphaFuse_BERT4Rec: 零空间融合 + 双向注意力（AlphaFuse 的 BERT4Rec 版本）
    elif args.model_type == "AlphaFuse_BERT4Rec":
        model = AlphaFuse_BERT4Rec(device, **key_words).to(device)
        logger.info("[AlphaFuse_BERT4Rec] Initialized with zero-space fusion + bidirectional attention")
        logger.info(f"[AlphaFuse_BERT4Rec] Embedding size: item_num + 2 = {model.num_embeddings}")
    
    # ProAlign_BERT4Rec: 原型对齐 + 双向注意力（用户的模型）
    elif args.model_type == "ProAlign_BERT4Rec":
        model = ProAlign_BERT4Rec(device, **key_words).to(device)
        # 加载意图 embedding
        user_intent_path = os.path.join(data_directory, 'usr_intent_emb.pkl')
        item_intent_path = os.path.join(data_directory, 'itm_intent_emb.pkl')
        model.load_intent_embeddings(user_intent_path, item_intent_path)
        
        # ==================== Item-only 消融实验控制 ====================
        if not args.use_user_intent:
            model.user_intent_emb = None
            logger.info("[ProAlign_BERT4Rec] ⚠️ Item-only mode: User intent embedding DISABLED")
        else:
            logger.info("[ProAlign_BERT4Rec] Full mode: Using both Item + User side LLM info")
        
        # ==================== 语义初始化 ID Embedding ====================
        if args.semantic_init:
            model.initialize_item_embeddings()
            logger.info("[ProAlign_BERT4Rec] ID Embeddings initialized from LLM semantics!")
        
        # 预计算语义困难负样本
        if hasattr(model, 'precompute_hard_negatives'):
            hard_neg_top_k = getattr(args, 'hard_neg_top_k', 10)
            model.precompute_hard_negatives(top_k=hard_neg_top_k)
            logger.info(f"[ProAlign_BERT4Rec] Semantic hard negatives precomputed (Top-{hard_neg_top_k})")
        
        # 初始化原型矩阵
        model.initialize_prototypes()
        model.prototypes.data = model.prototypes.data.to(device)
        logger.info("[ProAlign_BERT4Rec] Initialized with prototype alignment + bidirectional attention")
        logger.info(f"[ProAlign_BERT4Rec] Prototypes: {model.num_prototypes}, Fusion: {model.fusion_mode}")
        
        # ==================== [FIX 2024-12-21] PPD 调度器初始化 ====================
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
            logger.info(f"[PPD] Scheduler initialized for ProAlign_BERT4Rec")
        # ==================== [END FIX] ====================

    
    # IRLLRec_BERT4Rec: 意图表示学习 + 双向注意力
    elif args.model_type == "IRLLRec_BERT4Rec":
        # 添加 IRLLRec 特有参数到 key_words
        key_words['intent_num'] = args.intent_num
        key_words['kd_weight'] = args.kd_weight
        key_words['kd_temperature'] = args.kd_temperature
        key_words['kd_int_weight'] = args.kd_int_weight
        key_words['kd_int_temperature'] = args.kd_int_temperature
        key_words['kd_int_weight_2'] = args.kd_int_weight_2
        key_words['kd_int_weight_3'] = args.kd_int_weight_3
        key_words['momentum'] = args.irllrec_momentum
        
        model = IRLLRec_BERT4Rec(device, **key_words).to(device)
        
        # 加载 LLM 嵌入文件（直接使用 Beauty 数据集的文件名格式）
        usrprf_path = None
        itmprf_path = None
        usrint_path = os.path.join(data_directory, 'usr_intent_emb.pkl')
        itmint_path = os.path.join(data_directory, 'itm_intent_emb.pkl')
        
        model.load_embeddings(usrprf_path, itmprf_path, usrint_path, itmint_path)
        logger.info("[IRLLRec_BERT4Rec] Initialized with intent representation learning + bidirectional attention")
        logger.info(f"[IRLLRec_BERT4Rec] intent_num={args.intent_num}")
    # ==================== [END NEW] BERT4Rec 系列模型初始化 ====================

    #print(device)
    #print(next(model.item_embeddings.language_embeddings.parameters()).device)
    
    # ---------------------- 初始化优化器 ----------------------
    # 使用 Adam 优化器，设置学习率、epsilon防止除零、L2正则化（权重衰减）  注意：一般序列推荐中 weight_decay 默认为0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_delay_rate)
    
    # ---------------------- 打印模型参数统计 ----------------------
    total_params, trainable_params = count_parameters(model)

    # print(f"Total Parameters: {total_params}")  # 总参数量（原代码，已注释）
    # print(f"Trainable Parameters: {trainable_params}")  # 可训练参数量（原代码，已注释）
    # ==================== [NEW] 使用 logger 输出 ====================
    logger.info(f"Total Parameters: {total_params}")
    logger.info(f"Trainable Parameters: {trainable_params}")
    # ==================== [END NEW] ====================

    #for name, param in model.named_parameters():
    #    try:
    #        torch.nn.init.xavier_normal_(param.data)
    #    except:
    #        pass # just ignore those failed init layers
    
    # print(key_words)  # 打印所有超参数配置（原代码，已注释）
    # ==================== [NEW] 使用 logger 输出 ====================
    logger.info(str(args))  # 记录所有参数到日志
    # ==================== [END NEW] ====================

    #model.train() # enable model training

    # ---------------------- 加载训练数据 ----------------------
    # 列名	        含义	                            示例
    # seq	        用户交互序列（物品 ID 列表）	[5, 12, 8, 3, ...]
    # len_seq	    序列真实长度	                        25
    # next	        下一个要预测的物品（目标）	            42
    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))  # 读取训练集  (28478,3)，训练集中有28478条序列   seq  len_seq   next             Beauty：(22363,4) user_id seq  len_seq   next
    train_data.reset_index(inplace=True,drop=True)  # 重置索引     数据可能经过过滤/采样，导致索引不连续，drop=True：丢弃旧索引，不保存为新列
    
    # ==================== [NEW] RASD 支持：加载相似用户数据 ====================
    # [FIX 2024-12-16] 添加 BERT4Rec 版本到 RASD 支持列表
    if args.use_rasd and args.model_type in ["LLMESR", "LLMESR_GRU", "LLMESR_BERT4Rec",
                                              "ProAlign", "ProAlign_GRU", "ProAlign_BERT4Rec"]:
        # 加载相似用户文件
        sim_user_path = os.path.join(data_directory, 'sim_user_100.pkl')
        if os.path.exists(sim_user_path):
            sim_users = pickle.load(open(sim_user_path, 'rb'))
            logger.info(f"[RASD] Loaded sim_user_100.pkl: shape={sim_users.shape}")
            
            # 构建所有用户的序列列表（用于获取相似用户的序列）
            # 使用 train_data 中的序列，按 user_id 排序（假设已排序）
            all_user_seqs = [seq for seq in train_data['seq']]
            logger.info(f"[RASD] Built all_user_seqs: {len(all_user_seqs)} users")
            
            # [FIX] 从 train_data 和 data_statis.df 获取 seq_len 和 item_num
            # 而不是从 key_words（key_words 中没有这些键）
            seq_len = len(train_data['seq'].iloc[0])  # 从第一个序列获取长度
            data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
            item_num = data_statis['item_num'].iloc[0]  # 物品数量
            logger.info(f"[RASD] seq_len={seq_len}, item_num={item_num}")
            
            # 使用支持 RASD 的数据集类
            train_dataset = SeqDatasetWithSimUser(
                data=train_data,
                sim_users=sim_users,
                all_user_seqs=all_user_seqs,
                sim_user_num=args.sim_user_num,
                seq_len=seq_len,
                padding_id=item_num  # padding_id = item_num
            )
            logger.info(f"[RASD] Using SeqDatasetWithSimUser with K={args.sim_user_num}")
        else:
            logger.warning(f"[RASD] sim_user_100.pkl not found at {sim_user_path}, falling back to standard SeqDataset")
            train_dataset = SeqDataset(train_data)
    else:
        # 标准数据集（不使用 RASD）
        train_dataset = SeqDataset(train_data)  # 调用 SeqDataset 的 init()，转换为 PyTorch Dataset
    # ==================== [END NEW] ====================
    
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size)  # [原代码，已注释] 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # 应该加上 shuffle=True

    # ---------------------- 加载验证数据 ----------------------
    val_data = pd.read_pickle(os.path.join(data_directory, 'val_data.df'))  # 读取验证集    (3559,3)   Beauty：(22363,4) user_id seq  len_seq   next
    val_data.reset_index(inplace=True,drop=True)
    val_dataset = SeqDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # ---------------------- 加载测试数据 ----------------------
    test_data = pd.read_pickle(os.path.join(data_directory, 'test_data.df'))  # 读取测试集    Beauty：(22363,4) user_id seq  len_seq   next
    test_data.reset_index(inplace=True,drop=True)
    test_dataset = SeqDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # ==================== [NEW] 生成评测屏蔽矩阵（BSARec 风格）====================
    # 验证集：屏蔽 val_data['seq'] 中的非 padding 物品
    # 测试集：屏蔽 test_data['seq'] 中的非 padding 物品
    # 
    # 与 BSARec 逻辑等价：
    #   BSARec valid: 屏蔽 user_seq[:-2]  ←→  AlphaFuse val_data['seq'] = items[:-2]
    #   BSARec test:  屏蔽 user_seq[:-1]  ←→  AlphaFuse test_data['seq'] = items[:-1]
    #
    # [FIX] 矩阵列数 = item_num（不是 item_num+1）
    # 因为所有模型的 predict() 返回 [B, item_num]：
    #   - SASRec/GRU4Rec: item_embs[:-1] 去掉 padding
    #   - BERT4Rec: item_embs[:-2] 去掉 padding 和 [MASK]
    
    # [FIX 2026-01-06] 确保 item_num 和 padding_id 总是被定义
    # 注意：如果启用了 RASD，item_num 已经在 RASD 块中定义了，这里会重新读取（无影响）
    data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
    item_num = data_statis['item_num'].iloc[0]  # 物品数量
    padding_id = item_num  # AlphaFuse 约定：padding_id = item_num
    
    num_users = len(val_data)  # 用户数量
    num_items_for_matrix = item_num  # 物品数量（与 predict() 输出一致）
    
    valid_rating_matrix = generate_rating_matrix(val_data, num_users, num_items_for_matrix, padding_id)
    test_rating_matrix = generate_rating_matrix(test_data, num_users, num_items_for_matrix, padding_id)
    
    logger.info(f"[Rating Matrix] valid: {valid_rating_matrix.shape}, nnz={valid_rating_matrix.nnz}")
    logger.info(f"[Rating Matrix] test: {test_rating_matrix.shape}, nnz={test_rating_matrix.nnz}")
    # ==================== [END NEW] ====================

    # ---------------------- 早停机制（Early Stopping）相关变量 ----------------------

    best_ndcg20 = 0  # 初始化最佳验证集 NDCG@20 指标
    patience = 50  # 早停容忍次数：连续 patience 个 epoch 没有提升则停止训练
    counter = 0  # 早停计数器：记录连续未提升的 epoch 数

    step = 0  # 全局训练步数计数器
    T = 0.0   # 初始化累计时间
    # print("Loading Data Done.")  # 原代码，已注释
    # ==================== [NEW] 使用 logger ====================
    logger.info("Loading Data Done.")
    # ==================== [END NEW] ====================
    
    # ---------------------- 训练前先在验证集上评估初始性能 ----------------------
    t0 = time.time() # 返回当前的时间戳
    # 训练前的这次评估可以省略，因为第一个 epoch 后就会重新计算覆盖掉
    val_ndcg20 = evaluate(model, val_loader, device, rating_matrix=valid_rating_matrix)  # 评估模型在验证集上的表现，获取初始 NDCG@20 作为早停判断基准
    t1 = time.time() - t0
    # print("\n using ",t1, "s ", "Eval Time Cost",T,"s.")  # 原代码，已注释
    # ==================== [NEW] 使用 logger ====================
    logger.info(f"\n using {t1}s, Eval Time Cost {T}s.")
    # ==================== [END NEW] ====================
    # =============================================================================
    # 主训练循环
    # =============================================================================
    ''''''
    for epoch in range(args.epoch):
        model.train()  # 设置模型为训练模式（启用Dropout等）
        
        # ==================== [NEW] PPD 调度 ====================
        # 如果启用了 PPD，在每个 epoch 开始时更新原型状态
        # [FIX 2024-12-21] 支持三个 ProAlign 变体
        if args.model_type in ["ProAlign", "ProAlign_GRU", "ProAlign_BERT4Rec"] and args.use_ppd and ppd_scheduler is not None:
            ppd_scheduler.step(epoch)
        # ==================== [END NEW] ====================
        
        # ==================== [NEW 2024-12-25] 课程学习：更新当前 epoch ====================
        if args.model_type in ["ProAlign", "ProAlign_GRU", "ProAlign_BERT4Rec"]:
            if hasattr(model, 'current_epoch'):
                model.current_epoch = epoch
                if epoch == getattr(args, 'warmup_epochs', 5):
                    logger.info(f"[Curriculum] Epoch {epoch}: Hard negatives ACTIVATED! 🚀")
        # ==================== [END NEW] ====================
        
        # ==================== [NEW 2024-12-23] ID Embedding 冻结调度 ====================
        # 保护 LLM 语义初始化：前 N 个 epoch 冻结 ID Embedding
        if args.freeze_embedding_epochs > 0 and args.model_type in ["ProAlign", "ProAlign_GRU", "ProAlign_BERT4Rec"]:
            if epoch < args.freeze_embedding_epochs:
                # 冻结 ID Embedding
                if hasattr(model, 'ID_embeddings'):
                    for p in model.ID_embeddings.parameters():
                        p.requires_grad = False
                if epoch == 0:
                    logger.info(f"[Freeze Embedding] Epoch {epoch}: ID Embedding FROZEN (warmup {args.freeze_embedding_epochs} epochs)")
            else:
                # 解冻 ID Embedding
                if hasattr(model, 'ID_embeddings'):
                    for p in model.ID_embeddings.parameters():
                        p.requires_grad = True
                if epoch == args.freeze_embedding_epochs:
                    logger.info(f"[Freeze Embedding] Epoch {epoch}: ID Embedding UNFROZEN")
        # ==================== [END NEW] ====================
        
        # ---------------------- 遍历训练数据的每个 batch ----------------------
        for batch in train_loader:
            
            batch_size = len(batch['seq'])  # 当前 batch 的样本数  256
            seq = batch['seq'].to(device)  # 历史序列，shape: (batch_size, seq_len)  (256,10)  (256,50)
            #len_seq = batch['len_seq'].to(device)
            target = batch['next'].to(device)  # 目标物品ID，shape: (batch_size,)    (256,)     (256,)
            
            optimizer.zero_grad()  # 清空梯度
            
            # ---------------------- 根据损失类型计算主损失 ----------------------
            if args.loss_type == "CE":
                # Cross-Entropy 损失：对所有物品计算 softmax
                loss = model.calculate_ce_loss(seq, target)
            elif args.loss_type == "BCE":
                # Binary Cross-Entropy 损失：负采样二分类
                loss = model.calculate_bce_loss(seq, target, args.neg_ratio)
            elif args.loss_type == "infoNCE":
                # [OPTIMIZED] ProAlign 系列（包含三个 backbone）使用自己的损失函数
                # ==================== [FIX 2024-12-16] 添加 ProAlign_GRU 和 ProAlign_BERT4Rec ====================
                if args.model_type in ["ProAlign", "ProAlign_GRU", "ProAlign_BERT4Rec"]:
                    # ProAlign 使用 calculate_loss_with_align，包含 L_rec + L_align + L_cluster
                    user_ids = batch.get('user_id', None) # (256,)
                    if user_ids is not None:
                        user_ids = user_ids.to(device) # (256,)
                    # [DEBUG] 首次调用时打印 user_ids 状态
                    if step == 0:
                        logger.info(f"[DEBUG] user_ids is None: {user_ids is None}") # [DEBUG] user_ids is None: False
                        if user_ids is not None:
                            logger.info(f"[DEBUG] user_ids shape: {user_ids.shape}, first 5: {user_ids[:5]}") # [DEBUG] user_ids shape: torch.Size([256]), first 5: tensor([20282, 14099,  7973, 14728, 17473], device='cuda:0')
                    loss = model.calculate_loss_with_align(seq, target, user_ids, args.neg_ratio, args.temperature)
                # ==================== [FIX 2024-12-16] BERT4Rec 系列也使用标准 InfoNCE ====================
                # 注释掉原有的 calculate_bert4rec_loss，统一使用 calculate_infonce_loss
                # 这样三个 backbone (SASRec, GRU4Rec, BERT4Rec) 使用相同的训练损失
                # elif args.model_type in ["BERT4Rec", "AlphaFuse_BERT4Rec", "MoRec_BERT4Rec", 
                #                          "WhitenRec_BERT4Rec", "LLMInit_BERT4Rec", "UniSRec_BERT4Rec",
                #                          "RLMRec_BERT4Rec", "LLMESR_BERT4Rec", "ProAlign_BERT4Rec"]:
                #     loss = model.calculate_bert4rec_loss(seq, target, args.neg_ratio, args.temperature)
                # ==================== [END FIX] ====================
                # ==================== [NEW 2024-12-16] IRLLRec 系列使用带 user_ids 的损失 ====================
                # [FIX 2024-12-23] IRLLRec_BERT4Rec 单独处理，使用 mask 策略
                elif args.model_type in ["IRLLRec", "IRLLRec_GRU"]:
                    # IRLLRec (SASRec/GRU) 需要 user_ids 来计算蒸馏损失
                    user_ids = batch.get('user_id', None)
                    if user_ids is not None:
                        user_ids = user_ids.to(device)
                    # [DEBUG] 首次调用时打印 user_ids 状态
                    if step == 0:
                        logger.info(f"[IRLLRec] user_ids is None: {user_ids is None}")
                        if user_ids is not None:
                            logger.info(f"[IRLLRec] user_ids shape: {user_ids.shape}, first 5: {user_ids[:5]}")
                    loss = model.calculate_infonce_loss(seq, target, args.neg_ratio, args.temperature, user_ids)
                # ==================== [END NEW] ====================
                # ==================== [FIX 2024-12-19] BERT4Rec 系列使用 Mask 训练 ====================
                # 与 LLM-ESR 一致：随机 mask 多个位置，使用 InfoNCE 损失
                # 推理时 predict 使用 is_training=False，在末尾添加 [MASK]
                # [FIX 2024-12-23] 添加 IRLLRec_BERT4Rec 到列表中
                elif args.model_type in ["BERT4Rec", "AlphaFuse_BERT4Rec", "MoRec_BERT4Rec", 
                                         "WhitenRec_BERT4Rec", "LLMInit_BERT4Rec", "UniSRec_BERT4Rec",
                                         "RLMRec_BERT4Rec", "LLMESR_BERT4Rec", "IRLLRec_BERT4Rec"]:
                    loss = model.calculate_bert4rec_mask_loss(seq, target, args.neg_ratio, args.temperature)
                    if step == 0:
                        logger.info(f"[BERT4Rec] Using Mask + InfoNCE training (LLM-ESR style)")
                # ==================== [END FIX] ====================
                else:
                    # 其他基线模型（SASRec, GRU4Rec 等）使用标准 InfoNCE
                    loss = model.calculate_infonce_loss(seq, target, args.neg_ratio, args.temperature)

            
            # ---------------------- RLMRec / RLMRec_GRU 模型的额外重建损失 ----------------------
            # ==================== [MODIFIED 2024-12-16] 支持 RLMRec 三个 backbone ====================
            if args.model_type in ["RLMRec", "RLMRec_GRU", "RLMRec_BERT4Rec"]:
                # ==================== [ORIGINAL] 物品侧重建损失 ====================
                if args.SR_aligement_type == 'con':
                    # 对比式对齐：将语言嵌入映射到ID嵌入空间
                    item_recon_loss = model.reconstruct_con_loss()
                elif args.SR_aligement_type == 'gen':
                    # 生成式对齐：将ID嵌入映射到语言嵌入空间
                    item_recon_loss = model.reconstruct_gen_loss()
                loss = loss + args.beta * item_recon_loss  # 加权求和
                # ==================== [END ORIGINAL] ====================
                
                # ==================== [NEW 2024-12-15] 用户侧重建损失 ====================
                if args.use_user_llm:
                    # 获取用户 ID
                    user_ids = batch.get('user_id', None)
                    if user_ids is not None:
                        user_ids = user_ids.to(device)
                        
                        # 获取用户序列表示（forward 输出）
                        user_embeds = model.forward(seq)  # [B, hidden_dim]
                        
                        # 计算用户侧对齐损失
                        if args.SR_aligement_type == 'con':
                            # 对比式：使用 infonce 或 cosine 模式
                            user_recon_loss = model.user_alignment_loss(
                                user_embeds, user_ids, 
                                mode=args.user_align_mode, 
                                temperature=args.user_align_temp
                            )
                        elif args.SR_aligement_type == 'gen':
                            # 生成式：序列表示 → LLM 语义
                            user_recon_loss = model.user_alignment_loss_gen(user_embeds, user_ids)
                        
                        loss = loss + args.alpha_user * args.beta * user_recon_loss
                        
                        # [DEBUG] 首次调用时打印信息
                        if step == 0:
                            logger.info(f"[RLMRec User] user_embeds shape: {user_embeds.shape}")
                            logger.info(f"[RLMRec User] user_recon_loss: {user_recon_loss.item():.4f}")
                            logger.info(f"[RLMRec User] mode: {args.user_align_mode}, alpha_user: {args.alpha_user}")
                # ==================== [END NEW] ====================
            
            # ---------------------- LLMESR 模型的正则化损失 + RASD 损失 ----------------------
            # ==================== [FIX 2024-12-16] 添加 LLMESR_BERT4Rec ====================
            if args.model_type in ["LLMESR", "LLMESR_GRU", "LLMESR_BERT4Rec"]:
                # ==================== [ORIGINAL] 双视图对比正则化损失 ====================
                recon_loss = model.reg_loss(seq)
                loss = loss + args.beta * recon_loss
                
                # ==================== [NEW] RASD 对齐损失 ====================
                if args.use_rasd:
                    # 获取相似用户序列
                    sim_seqs = batch.get('sim_seqs', None)
                    if sim_seqs is not None:
                        sim_seqs = sim_seqs.to(device)  # [B, K, seq_len]
                        # 计算 RASD 对齐损失
                        rasd_loss = model.calculate_rasd_loss(seq, sim_seqs, args.user_sim_func)
                        loss = loss + args.alpha_rasd * rasd_loss
                        
                        # [DEBUG] 首次调用时打印信息
                        if step == 0:
                            logger.info(f"[RASD] sim_seqs shape: {sim_seqs.shape}")
                            logger.info(f"[RASD] rasd_loss: {rasd_loss.item():.4f}")
                # ==================== [END NEW] ====================
            
            # ==================== [NEW] ProAlign 模型的 RASD 损失 ====================
            # ==================== [FIX 2024-12-16] 添加 ProAlign_BERT4Rec ====================
            if args.model_type in ["ProAlign", "ProAlign_GRU", "ProAlign_BERT4Rec"] and args.use_rasd:
                # 获取相似用户序列
                sim_seqs = batch.get('sim_seqs', None)
                if sim_seqs is not None:
                    sim_seqs = sim_seqs.to(device)  # [B, K, seq_len]
                    # 计算 RASD 对齐损失
                    rasd_loss = model.calculate_rasd_loss(seq, sim_seqs, args.user_sim_func)
                    loss = loss + args.alpha_rasd * rasd_loss
                    
                    # [DEBUG] 首次调用时打印信息
                    if step == 0:
                        logger.info(f"[ProAlign RASD] sim_seqs shape: {sim_seqs.shape}")
                        logger.info(f"[ProAlign RASD] rasd_loss: {rasd_loss.item():.4f}")
            # ==================== [END NEW] ====================
            
            # ==================== [FIX 2024-12-16] 支持 IRLLRec 三个 backbone ====================
            if args.model_type in ["IRLLRec", "IRLLRec_GRU", "IRLLRec_BERT4Rec"]:
                # 获取用户 ID
                user_ids = batch.get('user_id', None)
                if user_ids is not None:
                    user_ids = user_ids.to(device)
                    
                    # 获取用户序列表示
                    seq_output = model.forward(seq)  # [B, hidden_dim]
                    
                    # 计算 IRLLRec 的所有蒸馏损失
                    irllrec_losses = model.calculate_irllrec_loss(seq_output, user_ids, target)
                    
                    # 添加到总损失
                    loss = loss + irllrec_losses['total_irllrec_loss']
                    
                    # [DEBUG] 首次调用时打印信息
                    if step == 0:
                        logger.info(f"[IRLLRec] seq_output shape: {seq_output.shape}")
                        logger.info(f"[IRLLRec] kd_loss: {irllrec_losses['kd_loss'].item():.6f}")
                        logger.info(f"[IRLLRec] kd_int_loss: {irllrec_losses['kd_int_loss'].item():.6f}")
                        logger.info(f"[IRLLRec] kd_int_2_loss: {irllrec_losses['kd_int_2_loss'].item():.6f}")
                        logger.info(f"[IRLLRec] itm_loss: {irllrec_losses['itm_loss'].item():.6f}")
                        logger.info(f"[IRLLRec] total_irllrec_loss: {irllrec_losses['total_irllrec_loss'].item():.6f}")
            # ==================== [END NEW] IRLLRec 意图蒸馏损失 ====================
            
            # [REMOVED] ProAlign 逻辑已移到前面，避免冗余计算


            # ---------------------- 反向传播和参数更新 ----------------------
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新模型参数
            step+=1  # 全局步数加1
            # print("loss in epoch {} iteration {}: {}".format(i, step, loss.item())) # expected 0.4~0.6 after init few epochs
        
        # 每个 epoch 结束后打印当前损失
        # print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs（原代码，已注释）
        # ==================== [NEW] 使用 logger ====================
        logger.info(f"loss in epoch {epoch} iteration {step}: {loss.item()}")
        # ==================== [END NEW] ====================
        #if (epoch+1) % args.lr_delay_epoch == 0:
        #    scheduler.step()
        #    print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()}")
        # ---------------------- 定期在训练集上评估（用于监控训练期间过拟合，可以删掉不影响训练结果） ----------------------
        if (epoch+1) % 50 == 0:
            _ = evaluate(model, train_loader, device)  # 训练集评估不需要屏蔽
        
        # ---------------------- 每个 epoch 在验证集上评估 ----------------------
        if (epoch+1) % 1 == 0:
            model.eval()  # 设置模型为评估模式（关闭Dropout等）
            # ==================== [NEW 2025-01-17] 推理效率优化 ====================
            if args.efficient_inference and hasattr(model, 'precompute_for_inference'):
                model.precompute_for_inference()
            # ==================== [END NEW] ====================
            # print('-------------------------- EVALUATE PHRASE --------------------------')  # 原代码，已注释
            # ==================== [NEW] 使用 logger ====================
            logger.info('-------------------------- EVALUATE PHRASE --------------------------')
            # ==================== [END NEW] ====================
            t0 = time.time()
            val_ndcg20 = evaluate(model, val_loader, device, rating_matrix=valid_rating_matrix)  # 在验证集上评估，返回 NDCG@20
            t1 = time.time() - t0
            # print("\n using ",t1, "s ", "Eval Time Cost",T,"s.")  # 原代码，已注释
            # ==================== [NEW] 使用 logger ====================
            logger.info(f"\n using {t1}s, Eval Time Cost {T}s.")
            # ==================== [END NEW] ====================

            model.train()  # 评估完成后恢复训练模式
            # ==================== [NEW 2025-01-17] 清除推理缓存 ====================
            if args.efficient_inference and hasattr(model, 'clear_inference_cache'):
                model.clear_inference_cache()
            # ==================== [END NEW] ====================
            tv_ndcg20 = val_ndcg20 
            
            # ---------------------- 早停判断和模型保存 ----------------------
            if tv_ndcg20 > best_ndcg20:
                # 验证集指标提升，保存当前最佳模型
                best_ndcg20 = tv_ndcg20
                counter = 0  # 重置早停计数器
                # 保存模型权重到文件
                # print("\n best NDCG@20 is updated to ",best_ndcg20,"at epoch",epoch)  # 原代码，已注释
                # ==================== [NEW] 使用 logger 和新的保存路径 ====================
                logger.info(f"\n best NDCG@20 is updated to {best_ndcg20} at epoch {epoch}")
                # ==================== [END NEW] ====================
                # 模型文件名包含：模型类型、随机种子、ID维度、零空间维度、学习率、损失类型
                # epoch_str = f"{args.model_type}_rs{args.random_seed}_IDdim{args.hidden_dim}_Textdim{args.null_dim}_{args.lr}_{args.loss_type}.pth"  # 原代码，已注释
                # torch.save(model.state_dict(), model_directory + epoch_str)  # 原代码，已注释
                # ==================== [NEW] 使用 checkpoint_path（与 BSARec 一致）====================
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Model saved to {checkpoint_path}")
                # ==================== [END NEW] ====================
            else:
                # 验证集指标没有提升
                counter += 1  # 早停计数器加1
                if counter >= patience:
                    # 连续 patience 个 epoch 没有提升，触发早停
                    # ==================== [NEW] 日志记录早停 ====================
                    logger.info("Early stopping")
                    # ==================== [END NEW] ====================
                    break   # 停止训练循环
            # print('----------------------------------------------------------------')  # 原代码，已注释
            # ==================== [NEW] 使用 logger ====================
            logger.info('----------------------------------------------------------------')
    
    # =============================================================================
    # 训练结束后的最终测试阶段
    # =============================================================================
    
    # ---------------------- 加载验证集上表现最好的模型 ----------------------
    # model.load_state_dict(torch.load(model_directory + epoch_str))  # 加载最佳模型权重（原代码，已注释）
    # ==================== [NEW] 使用 checkpoint_path 加载模型 ====================
    logger.info(f"Loading best model from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))  # 加载最佳模型权重
    # ==================== [END NEW] ====================
    items_emb = model.return_item_emb()  # 获取所有物品的嵌入表示（暂时没有用到，可用于分析或可视化） (18358,128)   调用 backbone_SASRec.py 的 def return_item_emb(self,) 的 def return_item_emb(self,):
    #with open('./item_emb/ATV_SASRec_ID64.pickle', 'wb') as f:
    #    pickle.dump(items_emb, f)
    
    # ---------------------- 在测试集上进行最终评估 ----------------------
    model.eval()  # 设置为评估模式
    # ==================== [NEW 2025-01-17] 推理效率优化 ====================
    if args.efficient_inference and hasattr(model, 'precompute_for_inference'):
        model.precompute_for_inference()
    # ==================== [END NEW] ====================
    # print('-------------------------- TEST RESULTS --------------------------')  # 原代码，已注释
    # ==================== [NEW] 使用 logger ====================
    logger.info('-------------------------- TEST RESULTS --------------------------')
    # ==================== [END NEW] ====================
    _ = evaluate(model, test_loader, device, rating_matrix=test_rating_matrix)  # 在测试集上评估，输出 HR@K, NDCG@K, MRR@K
    # print("Done.")  # 训练和评估全部完成（原代码，已注释）
    # ==================== [NEW] 使用 logger 并记录实验名称 ====================
    logger.info(f"Experiment: {args.train_name}")
    logger.info("Done.")
    # ==================== [END NEW] ==================== 