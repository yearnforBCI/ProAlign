import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# =============================================================================
# Item_Embedding 类：物品嵌入层（DreamRec 版本）
# 这是 DreamRec 系列模型共用的嵌入层，与 backbone_SASRec.py 中的同名类功能相同     embedding.py  是给 DreamRec 扩散模型用的，不是完全没用！
# 支持多种嵌入策略：
#   - ID: 纯 ID 嵌入（随机初始化）
#   - SI: 语义初始化（用 LLM 嵌入初始化 ID 嵌入）
#   - SR: 语义重建（ID 嵌入 + 语言嵌入用于重建损失）
#   - Dual_view: 双视图（LLMESR 用）
#   - AP: 自适应投影（语言嵌入 + 适配器）
#   - WAP: 白化自适应投影（白化后的语言嵌入 + 适配器）
#   - AF: AlphaFuse（零空间融合）
# =============================================================================
class Item_Embedding(nn.Module):
    def __init__(self, emb_pipline, **key_words):
        """
        初始化物品嵌入层
        
        Args:
            emb_pipline: 嵌入策略类型 ("ID"/"SI"/"SR"/"Dual_view"/"AP"/"WAP"/"AF")
            key_words: 包含各种配置参数的字典
        """
        super(Item_Embedding, self).__init__()
        # 读取数据统计信息（序列长度、物品数量）
        data_statis = pd.read_pickle(os.path.join(key_words["language_embs_path"], 'data_statis.df'))  
        self.state_size = data_statis['seq_size'][0]  # 序列长度
        self.item_num = data_statis['item_num'][0]    # 物品数量
        # 根据嵌入策略构建嵌入层
        self.construct_item_embeddings(emb_pipline, **key_words)
            
    def construct_item_embeddings(self, emb_pipline, **key_words):
        """
        根据嵌入策略构建物品嵌入层
        
        Args:
            emb_pipline: 嵌入策略类型
        """
        # -------------------- ID: 纯 ID 嵌入（DreamRec 基线）--------------------
        if emb_pipline == "ID":
            # 随机初始化 ID 嵌入，不使用任何语义信息
            self.init_ID_embedding(key_words["hidden_dim"], key_words["ID_embs_init_type"])
        
        # -------------------- SI: 语义初始化（LLMInit）--------------------
        elif emb_pipline == "SI": # semantic initialization
            # 用 LLM 语言嵌入初始化 ID 嵌入，之后可微调
            self.init_ID_embedding(key_words["hidden_dim"], "language_embeddings", **key_words)
        
        # -------------------- SR: 语义重建（RLMRec）--------------------
        elif emb_pipline == "SR": # semantic reconstruction
            # ID 嵌入随机初始化，同时加载冻结的语言嵌入用于重建损失
            self.init_ID_embedding(key_words["hidden_dim"], key_words["ID_embs_init_type"], **key_words)
            language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"], key_words["language_embs_scale"])
            #padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
            #language_embs = np.vstack([language_embs, padding_emb])
            # 语言嵌入冻结，仅用于计算重建损失
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs,dtype=torch.float32),
                freeze=True,
                )
        
        # -------------------- Dual_view: 双视图（LLMESR）--------------------
        elif emb_pipline == "Dual_view": # Dual view modeling of LLNESR
            # 同时使用 ID 嵌入和语言嵌入，通过交叉注意力融合
            self.init_ID_embedding(key_words["hidden_dim"], "language_embeddings", **key_words)
            language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"], key_words["language_embs_scale"])
            padding_emb = np.random.rand(language_embs.shape[1])  # padding 位置用随机向量
            language_embs = np.vstack([language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs,dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
                )
        
        # -------------------- AP: 自适应投影（MoRec/UniSRec）--------------------
        elif emb_pipline == "AP": # Adaptive Projection
            # 加载语言嵌入，通过适配器（MLP/MoE）投影到隐藏空间
            language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"], key_words["language_embs_scale"])
            padding_emb = np.random.rand(language_embs.shape[1])  # padding 位置用随机向量
            language_embs = np.vstack([language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs,dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
                )
        
        # -------------------- WAP: 白化自适应投影（WhitenRec）--------------------
        elif emb_pipline == "WAP": # Adaptive Projection for whitened language embeddings
            # 对语言嵌入进行 PCA 白化处理，消除各维度的相关性
            key_words["item_frequency_flag"] = False
            key_words['standardization'] = True
            language_embs = self.semantic_space_decomposion( None, **key_words)
            padding_emb = np.random.rand(language_embs.shape[1])  # padding 位置用随机向量
            language_embs = np.vstack([language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs,dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
                )
        
        # -------------------- AF: AlphaFuse（本文方法）--------------------
        elif emb_pipline == "AF": # AlphaFuse
            # 核心创新：在语言嵌入的零空间中注入 ID 信息
            # 1. 对语言嵌入进行 SVD 分解，识别零空间（方差小的维度）
            # 2. 语言嵌入投影到主成分空间（冻结）
            # 3. ID 嵌入只学习零空间维度，与语言嵌入相加融合
            cliped_language_embs = self.semantic_space_decomposion( key_words["hidden_dim"], **key_words)
            padding_emb = np.random.rand(cliped_language_embs.shape[1])  # padding 位置用随机向量
            cliped_language_embs = np.vstack([cliped_language_embs, padding_emb])
            # 语言嵌入冻结
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(cliped_language_embs,dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
                )
            # ID 嵌入只学习零空间维度（nullity 维）
            self.init_ID_embedding(self.nullity, key_words["ID_embs_init_type"])
            #self.init_ID_embedding(self.nullity, "zeros")        
        
    def load_language_embeddings(self, directory, language_model_type, scale):
        """
        加载预计算的 LLM 语言嵌入
        
        Args:
            directory: 数据目录路径
            language_model_type: 语言模型类型 ("3small" 或 "3large")
            scale: 缩放因子（放大嵌入值，避免数值过小）
            
        Returns:
            language_embs: [item_num, language_dim] 的语言嵌入矩阵
        """
        # 从 pickle 文件加载语言嵌入
        language_embs = pd.read_pickle(os.path.join(directory, language_model_type + '_emb.pickle'))
        self.item_num = len(language_embs)           # 物品数量
        self.language_dim = len(language_embs[0])    # 语言嵌入维度（3small=1536, 3large=3072）
        return np.stack(language_embs) * scale       # 堆叠并缩放
    
    def init_ID_embedding(self, ID_dim, init_type, **key_words):
        """
        初始化 ID 嵌入层
        
        Args:
            ID_dim: ID 嵌入维度
            init_type: 初始化方式
                - "language_embeddings": 用语言嵌入初始化（可微调）
                - "normal": 标准正态分布初始化
                - "zeros": 零初始化
                - "uniform": 均匀分布初始化
                - "ortho": 正交初始化
                - "xavier": Xavier 初始化
                - "sparse": 稀疏初始化
        """
        if init_type == "language_embeddings":
            # 用 LLM 语言嵌入初始化 ID 嵌入（可微调，freeze=False）
            language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"], key_words["language_embs_scale"])
            if self.language_dim == ID_dim:
                # 语言嵌入维度与 ID 维度相同，直接使用
                padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
                language_embs = np.vstack([language_embs, padding_emb])
                #language_embs = np.vstack([language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(language_embs, dtype=torch.float32),
                    freeze=False,   # 可微调
                    padding_idx=self.item_num
                    )
            else:
                # 语言嵌入维度与 ID 维度不同，需要 PCA 降维
                clipped_language_embs = self.semantic_space_decomposion(ID_dim, **key_words)
                padding_emb = np.random.rand(clipped_language_embs.shape[1])  # padding ID embedding
                clipped_language_embs = np.vstack([clipped_language_embs, padding_emb])
                #language_embs = np.vstack([language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(clipped_language_embs, dtype=torch.float32),
                    freeze=False,   # 可微调
                    padding_idx=self.item_num
                    )
        else:
            # 随机初始化 ID 嵌入
            self.ID_embeddings = nn.Embedding(
                num_embeddings=self.item_num+1,  # +1 是 padding 位置
                embedding_dim=ID_dim,
            )
            # 根据 init_type 选择初始化方式
            if init_type == "uniform":
                nn.init.uniform_(self.ID_embeddings.weight, a=0.0, b=1.0)   # U(0, 1)
            elif init_type == "normal":
                nn.init.normal_(self.ID_embeddings.weight, 0, 1)            # N(0, 1)
            elif init_type == "zeros":
                nn.init.zeros_(self.ID_embeddings.weight)                   # 全零
            elif init_type == "ortho":
                nn.init.orthogonal_(self.ID_embeddings.weight, gain=1.0)    # 正交矩阵
            elif init_type == "xavier":
                nn.init.xavier_uniform_(self.ID_embeddings.weight, gain=1.0) # Xavier
            elif init_type == "sparse":
                nn.init.sparse_(self.ID_embeddings.weight, 0.01, std=1)     # 稀疏矩阵
            else:
                raise NotImplementedError("This kind of init for ID embeddings is not implemented yet.")
                
    def semantic_space_decomposion(self, clipped_dim, **key_words):
        """
        语义空间分解（AlphaFuse 的核心算法）
        
        这是 AlphaFuse 的核心创新：对语言嵌入进行 SVD 分解，
        识别"零空间"（方差小的维度），用于注入 ID 信息。
        
        算法步骤：
        1. 计算语言嵌入的协方差矩阵
        2. SVD 分解得到特征向量 U 和特征值 S
        3. 根据阈值或维度确定零空间
        4. 可选：白化标准化（使各方向方差为 1）
        5. 投影到主成分空间
        
        Args:
            clipped_dim: 目标维度（投影后的维度）
            
        Returns:
            clipped_language_embs: [item_num, clipped_dim] 投影后的语言嵌入
        """
        # 加载语言嵌入
        language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"], key_words["language_embs_scale"])
        
        # 计算协方差矩阵
        if not key_words["item_frequency_flag"]:
            # 默认：均匀分布（所有物品权重相同）
            self.language_mean = np.mean(language_embs, axis=0)  # 计算均值
            cov = np.cov( language_embs - self.language_mean, rowvar=False)  # 计算协方差
        else:
            # 可选：按物品频率加权
            items_pop = np.load(os.path.join(key_words["language_embs_path"], 'items_pop.npy'))
            items_freq_scale = 1.0 / items_pop.sum()
            items_freq = (items_pop*items_freq_scale).reshape(-1, 1)
            self.language_mean = np.sum(language_embs*items_freq, axis=0)
            cov = np.cov( (language_embs - self.language_mean)*np.sqrt(items_freq), rowvar=False)
            #raise NotImplementedError("Custom item distribution is not implemented yet.")
        
        # SVD 分解：Cov = U @ diag(S) @ U^T
        # U: 特征向量矩阵（列为主成分方向）
        # S: 特征值（各方向的方差）
        U, S, _ = np.linalg.svd(cov, full_matrices=False)
        
        # 确定零空间维度（nullity）
        if key_words["null_thres"] is not None:
            # 方式1：根据阈值确定（特征值 < 阈值的维度为零空间）
            indices_null = np.where(S <= key_words["null_thres"])[0]
            self.nullity = len(indices_null)
        elif key_words["null_dim"] is not None:
            # 方式2：直接指定零空间维度
            self.nullity = key_words["null_dim"]
        #print("The Nullity is", self.nullity)
        #self.squared_singular_values = S
        #self.language_bases = U
        
        # 确定投影维度
        if clipped_dim is None:
            clipped_dim = self.language_dim
        if key_words["cover"]:
            # cover 模式：语言嵌入维度减去零空间维度
            clipped_dim = clipped_dim - self.nullity
            print()
        
        # 构造投影矩阵
        Projection_matrix = U[...,:clipped_dim]  # 取前 clipped_dim 个主成分
        
        # 可选：白化标准化（使各方向方差为 1）
        if key_words['standardization']:
            Diagnals = np.sqrt(1/S)[:clipped_dim]  # 1/sqrt(特征值)
            Projection_matrix = Projection_matrix.dot(np.diag(Diagnals)) # V_{\lambda} -> V_1
        
        # 投影：(X - mean) @ Projection_matrix
        clipped_language_embs = (language_embs-self.language_mean).dot(Projection_matrix)
        return clipped_language_embs


# =============================================================================
# AlphaFuse_embs 类：AlphaFuse 嵌入层（高级版本）
# 这是一个更灵活的 AlphaFuse 嵌入实现，支持多种 ID 空间和注入空间的组合
# 
# 主要特性：
# 1. 支持 4 种空间组合：
#    - singular + singular: ID 和注入都在奇异值空间
#    - euclidean + singular: ID 在欧氏空间，注入在奇异值空间
#    - singular + euclidean: ID 在奇异值空间，注入在欧氏空间
#    - euclidean + euclidean: ID 和注入都在欧氏空间
# 2. 支持 cover 模式（替换零空间）和非 cover 模式（相加）
# 3. 支持白化标准化
# =============================================================================
class AlphaFuse_embs(nn.Module):
    def __init__(self, data_directory, emb_std, emb_type, emb_dim, emb_init_type, null_thres, null_dim, standardization, cover, ID_space, inject_space):
        """
        初始化 AlphaFuse 嵌入层
        
        Args:
            data_directory: 数据目录路径
            emb_std: 嵌入缩放因子
            emb_type: 语言模型类型 ("3small" 或 "3large")
            emb_dim: 目标嵌入维度
            emb_init_type: ID 嵌入初始化方式
            null_thres: 零空间阈值（特征值 < 阈值的维度）
            null_dim: 零空间维度（直接指定）
            standardization: 是否白化标准化
            cover: 是否覆盖模式（True=替换，False=相加）
            ID_space: ID 嵌入空间 ("singular" 或 "euclidean")
            inject_space: 注入空间 ("singular" 或 "euclidean")
        """
        super(AlphaFuse_embs, self).__init__()
        # 加载基础语言嵌入
        base_embs = self.load_base_embs(data_directory, emb_type, emb_std)
        self.emb_dim = emb_dim
        # 构建零空间（SVD 分解）
        self.construct_null_space(base_embs, null_thres=null_thres, null_dim=null_dim)
        #print(standardization)
        #print(cover)
        #print(ID_space)
        #print(inject_space)
        # 初始化注入函数和返回函数
        self.inject, self.return_embs = self.init_injection(
            base_embs, 
            standardization=standardization,
            cover=cover,
            ID_space=ID_space,
            inject_space=inject_space
            )
        # 初始化 ID 嵌入权重
        self.ID_embs_init(emb_init_type)
    
    def ID_embs_init(self, emb_init_type):
        """
        初始化 ID 嵌入权重
        
        Args:
            emb_init_type: 初始化方式
        """
        if emb_init_type == "uniform":
            nn.init.uniform_(self.ID_embeddings.weight, a=0.0, b=1.0)   # U(0, 1)
        elif emb_init_type == "normal":
            nn.init.normal_(self.ID_embeddings.weight, 0, 1)            # N(0, 1)
        elif emb_init_type == "zero":
            nn.init.zeros_(self.ID_embeddings.weight)                   # 全零
        elif emb_init_type == "ortho":
            nn.init.orthogonal_(self.ID_embeddings.weight, gain=1.0)    # 正交
        elif emb_init_type == "xavier":
            nn.init.xavier_uniform_(self.ID_embeddings.weight, gain=1.0) # Xavier
        elif emb_init_type == "sparse":
            nn.init.sparse_(self.ID_embeddings.weight, 0.01, std=1)     # 稀疏

        
    def load_base_embs(self,data_directory, emb_type, emb_std):
        """
        加载基础语言嵌入
        
        Args:
            data_directory: 数据目录
            emb_type: 嵌入类型 ("3small" 或 "3large")
            emb_std: 缩放因子
            
        Returns:
            base_embs: [item_num, language_dim] 语言嵌入矩阵
        """
        text_embs = pd.read_pickle(os.path.join(data_directory, emb_type+'_emb.pickle'))
        self.item_num = len(text_embs)
        return np.stack(text_embs) * emb_std
    
    def construct_null_space(self, base_embs, null_thres=None, null_dim=None):
        """
        构建零空间（SVD 分解）
        
        通过 SVD 分解识别语言嵌入的零空间（方差小的维度）
        
        Args:
            base_embs: 语言嵌入矩阵
            null_thres: 零空间阈值
            null_dim: 零空间维度
        """
        # 计算均值和协方差
        self.mean = np.mean(base_embs, axis=0)
        cov = np.cov( base_embs - self.mean, rowvar=False)
        # SVD 分解
        U, S, _ = np.linalg.svd(cov, full_matrices=False)

        # 确定零空间维度
        if null_thres is not None:
            # 根据阈值确定
            indices_null = np.where(S <= null_thres)[0]
            indices_rank = np.where(S > null_thres)[0]
        elif null_dim is not None:
            # 直接指定
            indices = np.arange(len(S))
            indices_null = indices[-null_dim:]    # 最后 null_dim 维是零空间
            indices_rank = indices[:-null_dim]    # 其余是秩空间
            
        self.nullity = len(indices_null)
        print("The Nullity is", self.nullity)
        #self.S_null = S[indices_null]
        #self.S_rank = S[indices_rank]
        self.S = S                                          # 特征值
            # U[:, indices_rank].dot(np.diag(np.sqrt(1/S_rank)))
        self.U = U                                          # 特征向量矩阵
        self.U_null = torch.tensor(U[:, indices_null]).float()  # 零空间基向量
        #self.U_rank = U[:, indices_rank]
        return None
    
    def init_injection(self, base_embs, standardization=False, cover=False, ID_space="singular", inject_space="singular"):
        """
        初始化注入函数
        
        根据 ID 空间和注入空间的组合，创建不同的注入策略
        
        4 种组合：
        1. singular + singular: ID 和语言都在 SVD 变换后的空间
           - 最简单，直接在零空间维度相加/替换
        2. euclidean + singular: ID 在原始欧氏空间，注入在 SVD 空间
           - ID 嵌入先投影到零空间再相加
        3. singular + euclidean: ID 在 SVD 空间，注入在原始欧氏空间
           - ID 嵌入从零空间映射回欧氏空间
        4. euclidean + euclidean: ID 和注入都在原始欧氏空间
           - 通过 U_null @ U_null^T 投影到零空间
        
        Args:
            base_embs: 语言嵌入矩阵
            standardization: 是否白化
            cover: 是否覆盖零空间
            ID_space: ID 嵌入空间
            inject_space: 注入空间
            
        Returns:
            injection: 注入函数（获取融合后的物品嵌入）
            return_embs: 返回全量嵌入的函数
        """
        # ==================== 组合 1: singular + singular ====================
        # ID 和语言都在 SVD 变换后的奇异值空间
        # 这是最常用的模式
        if ID_space == "singular" and inject_space == "singular":
            P = self.U
            S = self.S
            if not cover:
                def injection(id, emb_type="both"):
                    x = self.text_embeddings(id)
                    #x_null = self.ID_embeddings(id)
                    y = x.clone()
                    if emb_type == "both":
                        x_null = self.ID_embeddings(id)
                        # Cover=False, hidden_dim=128, null_dim=64
                        #
                        # 最终 Item_Embedding = 128 维
                        # ├── 前 64 维：纯语义（强语义区，来自 LLM）
                        # └── 后 64 维：语义 + ID 混合（弱语义区 + ID 嵌入相加）
                        # 图示
                        # 语言嵌入 (128维)           ID 嵌入 (64维)
                        # ┌────────┬────────┐       ┌────────┐
                        # │ 前64维 │ 后64维 │   +   │  64维  │
                        # │ 强语义 │ 弱语义 │       │   ID   │
                        # └────────┴────────┘       └────────┘
                        #     ↓         ↓               ↓
                        #     │         └───── 相加 ────┘
                        #     ↓              ↓
                        # ┌────────┬────────────────┐
                        # │ 前64维  │    后64维      │
                        # │ 纯语义  │  语义+ID 混合   │  ← 最终 Item_Embedding (128维)
                        # └────────┴────────────────┘
                        y[..., -self.nullity:] = x[..., -self.nullity:] + x_null
                    elif emb_type == "id":
                        x_null = self.ID_embeddings(id)
                        y[..., :-self.nullity] = 0
                        y[..., -self.nullity:] = x_null 
                    return y
                def return_embs(emb_type="both"):
                    x = self.text_embeddings.weight
                    #x_null = self.ID_embeddings.weight
                    y = x.clone()
                    if emb_type == "both":
                        x_null = self.ID_embeddings.weight
                        y[..., -self.nullity:] = x[..., -self.nullity:] + x_null
                    elif emb_type == "id":
                        x_null = self.ID_embeddings.weight
                        y[..., :-self.nullity] = 0
                        y[..., -self.nullity:] = x_null 
                    #x[..., -self.nullity:] = x[..., -self.nullity:] + x_null
                    return y
            else:
                def injection(id, emb_type="both"):
                    x = self.text_embeddings(id)
                    x_null = self.ID_embeddings(id)
                    y = x.clone()
                    y[..., -self.nullity:] = 0
                    if emb_type == "both":
                        x_null = self.ID_embeddings(id)
                        y[..., -self.nullity:] = x_null
                    elif emb_type == "id":
                        x_null = self.ID_embeddings(id)
                        y[..., :-self.nullity] = 0
                        y[..., -self.nullity:] = x_null 
                    return y
                def return_embs(emb_type="both"):
                    x = self.text_embeddings.weight
                    #x_null = self.ID_embeddings.weight
                    #x[..., -self.nullity:] = 0
                    #x[..., -self.nullity:] = x[..., -self.nullity:] + x_null
                    y = x.clone()
                    y[..., -self.nullity:] = 0
                    if emb_type == "both":
                        x_null = self.ID_embeddings.weight
                        y[..., -self.nullity:] = x_null
                    elif emb_type == "id":
                        x_null = self.ID_embeddings.weight
                        y[..., :-self.nullity] = 0
                        y[..., -self.nullity:] = x_null 
                    return y
            # 白化标准化：使各方向方差为 1
            if standardization:
                P = P.dot(np.diag(np.sqrt(1/S)))  # P @ diag(1/sqrt(S))
            else:
                P = P
            #base_embs = base_embs.dot(P)
            # 投影到 SVD 空间
            base_embs = (base_embs-self.mean).dot(P[:,:self.emb_dim])
            # ID 嵌入只学习零空间维度
            self.ID_embeddings = nn.Embedding(
                num_embeddings=self.item_num+1,
                embedding_dim=self.nullity,
            )
        
        # ==================== 组合 2: euclidean + singular ====================
        # ID 在原始欧氏空间，注入在 SVD 空间
        # ID 嵌入需要先投影到零空间再相加
        elif ID_space == "euclidean"and inject_space == "singular":
            P = self.U
            S = self.S
            if not cover:
                def injection(id):
                    x = self.text_embeddings(id)
                    x_null = self.ID_embeddings(id)
                    #x_null = x_null @ self.U_null.to(x.device)
                    x_null = (x_null-torch.tensor(self.mean).to(x.device).float()) @ self.U_null.to(x.device)
                    x = x.clone()
                    x[..., -self.nullity:] = x[..., -self.nullity:] + x_null
                    return x
                def return_embs():
                    x = self.text_embeddings.weight
                    x_null = self.ID_embeddings.weight
                    #x_null = x_null @ self.U_null.to(x.device)
                    x_null = (x_null-torch.tensor(self.mean).to(x.device).float()) @ self.U_null.to(x.device)
                    x = x.clone()
                    x[..., -self.nullity:] = x[..., -self.nullity:] + x_null
                    return x
                    
            else:
                def injection(id):
                    x = self.text_embeddings(id)
                    x_null = self.ID_embeddings(id)
                    #x_null = x_null @ self.U_null.to(x.device)
                    x_null = (x_null-torch.tensor(self.mean).to(x.device).float()) @ self.U_null.to(x.device)
                    x = x.clone()
                    x[..., -self.nullity:] = 0
                    x[..., -self.nullity:] = x[..., -self.nullity:] + x_null
                    return x
                def return_embs():
                    x = self.text_embeddings.weight
                    x_null = self.ID_embeddings.weight
                    #x_null = x_null @ self.U_null.to(x.device)
                    x_null = (x_null-torch.tensor(self.mean).to(x.device).float()) @ self.U_null.to(x.device)
                    x = x.clone()
                    x[..., -self.nullity:] = 0
                    x[..., -self.nullity:] = x[..., -self.nullity:] + x_null
                    return x
            # 白化标准化
            if standardization:
                P = P.dot(np.diag(np.sqrt(1/S)))
            else:
                P = P
            #base_embs = base_embs.dot(P)
            # 投影到 SVD 空间
            base_embs = (base_embs-self.mean).dot(P[:,:self.emb_dim])
            # ID 嵌入维度与投影后的语言嵌入相同
            self.ID_embeddings = nn.Embedding(
                num_embeddings=self.item_num+1,
                embedding_dim=base_embs.shape[-1],
            )
        
        # ==================== 组合 3: singular + euclidean ====================
        # ID 在 SVD 空间，注入在原始欧氏空间
        # ID 嵌入需要从零空间映射回欧氏空间
        elif ID_space == "singular"and inject_space == "euclidean":
            P = self.U
            self.ID_embeddings = nn.Embedding(
                num_embeddings=self.item_num+1,
                embedding_dim=self.nullity,
            )
            if standardization:
                singulars = np.ones(base_embs.shape[-1])
                singulars[:-self.nullity] = np.sqrt(1/self.S[:-self.nullity])
                P = P.dot(np.dot(np.diag(singulars),P.T))
                #base_embs = base_embs.dot(P)
                base_embs = (base_embs-self.mean).dot(P) + self.mean
            def injection(id):
                x = self.text_embeddings(id)
                x_null = self.ID_embeddings(id)
                #x_null = x_null @ self.U_null.T.to(x.device)
                x_null = x_null @ self.U_null.T.to(x.device) + torch.tensor(self.mean).to(x.device).float()
                return x + x_null
            def return_embs():
                x = self.text_embeddings.weight
                x_null = self.ID_embeddings.weight
                #x_null = x_null @ self.U_null.T.to(x.device)
                # 从零空间映射回欧氏空间：x_null @ U_null^T + mean
                x_null = x_null @ self.U_null.T.to(x.device) + torch.tensor(self.mean).to(x.device).float()
                return x + x_null
        
        # ==================== 组合 4: euclidean + euclidean ====================
        # ID 和注入都在原始欧氏空间
        # 通过 U_null @ U_null^T 投影到零空间
        elif ID_space == "euclidean"and inject_space == "euclidean":
            P = self.U
            self.ID_embeddings = nn.Embedding(
                num_embeddings=self.item_num+1,
                embedding_dim=base_embs.shape[-1],
            )
            if standardization:
                singulars = np.ones(base_embs.shape[-1])
                singulars[:-self.nullity] = np.sqrt(1/self.S[:-self.nullity])
                P = P.dot(np.dot(np.diag(singulars),P.T))
                #base_embs = base_embs.dot(P)
                base_embs = (base_embs-self.mean).dot(P) + self.mean
            self.UUT = torch.tensor(np.dot(self.U_null,self.U_null.T)).float()
            def injection(id):
                x = self.text_embeddings(id)
                x_null = self.ID_embeddings(id)
                #x_null = x_null @ self.UUT.to(x.device)
                x_null = (x_null-torch.tensor(self.mean).to(x.device).float()) @ self.UUT.to(x.device) + torch.tensor(self.mean).to(x.device).float()
                return x + x_null
            def return_embs():
                x = self.text_embeddings.weight
                x_null = self.ID_embeddings.weight
                #x_null = x_null @ self.UUT.to(x.device)
                x_null = (x_null-torch.tensor(self.mean).to(x.device).float()) @ self.UUT.to(x.device) + torch.tensor(self.mean).to(x.device).float()
                return x + x_null
        
        # ==================== 创建语言嵌入层 ====================
        # 添加 padding 向量（随机初始化）
        padding_vector = np.random.randn(base_embs.shape[-1])  
        base_embs = np.vstack([base_embs, padding_vector])
        # 语言嵌入冻结，不参与训练
        self.text_embeddings = nn.Embedding.from_pretrained(torch.tensor(base_embs,dtype=torch.float32), freeze=True, padding_idx=self.item_num)
        
        return injection, return_embs