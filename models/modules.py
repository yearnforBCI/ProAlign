import numpy as np
import math
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 辅助函数：从序列中按索引提取特定时间步
# =============================================================================
def extract_axis_1(data, indices):
    """
    从 3D 张量中按索引提取特定时间步
    
    Args:
        data: [B, S, D] 输入张量
        indices: [B] 每个样本要提取的时间步索引
        
    Returns:
        res: [B, 1, D] 提取的张量
    """
    res = [] # 初始化结果列表
    for i in range(data.shape[0]): # 遍历 batch 中的每个样本
        res.append(data[i, indices[i], :])  # 提取第 i 个样本的第 indices[i] 个时间步
    res = torch.stack(res, dim=0).unsqueeze(1) # 堆叠并增加维度 [B, D] -> [B, 1, D]
    return res # 返回提取后的张量

class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        """
        Args:
            input_size: 输入维度
            output_size: 输出维度
            dropout: Dropout 概率
        """
        super(PWLayer, self).__init__() # 调用父类初始化

        self.dropout = nn.Dropout(p=dropout) # Dropout 层
        # 可学习的偏置（用于去中心化）
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True) # 初始化偏置参数
        # 线性变换（无偏置）
        self.lin = nn.Linear(input_size, output_size, bias=False) # 线性层（如果不使用偏置）

        self.apply(self._init_weights) # 应用权重初始化

    def _init_weights(self, module):
        """权重初始化：正态分布 N(0, 0.02)"""
        if isinstance(module, nn.Linear): # 如果是线性层
            module.weight.data.normal_(mean=0.0, std=0.02) # 使用正态分布初始化权重

    def forward(self, x):
        """前向传播：先去中心化，再线性变换"""
        return self.lin(self.dropout(x) - self.bias) # 计算公式：Linear(Dropout(x) - bias)

class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        """
        Args:
            n_exps: 专家数量
            layers: [input_dim, output_dim] 各层维度
            dropout: Dropout 概率
            noise: 是否使用噪声门控（训练时增加随机性）
        """
        super(MoEAdaptorLayer, self).__init__() # 调用父类初始化

        self.n_exps = n_exps          # 专家数量
        self.noisy_gating = noise     # 是否使用噪声门控

        # n_exps 个专家网络（每个是 PWLayer）
        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)]) # 创建专家列表
        # 门控权重：决定各专家的贡献比例
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True) # 初始化门控权重
        # 噪声权重：控制噪声强度
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True) # 初始化噪声权重

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
        带噪声的门控机制
        
        训练时添加噪声可以：
        1. 防止门控坍缩（只激活少数专家）
        2. 增加探索性（让更多专家参与学习）
        
        Args:
            x: [B, D] 输入
            train: 是否训练模式
            
        Returns:
            gates: [B, n_exps] 各专家的权重（softmax 归一化）
        """
        clean_logits = x @ self.w_gate  # [B, n_exps] 计算未归一化的门控分数
        if self.noisy_gating and train: # 如果启用噪声且处于训练模式
            # 训练时添加噪声
            raw_noise_stddev = x @ self.w_noise # 计算噪声标准差的原始值
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon)) # 计算实际噪声标准差（保证为正）
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev) # 添加随机噪声
            logits = noisy_logits # 使用带噪声的 logits
        else:
            # 推理时不加噪声
            logits = clean_logits # 使用原始 logits

        gates = F.softmax(logits, dim=-1)  # 归一化为概率分布
        return gates # 返回门控权重

    def forward(self, x):
        """
        前向传播
        
        流程：
        1. 计算门控权重
        2. 各专家分别处理输入
        3. 加权求和专家输出
        """
        gates = self.noisy_top_k_gating(x, self.training) # [B, n_exps] 计算门控权重
        # 各专家处理输入
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]让每个专家处理输入
        expert_outputs = torch.cat(expert_outputs, dim=-2)  # [B, n_exps, D] 拼接所有专家的输出
        # 加权求和
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs  # [B, n_exps, D] 权重 * 专家输出
        return multiple_outputs.sum(dim=-2)  # [B, D] 求和得到最终输出

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        """
        Args:
            d_in: 输入/输出维度
            d_hid: 隐藏层维度（通常是 d_in 的 4 倍）
            dropout: Dropout 概率
        """
        super().__init__() # 调用父类初始化
        # 使用 1x1 卷积实现线性变换
        #
        # BSARec 用线性层，这里用的是卷积，两者数学上等价
        # 实现方式	代码	                            输入格式
        # 1x1 卷积	nn.Conv1d(d_in, d_out, 1)	    (B, C, L)
        # 线性层	    nn.Linear(d_in, d_out)	        (B, L, C)
        #
        # 为什么用卷积？
        # 历史原因 / 代码风格不同：
        # 卷积：输入 (B, C, L)，不需要 transpose
        # 线性：输入 (B, L, C)，更直观
        # 效果完全一样，只是数据维度排列不同

        self.w_1 = nn.Conv1d(d_in, d_hid, 1)   # 升维：d_in -> d_hid  128——>128     Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)   # 降维：d_hid -> d_in  128——>128     Conv1d(128, 128, kernel_size=(1,), stride=(1,))
        self.layer_norm = nn.LayerNorm(d_in)   # 层归一化
        self.dropout = nn.Dropout(dropout) # Dropout 层  0.1

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, S, D] 输入
            
        Returns:
            output: [B, S, D] 输出（与输入维度相同）
        """
        residual = x                           # 保存残差                     (256,10,128)
        output = x.transpose(1, 2)             # [B, D, S]（卷积需要通道在前）  (256,128,10)
        output = self.w_2(F.relu(self.w_1(output)))  # FFN: W2(ReLU(W1(x)))  (256,128,10)
        output = output.transpose(1, 2)        # [B, S, D] 转回原来的形状      (256,10,128)
        output = self.dropout(output) # 应用 Dropout 0.1                      (256,10,128)
        output = self.layer_norm(output + residual)  # 残差连接 + LayerNorm    (256,10,128)
        return output # 返回输出

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        """
        Args:
            hidden_size: 输入维度
            num_units: 输出维度
            num_heads: 注意力头数
            dropout_rate: Dropout 概率
        """
        super().__init__() # 调用父类初始化
        self.hidden_size = hidden_size   # 128
        self.num_heads = num_heads       # 1
        assert hidden_size % num_heads == 0  # 确保能整除

        # Q, K, V 投影层
        self.linear_q = nn.Linear(hidden_size, num_units) # Query 投影层  查询权重矩阵 WQ  (128,128)
        self.linear_k = nn.Linear(hidden_size, num_units) # Key 投影层    键权重矩阵 WK
        self.linear_v = nn.Linear(hidden_size, num_units) # Value 投影层  值权重矩阵 WV
        self.dropout = nn.Dropout(dropout_rate) # Dropout 层  0.1
        self.softmax = nn.Softmax(dim=-1) # Softmax 层


    def forward(self, queries, keys, ):
        """
        多头自注意力前向传播
        
        :param queries: [N, T_q, C_q] 查询张量                     经过归一化的序列
        :param keys: [N, T_k, C_k] 键张量（也用于计算 V）           未经归一化的序列
        
        :return: [N, T_q, C] 注意力输出（带残差连接）
        """


        # Transformer 的两种 LayerNorm 放置方式
        # 在 Transformer 架构中，LayerNorm (LN) 的位置有两种主流设计：
        #
        # 1. Post-LN（原始 Transformer）
        # 输入 x
        #    │
        #    ▼
        # ┌──────────────┐
        # │  Attention   │
        # └──────────────┘
        #    │
        #    ▼
        #  x + Attention(x)   ← 残差连接
        #    │
        #    ▼
        # ┌──────────────┐
        # │  LayerNorm   │   ← LN 在残差连接之后
        # └──────────────┘
        #    │
        #    ▼
        #   输出
        # 公式：output = LayerNorm(x + Attention(x))
        #
        # 2. Pre-LN（变体，更稳定）
        # 输入 x ──────────────────────┐
        #    │                         │
        #    ▼                         │
        # ┌──────────────┐             │
        # │  LayerNorm   │   ← LN 在 Attention 之前
        # └──────────────┘             │
        #    │                         │
        #    ▼                         │
        # ┌──────────────┐             │
        # │  Attention   │             │
        # └──────────────┘             │
        #    │                         │
        #    ▼                         │
        #  Attention(LN(x)) + x  ◄─────┘  残差连接
        #    │
        #    ▼
        #   输出
        # 公式：output = x + Attention(LayerNorm(x))

        # 为什么 Pre-LN 更稳定？
        # 梯度流动分析
        # Post-LN 的问题：
        # output = LN(x + sublayer(x))
        # 梯度需要穿过 LayerNorm
        # 深层网络中梯度容易爆炸或消失
        #
        # Pre-LN 的优势：
        # output = x + sublayer(LN(x))
        # 残差路径 x 直接传递梯度
        # 梯度有一条"高速公路"直达底层
        # 类似于 ResNet 的设计理念


        # ==================== 线性投影 ====================
        # Q 来自第一个参数，K/V 来自第二个参数
        #
        # 这里的设计选择是：
        #
        # Q 用归一化后的 → 查询更稳定
        # K/V 用原始的 → 保留原始信息
        #
        # 可能的原因  对于AlphaFuse，本质上还是自注意力机制，不是交叉注意力
        # 1.设计选择	    效果
        # Q 归一化	    查询向量的尺度更稳定，注意力分数更平滑
        # K/V 不归一化	保留原始序列的完整信息供检索
        #
        # 2.代码复用
        # 这个 MultiHeadAttention 类设计成通用接口：
        # 传入两个相同的参数 = 自注意力
        # 传入两个不同的参数 = 交叉注意力（或变体）
        Q = self.linear_q(queries)  # (N, T_q, C) 计算 Query  (256,10,128)
        K = self.linear_k(keys)  # (N, T_k, C)    计算 Key    (256,10,128)
        V = self.linear_v(keys)  # (N, T_k, C)    计算 Value  (256,10,128)
        
        # ==================== 分割多头 ====================
        # 将维度 C 分割成 h 个头，每个头维度 C/h
        split_size = self.hidden_size // self.num_heads # 计算每个头的维度  128，这里使用的一个注意力头，可以改为两个注意力头
        # 两种实现方式对比
        # 方式 1：AlphaFuse（split + cat）
        # 步骤：
        # 1. split: [256, 10, 128] → [[256, 10, 128]]（1个头，所以只有1个块）
        # 2. cat:   在 dim=0 拼接 → [256, 10, 128]
        # 如果是 2 个头：
        # 1. split: [256, 10, 128] → [[256, 10, 64], [256, 10, 64]]
        # 2. cat:   在 dim=0 拼接 → [512, 10, 64]  即 [h*B, S, d]
        #
        # 方式 2：BSARec（view + permute）
        # 原始 Q: [B, S, H]  例如 [256, 50, 64]
        # view: [B, S, H] → [B, S, h, d]
        # permute: [B, S, h, d] → [B, h, S, d]
        # new_shape = x.size()[:-1] + (num_heads, head_size)  # [B, S, h, d]
        # Q_ = x.view(*new_shape).permute(0, 2, 1, 3)
        # [256, 50, 64] → [256, 50, 1, 64] → [256, 1, 50, 64]  即 [B, h, S, d]
        #
        # 对比表
        # 特性	        AlphaFuse (split+cat)	            BSARec (view+permute)
        # 输出形状	    [h*B, S, d]	                        [B, h, S, d]
        # 头的位置	    混入 batch 维	                    独立的维度
        # 注意力计算	    bmm(Q_, K_.T) → [h*B, S, S]	        matmul(Q, K.T) → [B, h, S, S]
        # 内存布局	    连续的大 batch	                    4D 张量
        #
        # 图示对比
        # AlphaFuse 方式（把头混入 batch）
        # 原始: [B=256, S=10, H=128]
        #
        # 2个头时（h=2, d=64）:
        # split: [[256, 10, 64], [256, 10, 64]]
        #  cat:  [512, 10, 64]  ← 形状是 [h*B, S, d]
        #
        #           头1 (256个样本)        头2 (256个样本)
        #       ┌─────────────────┐  ┌─────────────────┐
        #       │ [256, 10, 64]   │  │ [256, 10, 64]   │
        #       └─────────────────┘  └─────────────────┘
        #                ↓ cat(dim=0) ↓
        #       ┌─────────────────────────────────────┐
        #       │         [512, 10, 64]               │
        #       └─────────────────────────────────────┘
        #                    [h*B, S, d]
        #
        #
        # BSARec 方式（保持头为独立维度）
        # 原始: [B=256, S=50, H=64]
        #
        # 1个头时（h=1, d=64）:
        # view:    [256, 50, 1, 64]  ← [B, S, h, d]
        # permute: [256, 1, 50, 64]  ← [B, h, S, d]
        #
        #       ┌─────────────────────────────────────┐
        #       │    [256, 1, 50, 64]                 │
        #       │    [B,   h,  S,  d]                 │
        #       │                                     │
        #       │    头维度 h 是独立的一个轴             │
        #       └─────────────────────────────────────┘
        #
        # 学上等价吗？
        # ✅ 是的，数学上完全等价！
        #
        # 两种方式最终计算的注意力分数是相同的：
        # AlphaFuse:
        # attention = softmax(Q_ @ K_.T / √d)  # [h*B, S, S]
        # output = attention @ V_              # [h*B, S, d]
        # 然后 split+cat 回去
        #
        # BSARec:
        # attention = softmax(Q @ K.T / √d)    # [B, h, S, S]
        # output = attention @ V               # [B, h, S, d]
        # 然后 view+permute 回去
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  # (h*N, T_q, C/h) 分割 Q  (256,10,128)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)  # (h*N, T_k, C/h) 分割 K  (256,10,128)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  # (h*N, T_k, C/h) 分割 V  (256,10,128)
        
        # ==================== 计算注意力分数 ====================
        # Scaled Dot-Product Attention: softmax(QK^T / sqrt(d))
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5  # (h*N, T_q, T_k) 计算点积并缩放
        
        # ==================== Key Masking（屏蔽 padding）====================
        # 1	keys（未经归一化的序列）	    [B, S, D]	原始 Key 张量，例如 [256, 10, 128]
        # 2	.sum(dim=-1)	[B, S]	    对最后一维求和 → [256, 10]
        # 3	torch.abs(...)	[B, S]	    取绝对值
        # 4	torch.sign(...)	[B, S]	    符号函数：正数→1，0→0，负数→-1
        # 5	.repeat(h, 1)	[h*B, S]    复制 h 份（多头）→ [256, 10]（1头时不变）
        #
        # 核心逻辑：
        # padding 位置：嵌入向量通常是 0（或接近 0）
        #               → sum = 0 → abs = 0 → sign = 0 ← 标记为需要屏蔽
        #
        # 有效位置：   嵌入向量非零
        #               → sum ≠ 0 → abs > 0 → sign = 1 ← 标记为有效
        #
        # 示例：
        # keys = [[0.5, 0.3, 0.2],   # 有效物品 → sum=1.0 → sign=1
        #         [0.0, 0.0, 0.0],   # padding  → sum=0.0 → sign=0
        #         [0.1, 0.4, 0.5]]   # 有效物品 → sum=1.0 → sign=1
        #
        # key_mask = [1, 0, 1]  # 0 表示 padding，需要屏蔽
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_k) 生成 Key 掩码
        # 步骤	  操作                	形状变化	                说明
        # 1	    key_mask	            [h*B, S_k]	        例如 [256, 10]
        # 2	   .unsqueeze(1)	        [h*B, 1, S_k]	    添加查询维度 → [256, 1, 10]
        # 3	   .repeat(1, T_q, 1)	    [h*B, T_q, S_k]	    复制到所有查询位置 → [256, 10, 10]
        #
        # 为什么需要这个形状？
        # 注意力分数矩阵 [h*B, T_q, T_k]：
        #      Key位置 0   1   2   3   4  (T_k=5)
        # Query位置0  [s00, s01, s02, s03, s04]
        # Query位置1  [s10, s11, s12, s13, s14]
        # Query位置2  [s20, s21, s22, s23, s24]
        #   (T_q=3)
        #
        # key_mask_reshaped 需要匹配这个形状，
        # 每一行都是相同的 key_mask（因为每个 query 都要屏蔽相同的 key 位置）
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (h*N, T_q, T_k) 如果需要广播
        # 部分	                              说明
        # torch.ones_like(matmul_output)	创建与注意力分数相同形状的全 1 张量
        # * (-2 ** 32 + 1)	                乘以极小值 ≈ -4.29 × 10^9
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)  # 极小值
        # 部分	                                   说明
        # torch.eq(key_mask_reshaped, 0)	找出掩码为 0 的位置（padding 位置），返回 bool 张量
        # torch.where(condition, A, B)	    条件为 True 时取 A，否则取 B
        #
        # 效果：
        # condition = (key_mask_reshaped == 0)
        #
        # 如果是 padding 位置（mask=0）→ 用极小值替换
        # 如果是有效位置（mask=1）→ 保留原始注意力分数
        #
        #
        # 完整流程图
        # 原始注意力分数 matmul_output [256, 10, 10]：
        #      Key: 物品1  物品2  PAD   PAD   物品3
        # Query:
        # 物品1    [0.3,   0.5,   0.1,  0.2,  0.4]
        # 物品2    [0.2,   0.6,   0.3,  0.1,  0.5]
        # PAD      [0.1,   0.4,   0.2,  0.3,  0.2]
        # PAD      [0.3,   0.3,   0.1,  0.4,  0.1]
        # 物品3    [0.5,   0.2,   0.4,  0.2,  0.6]
        #
        # key_mask = [1, 1, 0, 0, 1]  ← 第3、4列是 padding
        #
        # 应用 Key Masking 后 matmul_output_m1：
        #      Key: 物品1  物品2  PAD      PAD      物品3
        # Query:
        # 物品1    [0.3,   0.5,   -∞,      -∞,      0.4]
        # 物品2    [0.2,   0.6,   -∞,      -∞,      0.5]
        # PAD      [0.1,   0.4,   -∞,      -∞,      0.2]
        # PAD      [0.3,   0.3,   -∞,      -∞,      0.1]
        # 物品3    [0.5,   0.2,   -∞,      -∞,      0.6]
        #                         ↑        ↑
        #                    padding 列被设为 -∞
        #
        # softmax 后：
        #      Key: 物品1  物品2  PAD   PAD   物品3
        # Query:
        # 物品1    [0.25,  0.40,  0,    0,    0.35]  ← padding 位置权重≈0
        # 物品2    [0.18,  0.52,  0,    0,    0.30]
        # ...
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  # (h*N, T_q, T_k) 应用 Key 掩码
        
        # ==================== Causality Masking（因果掩码）====================
        # 下三角矩阵：位置 t 只能看到 ≤ t 的位置，防止看到未来
        diag_vals = torch.ones_like(matmul_output[0, :, :])   # (T_q, T_k)
        tril = torch.tril(diag_vals)  # 下三角矩阵 (T_q, T_k)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)  # (h*N, T_q, T_k) 广播因果掩码
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1) # 极小值
        # 掩码类型	            作用      	                屏蔽位置
        # Key Masking	        屏蔽 padding	            列方向（某些 Key 位置）
        # Causality Masking	    屏蔽未来	                上三角区域
        # 两者叠加	        只看过去的有效位置	            padding列 + 上三角
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings, matmul_output_m1)  # (h*N, T_q, T_k) 应用因果掩码
        
        # ==================== Softmax 激活 ====================
        matmul_output_sm = self.softmax(matmul_output_m2)  # (h*N, T_q, T_k) 计算 Attention 权重
        
        # ==================== Query Masking（屏蔽 padding 的输出）====================
        #


        # 完整流程图
        # 原始注意力分数 [h*B, T_q, T_k]
        #         ↓
        # Key Masking（屏蔽 padding 列）→ 不关注 padding Key
        #         ↓
        # Causality Masking（屏蔽上三角）→ 不关注未来
        #         ↓
        # Softmax
        #         ↓
        # Query Masking（屏蔽 padding 行）→ padding Query 不产生输出
        #         ↓
        # matmul(attention, V) → 加权求和
        #
        #
        # BSARec 的实现更优雅，推荐学习 BSARec 的写法！两者最终效果相同，但实现方式有多处不同：
        #
        # 核心区别总结
        # AlphaFuse:                          BSARec:
        # ┌─────────────────────┐            ┌─────────────────────┐
        # │ 嵌入 → Dropout      │            │ 嵌入 → LN → Dropout │
        # │         ↓           │            │         ↓           │
        # │ LN(seq) → Q         │            │ seq → Q, K, V       │
        # │ seq → K, V          │            │         ↓           │
        # │         ↓           │            │ 统一加性掩码        │
        # │ 分离乘性掩码 x3     │            │         ↓           │
        # │         ↓           │            │ Softmax             │
        # │ Softmax             │            │         ↓           │
        # │         ↓           │            │ 输出                │
        # │ Query Masking       │            └─────────────────────┘
        # │         ↓           │
        # │ 输出                │
        # └─────────────────────┘
        #
        # 掩码	            时机	            作用	                            为什么需要
        # Key Masking	    softmax 前	    不关注 padding Key	            让 padding Key 的权重→0
        # Causality Masking	softmax 前	    不关注未来	                    让未来位置的权重→0
        # Query Masking	    softmax 后	    padding Query 输出→0	            softmax 归一化后 padding Query 的权重加起来仍是 1，需要额外置零
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_q) 生成 Query 掩码
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])  # (h*N, T_q, T_k) 广播 Query 掩码
        matmul_output_qm = matmul_output_sm * query_mask # 应用 Query 掩码
        
        # ==================== Dropout ====================
        matmul_output_dropout = self.dropout(matmul_output_qm) # 对 Attention 权重应用 Dropout
        
        # ==================== 加权求和 ====================
        output_ws = torch.bmm(matmul_output_dropout, V_)  # (h*N, T_q, C/h) 计算加权求和
        
        # ==================== 合并多头 ====================
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  # (N, T_q, C) 拼接多头输出
        
        # ==================== 残差连接 ====================
        output_res = output + queries # 添加残差连接
        
        return output_res # 返回最终输出

# =============================================================================
# CalculateAttention：注意力计算模块
# 实现 Scaled Dot-Product Attention
# =============================================================================
class CalculateAttention(nn.Module):

    def __init__(self):
        super().__init__()


    
    def forward(self, Q, K, V, mask):
        """
        计算缩放点积注意力
        
        Args:
            Q: [B, H, S, D] 查询
            K: [B, H, S, D] 键
            V: [B, H, S, D] 值
            mask: [B, 1, S, S] 注意力掩码
            
        Returns:
            attention: [B, H, S, D] 注意力输出
        """
        # 计算注意力分数：Q @ K^T
        attention = torch.matmul(Q,torch.transpose(K, -1, -2)) # 计算 Q 和 K 转置的矩阵乘法
        # 应用掩码（将 padding 位置设为极小值）
        attention = attention.masked_fill_(mask, -1e9) # 使用掩码填充注意力分数
        # 缩放并 softmax
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1) # 对注意力分数进行 Softmax 归一化
        # 加权求和
        attention = torch.matmul(attention,V) # 计算注意力权重和 V 的乘积
        return attention # 返回注意力输出

# =============================================================================
# Multi_CrossAttention：多头交叉注意力（LLMESR 使用）
# 
# 用于双视图模型中 ID 视图和语言视图的交互
# forward 时，第一个参数用于计算 query，第二个参数用于计算 key 和 value
# =============================================================================
class Multi_CrossAttention(nn.Module):
    """
    forward时，第一个参数用于计算query，第二个参数用于计算key和value
    """
    def __init__(self,hidden_size,all_head_size,head_num):
        """
        Args:
            hidden_size: 输入维度
            all_head_size: 输出维度（通常等于 hidden_size）
            head_num: 注意力头数
        """
        super().__init__() # 调用父类初始化
        self.hidden_size    = hidden_size       # 输入维度
        self.all_head_size  = all_head_size     # 输出维度
        self.num_heads      = head_num          # 注意头的数量
        self.h_size         = all_head_size // head_num  # 每个头的维度

        assert all_head_size % head_num == 0 # 确保 hidden_size 能被 head_num 整除

        # Q, K, V 投影层（无偏置）
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False) # Query 投影层
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False) # Key 投影层
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False) # Value 投影层
        # 输出投影层
        self.linear_output = nn.Linear(all_head_size, hidden_size) # 输出线性层

        # 归一化因子
        self.norm = sqrt(all_head_size) # 计算归一化因子


    def print(self):
        """打印调试信息"""
        print(self.hidden_size,self.all_head_size) # 打印隐藏层大小和总头大小
        print(self.linear_k,self.linear_q,self.linear_v) # 打印线性层信息
    

    def forward(self,x,y,log_seqs, pad_id):
        """
        交叉注意力前向传播
        
        Args:
            x: [B, S, D] 用于计算 Q 的输入（如 ID 视图）
            y: [B, S, D] 用于计算 K, V 的输入（如语言视图）
            log_seqs: [B, S] 序列（用于生成掩码）
            pad_id: padding ID
            
        Returns:
            output: [B, S, D] 交叉注意力输出
        """
        batch_size = x.size(0) # 获取 batch size
        # 维度变换流程：(B, S, D) -> (B, S, H, W) -> (B, H, S, W)

        # Q 来自 x，K 和 V 来自 y（这是交叉注意力的关键）
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2) # 计算 Q 并调整维度
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2) # 计算 K 并调整维度
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2) # 计算 V 并调整维度

        # 生成 padding 掩码
        attention_mask = (log_seqs == pad_id).unsqueeze(1).repeat(1, log_seqs.size(1), 1).unsqueeze(1) # 创建 attention mask

        # 计算注意力
        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask) # 计算注意力
        # 合并多头：[B, H, S, W] -> [B, S, H*W]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size) # 调整维度
        
        # 输出投影
        output = self.linear_output(attention) # 输出投影

        return output # 返回输出结果

# =============================================================================
# Contrastive_Loss2：对比学习损失（LLMESR 使用）
# 
# 用于对齐双视图模型中的 ID 视图和语言视图
# 核心思想：让相似的样本在两个视图中都相似
# =============================================================================
class Contrastive_Loss2(nn.Module):

    def __init__(self, tau=1) -> None:
        """
        Args:
            tau: 温度参数（控制分布锐度）
        """
        super().__init__() # 调用父类初始化
        self.temperature = tau # 设置温度参数


    def forward(self, X, Y):
        """
        计算对比损失
        
        Args:
            X: [B, D] 视图1的表示（如 ID 视图）
            Y: [B, D] 视图2的表示（如语言视图）
            
        Returns:
            loss: 标量损失值
        """
        # 计算跨视图相似度矩阵
        logits = (X @ Y.T) / self.temperature  # [B, B] 计算 logits
        
        # 计算软标签（基于视图内相似度）
        X_similarity = Y @ Y.T  # Y 视图内相似度
        Y_similarity = X @ X.T  # X 视图内相似度
        # 软标签 = 两个视图内相似度的平均
        targets = F.softmax(
            (X_similarity + Y_similarity) / 2 * self.temperature, dim=-1 
        ) # 计算目标概率分布
        
        # 双向对比损失
        X_loss = self.cross_entropy(logits, targets, reduction='none') # 计算 X 方向的损失
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction='none') # 计算 Y 方向的损失
        loss =  (Y_loss + X_loss) / 2.0  # 取平均
        return loss.mean() # 返回平均损失
    

    def cross_entropy(self, preds, targets, reduction='none'):
        """
        软标签交叉熵损失
        
        Args:
            preds: 预测 logits
            targets: 软标签（概率分布）
        """
        log_softmax = nn.LogSoftmax(dim=-1) # LogSoftmax
        loss = (-targets * log_softmax(preds)).sum(1) # 计算交叉熵损失
        if reduction == "none": # 如果不需要 reduction
            return loss # 返回原始损失
        elif reduction == "mean": # 如果需要 mean reduction
            return loss.mean() # 返回平均损失

# =============================================================================
# SinusoidalPositionEmbeddings：正弦位置嵌入
# 
# 用于扩散模型中的时间步编码
# 公式：PE(pos, 2i) = sin(pos / 10000^(2i/d))
#       PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
# =============================================================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        """
        Args:
            dim: 嵌入维度
        """
        super().__init__() # 调用父类初始化
        self.dim = dim # 嵌入维度

    def forward(self, time):
        """
        计算时间步的正弦位置嵌入
        
        Args:
            time: [B] 时间步（标量）
            
        Returns:
            embeddings: [B, dim] 位置嵌入
        """
        device = time.device # 获取设备
        half_dim = self.dim // 2 # 计算一半维度
        # 计算频率指数
        embeddings = math.log(10000) / (half_dim - 1) # 计算频率
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) # 计算 exp 频率
        # 计算位置编码
        embeddings = time[:, None] * embeddings[None, :] # 扩展维度
        # 拼接 sin 和 cos
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # 连接 sin 和 cos
        return embeddings # 返回嵌入
    

# =============================================================================
# TimestepEmbedder：时间步嵌入器（DreamRec/扩散模型使用）
# 
# 将标量时间步转换为向量表示
# 流程：时间步 -> 正弦编码 -> MLP -> 时间步嵌入
# =============================================================================
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        """
        Args:
            hidden_size: 输出嵌入维度
            frequency_embedding_size: 正弦编码维度（中间维度）
        """
        super().__init__() # 调用父类初始化
        # MLP：正弦编码 -> 时间步嵌入
        self.mlp = nn.Sequential(
            #nn.Linear(frequency_embedding_size, 2*hidden_size, bias=True),
            nn.Linear(frequency_embedding_size, hidden_size), # 线性层
            #nn.SiLU(),
            nn.GELU(),  # GELU 激活
            nn.Linear(hidden_size, hidden_size), # 线性层
            #nn.Linear(2*hidden_size, hidden_size),
        ) # MLP 网络
        self.frequency_embedding_size = frequency_embedding_size # 频率嵌入大小

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        创建正弦时间步嵌入
        
        这是扩散模型中常用的时间步编码方式，
        参考 OpenAI GLIDE 实现
        
        :param t: [N] 时间步索引（可以是小数）
        :param dim: 输出维度
        :param max_period: 控制最小频率
        :return: [N, dim] 位置嵌入
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        #embeddings = math.log(10000) / (half_dim - 1)
        #embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        #embeddings = time[:, None] * embeddings[None, :]
        #embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        half = dim // 2 # 计算一半维度
        # 计算频率
        freqs = math.log(10000) / (half - 1) # 计算频率
        freqs = torch.exp(torch.arange(half, device=t.device) * -freqs) # 计算 exp 频率
        #freqs = torch.exp(
        #    -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        #).to(device=t.device)
        # 计算参数
        args = t[:, None].float() * freqs[None, :] # 扩展维度
        # 拼接 sin 和 cos
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) # 连接 sin 和 cos
        #embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # 如果维度是奇数，补零
        if dim % 2: # 如果维度是奇数
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1) # 补零
        return embedding # 返回嵌入

    def forward(self, t):
        """
        前向传播
        
        Args:
            t: [B] 时间步
            
        Returns:
            t_emb: [B, hidden_size] 时间步嵌入
        """
        # 正弦编码
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size) # 计算时间步嵌入
        # MLP 变换
        t_emb = self.mlp(t_freq) # 通过 MLP
        return t_emb # 返回结果