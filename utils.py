import os
import copy
import math
import numpy as np
import pandas as pd
from collections import deque
import torch.nn as nn
import torch
import torch.nn.functional as F

def calculate_hit_loader(sorted_list,topk,true_items,hit_purchase,ndcg_purchase,mrr_purchase):
    true_items = true_items.tolist()
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        # print(rec_list)
        # print(true_items)
        # print('...........')
        # break
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])[0,0]
                # total_reward[i] += rewards[j]
                # if rewards[j] == r_click:
                #     hit_click[i] += 1.0
                #     ndcg_click[i] += 1.0 / np.log2(rank + 1)
                # else:
                hit_purchase[i] += 1.0
                ndcg_purchase[i] += 1.0 / np.log2(rank + 1)
                mrr_purchase[i] += 1.0/ rank
    
# =============================================================================
# 扩散模型（DreamRec）评估函数
# 
# 功能：在验证集/测试集上评估扩散推荐模型的性能
# 
# 评估指标：
#   - HR@K (Hit Rate): 命中率，Top-K 推荐列表中包含真实物品的比例
#   - NDCG@K (Normalized Discounted Cumulative Gain): 归一化折损累积增益，考虑排名位置
#   - MRR@K (Mean Reciprocal Rank): 平均倒数排名，第一个正确推荐的排名倒数
# 
# 参数：
#   model: 待评估的推荐模型
#   diff: 扩散模型对象（用于生成预测）
#   val_loader: 验证/测试数据的 DataLoader
#   device: 计算设备（CPU/GPU）
# 
# 返回：
#   ndcg_list[3]: NDCG@20 的值（用于早停判断）
# =============================================================================
def evaluate_diff(model, diff, val_loader, device):

    # ---------------------- 初始化评估指标累计器 ----------------------
    total_purchase = 0.0                  # 总样本数（累计所有 batch 的样本数）
    hit_purchase=[0,0,0,0,0]              # HR 累计器，分别对应 topk=[1,5,10,20,50] 五个 K 值
    ndcg_purchase=[0,0,0,0,0]             # NDCG 累计器，分别对应 topk=[1,5,10,20,50]
    mrr_purchase = [0,0,0,0,0]            # MRR 累计器，分别对应 topk=[1,5,10,20,50]
    topk = [1,5,10,20,50]                 # 评估的 K 值列表

    # ---------------------- 遍历验证集的每个 batch ----------------------
    for batch in val_loader:              # 迭代 DataLoader 中的每个 batch
        seq = batch["seq"].to(device)     # 获取历史交互序列，并移动到指定设备，shape: (batch_size, seq_len)
        target = batch["next"]            # 获取真实的下一个物品 ID，shape: (batch_size,)，保持在 CPU

        #_, prediction = model.predict(seq, diff)  # 旧版本接口（已注释）
        _, prediction = model.predict(seq,  diff) # 调用模型预测函数，传入序列和扩散模型，返回所有物品的得分 args.linespace
        #print(prediction.shape)                   # 调试语句（已注释）：打印预测得分的形状
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)  # 取 Top-100 的物品索引（largest=True 取最大值，sorted=True 降序排列）
        topK = topK.cpu().detach().numpy()        # 将 topK 索引转移到 CPU，分离计算图，转换为 numpy 数组
        sorted_list2=np.flip(topK,axis=1)         # 沿 axis=1 翻转数组，使最大值在最后（方便用 [:, -K:] 取 Top-K）
        calculate_hit_loader(sorted_list2,topk,target,hit_purchase,ndcg_purchase,mrr_purchase)  # 计算当前 batch 的 HR/NDCG/MRR，累加到全局计数器
        total_purchase+=len(seq)                  # 累加当前 batch 的样本数到总样本数

    # ---------------------- 计算最终的评估指标 ----------------------
    hr_list = []                                  # 存储各 K 值的 HR（命中率）
    ndcg_list = []                                # 存储各 K 值的 NDCG（归一化折损累积增益）
    mrr_list = []                                 # 存储各 K 值的 MRR（平均倒数排名）
    for i in range(len(topk)):                    # 遍历每个 K 值
        hr_purchase=hit_purchase[i]/total_purchase    # HR@K = 命中次数 / 总样本数
        ng_purchase=ndcg_purchase[i]/total_purchase   # NDCG@K = NDCG 累计值 / 总样本数
        mr_purchase=mrr_purchase[i]/total_purchase    # MRR@K = MRR 累计值 / 总样本数
        hr_list.append(hr_purchase)               # 将 HR@K 添加到列表
        ndcg_list.append(ng_purchase)             # 将 NDCG@K 添加到列表
        mrr_list.append(mr_purchase)              # 将 MRR@K 添加到列表
    
    # ---------------------- 打印评估结果 ----------------------
    print('{:<10s} {:<10s} {:<10s} '.format("ACC","ACC","ACC"))  # 打印表头：ACC（准确率），实际是 HR@1, NDCG@1, MRR@1
    print('{:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), mrr_list[0]))  # 打印 @1 的三个指标值
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@'+str(topk[1]), 'HR@'+str(topk[2]), 'HR@'+str(topk[3]), 'HR@'+str(topk[4])))  # 打印 HR@5, HR@10, HR@20, HR@50 表头
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[1], (hr_list[2]), hr_list[3], hr_list[4]))  # 打印 HR@5, HR@10, HR@20, HR@50 的值
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('NDCG@'+str(topk[1]), 'NDCG@'+str(topk[2]), 'NDCG@'+str(topk[3]), 'NDCG@'+str(topk[4])))  # 打印 NDCG@5, NDCG@10, NDCG@20, NDCG@50 表头
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(ndcg_list[1], (ndcg_list[2]), ndcg_list[3], ndcg_list[4]))  # 打印 NDCG@5, NDCG@10, NDCG@20, NDCG@50 的值
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('MRR@'+str(topk[1]), 'MRR@'+str(topk[2]), 'MRR@'+str(topk[3]), 'MRR@'+str(topk[4])))  # 打印 MRR@5, MRR@10, MRR@20, MRR@50 表头
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(mrr_list[1], (mrr_list[2]), mrr_list[3], mrr_list[4]))  # 打印 MRR@5, MRR@10, MRR@20, MRR@50 的值

    return ndcg_list[3]  # 返回 NDCG@20，用于早停判断（topk[3] = 20）


# =============================================================================
# 模型评估函数
# 
# 功能：在验证集/测试集上评估推荐模型的性能
# 
# 评估指标：
#   - HR@K (Hit Rate): 命中率，Top-K 推荐列表中包含真实物品的比例
#   - NDCG@K (Normalized Discounted Cumulative Gain): 归一化折损累积增益，考虑排名位置
#   - MRR@K (Mean Reciprocal Rank): 平均倒数排名，第一个正确推荐的排名倒数
# 
# 参数：
#   model: 待评估的推荐模型
#   val_loader: 验证/测试数据的 DataLoader
#   device: 计算设备（CPU/GPU）
#   emb_type: 嵌入类型（默认 "both"，当前未使用）
# 
# 返回：
#   ndcg_list[3]: NDCG@20 的值（用于早停判断）
# =============================================================================
def evaluate(model, val_loader, device, rating_matrix=None, emb_type="both"):
    """
    模型评估函数
    
    Args:
        model: 待评估的推荐模型
        val_loader: 验证/测试数据的 DataLoader
        device: 计算设备（CPU/GPU）
        rating_matrix: 可选的稀疏屏蔽矩阵 [num_users, num_items]（BSARec 风格）
                       如果提供，将屏蔽用户历史交互物品
        emb_type: 嵌入类型（默认 "both"，当前未使用）
    
    Returns:
        ndcg_list[3]: NDCG@20 的值（用于早停判断）
    """

    # ---------------------- 初始化评估指标累计器 ----------------------
    total_purchase = 0.0          # 总样本数（累计所有 batch 的样本）
    hit_purchase=[0,0,0,0,0]      # HR 累计器，对应 topk=[1,5,10,20,50] 五个 K 值
    ndcg_purchase=[0,0,0,0,0]     # NDCG 累计器
    mrr_purchase = [0,0,0,0,0]    # MRR 累计器
    topk = [1,5,10,20,50]         # 评估的 K 值列表

    # ---------------------- 遍历验证集的每个 batch ----------------------
    for batch in val_loader:
        seq = batch["seq"].to(device)  # 历史交互序列，shape: (batch_size, seq_len)
        len_seq = batch["len_seq"]  # 序列真实长度（当前未使用）
        target = batch["next"]  # 真实的下一个物品 ID，shape: (batch_size,)，保持在 CPU
        
        # ==================== [NEW] 获取 user_id（用于屏蔽矩阵）====================
        user_ids = batch.get("user_id", None)
        # ==================== [END NEW] ====================

        prediction = model.predict(seq)  # 模型预测，返回所有物品的得分，shape: (batch_size, item_num)
        
        # ==================== [NEW] 屏蔽历史物品（BSARec 风格）====================
        # 将用户已交互物品的分数置为 -inf，确保不会进入 Top-K
        if rating_matrix is not None and user_ids is not None:
            prediction_np = prediction.detach().cpu().numpy().copy()
            batch_user_index = user_ids.numpy()
            
            try:
                # 核心屏蔽逻辑：与 BSARec trainers.py 完全一致
                prediction_np[rating_matrix[batch_user_index].toarray() > 0] = -np.inf
            except IndexError:
                # BERT4Rec 等模型可能有额外的 token 列（如 [MASK]），导致维度不匹配
                # rating_matrix 列数是 V，而 prediction 列数可能是 V+1 或 V+2
                # 去掉多余的列，让维度对齐
                n_cols_matrix = rating_matrix.shape[1]
                prediction_np = prediction_np[:, :n_cols_matrix]
                prediction_np[rating_matrix[batch_user_index].toarray() > 0] = -np.inf
            
            prediction = torch.from_numpy(prediction_np).to(device)
        # ==================== [END NEW] ====================
        
        # 取 Top-100 的物品索引（足够覆盖 topk=[1,5,10,20,50]）
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)  # shape: (batch_size, 100)

        
        topK = topK.cpu().detach().numpy()  # 转移到 CPU 并转换为 numpy 数组
        
        # np.flip: 沿 axis=1 翻转数组
        # 原因：topk 返回的是降序（最大在前），flip 后变成升序（最大在后）
        # 这样 sorted_list2[:, -K:] 就是 Top-K 推荐列表
        sorted_list2=np.flip(topK,axis=1)  # shape: (batch_size, 100)
        
        # 计算当前 batch 的 HR/NDCG/MRR，累加到全局计数器
        calculate_hit_loader(sorted_list2,topk,target,hit_purchase,ndcg_purchase,mrr_purchase)
        
        total_purchase+=len(seq)  # 累加当前 batch 的样本数

    # ---------------------- 计算最终的评估指标 ----------------------
    hr_list = []  # 存储各 K 值的 HR
    ndcg_list = []  # 存储各 K 值的 NDCG
    mrr_list = []  # 存储各 K 值的 MRR
    
    for i in range(len(topk)):
        # 除以总样本数，得到平均指标
        hr_purchase=hit_purchase[i]/total_purchase  # HR@K = 命中次数 / 总样本数
        ng_purchase=ndcg_purchase[i]/total_purchase  # NDCG@K = NDCG 累计 / 总样本数
        mr_purchase=mrr_purchase[i]/total_purchase  # MRR@K = MRR 累计 / 总样本数
        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase)
        mrr_list.append(mr_purchase)
    
    # ---------------------- 打印评估结果 ----------------------
    # 第一行：ACC（准确率），这里实际上是 HR@1, NDCG@1, MRR@1
    print('{:<10s} {:<10s} {:<10s} '.format("ACC","ACC","ACC"))
    print('{:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), mrr_list[0]))
    
    # HR@5, HR@10, HR@20, HR@50
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@'+str(topk[1]), 'HR@'+str(topk[2]), 'HR@'+str(topk[3]), 'HR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[1], (hr_list[2]), hr_list[3], hr_list[4]))
    
    # NDCG@5, NDCG@10, NDCG@20, NDCG@50
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('NDCG@'+str(topk[1]), 'NDCG@'+str(topk[2]), 'NDCG@'+str(topk[3]), 'NDCG@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(ndcg_list[1], (ndcg_list[2]), ndcg_list[3], ndcg_list[4]))
    
    # MRR@5, MRR@10, MRR@20, MRR@50
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('MRR@'+str(topk[1]), 'MRR@'+str(topk[2]), 'MRR@'+str(topk[3]), 'MRR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(mrr_list[1], (mrr_list[2]), mrr_list[3], mrr_list[4]))

    # 返回 NDCG@20，用于早停判断（topk[3] = 20）
    return ndcg_list[3]