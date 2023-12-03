#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/21 11:32
# @Author  : Liangliang
# @File    : exection.py
# @Software: PyCharm

import argparse
import time
import gibcl

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--lr", help="学习率", type=float, default=0.0001)
    parser.add_argument("--n_heads", help="多头自注意力机制的头数", type=int, default=3)
    parser.add_argument("--stop_num", help="执行Early Stopping的最低epoch", type=int, default=10)
    parser.add_argument("--k_max", help="最大采样子图次数", type=int, default=2)
    parser.add_argument("--epoch", help="迭代次数", type=int, default=2)
    parser.add_argument("--mul", help="互信息loss的权重", type=float, default=1.0)
    parser.add_argument("--tau", help="在数据增广过程中的temperature参数", type=float, default=1.0)
    parser.add_argument("--hard_rate", help="困难负样本中的正样本信息比例", type=float, default=0.2)
    parser.add_argument("--neg_num", help="负采样的负样本数目", type=int, default=3)
    parser.add_argument("--epoch_during", help="打印训练loss信息的周期", type=int, default=1)
    parser.add_argument("--input_dim", help="输入特征的维度", type=int, default=14)
    parser.add_argument("--output_dim", help="输出特征的维度", type=int, default=64)
    parser.add_argument("--sub_graph_nodes_max", help="采样子图最大的节点数目", type=int, default=20000)
    parser.add_argument("--sub_graph_edges_max", help="采样子图最大的边数目", type=int, default=200000)
    parser.add_argument("--sub_graph_nodes_min", help="采样子图最小的节点数目", type=int, default=100)
    parser.add_argument("--sub_graph_edges_min", help="采样子图最小的边数目", type=int, default=120)
    parser.add_argument("--k_hop", help="采样子图的跳连数目", type=int, default=2)
    parser.add_argument("--file_nodes_max_num", help="单个csv文件中写入数据的最大行数", type=int, default=150000)
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--model_output", help="模型的输出位置",
                        type=str, default='s3://models/gclmec/')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    gibcl.train(args)
    end_time = time.time()
    print("算法总共耗时:{}".format(end_time - start_time))