#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/21 11:37
# @Author  : liangliang
# @File    : gibcl.py
# @Software: PyCharm

import os
import datetime
import time
import math
import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import s3fs
import concurrent.futures

os.system("pip install dgl dglgo -f https://data.dgl.ai/wheels/repo.html")
os.environ['DGLBACKEND'] = "tensorflow"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import dgl
from dgl import nn as nn

#防止除数为0
e = 1e-5
g = ""

# 设置随机种子点
'''
random.seed(921208)
np.random.seed(921208)
tf.random.set_seed(921208)
os.environ['PYTHONHASHSEED'] = "921208"
# 设置GPU随机种子点
os.environ['TF_DETERMINISTIC_OPS'] = '1'
'''


#读取文件系统
class S3FileSystemPatched(s3fs.S3FileSystem):
    def __init__(self, *k, **kw):
        super(S3FileSystemPatched, self).__init__(*k,
                                                  key=os.environ['AWS_ACCESS_KEY_ID'],
                                                  secret=os.environ['AWS_SECRET_ACCESS_KEY'],
                                                  client_kwargs={'endpoint_url': 'http://' + os.environ['S3_ENDPOINT']},
                                                  **kw
                                                  )


class GIBnet(keras.Model):
    def __init__(self, input_dim, output_dim=64, num_heads=3):
        super(GIBnet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()
        self.cov1 = nn.GATConv(self.input_dim, self.output_dim, num_heads=self.num_heads)
        #self.cov2 = nn.GATConv(self.input_dim, self.output_dim, num_heads=self.num_heads, allow_zero_in_degree=True)
        self.layer1 = keras.layers.Dense(self.output_dim)
        self.layer2 = keras.layers.Dense(self.output_dim)
        self.lightgcn1 = nn.GraphConv(self.output_dim, self.output_dim, weight=False, bias=False)
        self.lightgcn2 = nn.GraphConv(self.output_dim, self.output_dim, weight=False, bias=False)
        self.lightgcn_low1 = nn.GraphConv(self.output_dim, self.output_dim, weight=False, bias=False)
        self.lightgcn_low2 = nn.GraphConv(self.output_dim, self.output_dim, weight=False, bias=False)
        self.last_layer = keras.layers.Dense(self.output_dim)

    def call(self, g, args, training=False, mask=None):
        g = dgl.add_self_loop(g)
        g = dgl.to_simple(g)
        inputs = g.ndata["feat"]
        inputs = self.cov1(g, inputs)
        inputs = tf.nn.leaky_relu(inputs)

        #把三维张量变为二维张量
        inputs = tf.reduce_sum(inputs, axis=1)
        input1 = inputs
        input2 = inputs

        if training == True:
            pro1 = self.layer1(input1)
            pro1 = tf.nn.sigmoid(pro1)

            #reparameterization trick
            r = tf.random.uniform((pro1.shape[0], self.output_dim))
            pro1 = tf.sigmoid((tf.math.log(r + e) - tf.math.log(1 - r + e) + pro1) / args.tau)

            #对数据进行增广获得视角1
            input1 = input1 * pro1


            pro2 = self.layer2(input2)
            pro2 = tf.nn.sigmoid(pro2)
            #reparameterization trick
            r = tf.random.uniform((pro2.shape[0], self.output_dim))
            pro2 = tf.sigmoid((tf.math.log(r + e) - tf.math.log(1 - r + e) + pro2) / args.tau)

            # 对数据进行增广获得视角2
            input2 = input2 * pro2

        #对视角1数据进行归一化
        input1 = self.batch_normalization1(input1, training=training)

        # 对视角2数据进行归一化
        input2 = self.batch_normalization2(input2, training=training)

        h_t1 = self.lightgcn1(g, input1)
        h_t2 = self.lightgcn2(g, h_t1)
        h_t = (h_t1 + h_t2 + input1) / 3
        h_t = self.last_layer(h_t)
        h_t = tf.nn.leaky_relu(h_t)
        h_t = tf.nn.l2_normalize(h_t, axis=1)

        h_s1 = self.lightgcn_low1(g, input2)
        h_s2 = self.lightgcn_low2(g, h_s1)
        h_s = (h_s1 + h_s2 + input2) / 3
        h_s = tf.nn.l2_normalize(h_s, axis=1)

        inputs = tf.nn.l2_normalize(inputs, axis=1)
        return [h_t, h_s, inputs]


def loss_function_mutual_infor(E, E1, E2, args):
    '''
    评估E与E1的互信息
    互信息计算方法参考: Wei C, Liang J, Liu D, et al. Contrastive Graph Structure Learning via Information Bottleneck
    for Recommendation[C]//Advances in Neural Information Processing Systems, NeurIPS 2022.
    '''
    E_t = tf.repeat(E, args.neg_num, axis=0)
    E_t = tf.reshape(E_t, [E.shape[0], args.neg_num, E.shape[1]])
    neg_samples = tf.gather(E1, np.random.randint(0, E1.shape[0], args.neg_num * E.shape[0]), axis=0)
    neg_samples = tf.reshape(neg_samples, [E.shape[0], args.neg_num, E.shape[1]])
    neg_samples_hard = (1 - args.hard_rate) * E_t + args.hard_rate * neg_samples
    loss1 = tf.reduce_sum(tf.math.log((tf.reduce_sum(tf.math.exp(E * E1), axis=1) / args.tau) /
                                      (tf.reduce_sum(tf.reduce_sum(tf.math.exp(E_t * neg_samples), axis=2), axis=1) /
                                       args.tau + tf.reduce_sum(tf.reduce_sum(tf.math.exp(E_t * neg_samples_hard),
                                                                              axis=2), axis=1) / args.tau)))

    #评估E与E2的互信息
    neg_samples = tf.gather(E2, np.random.randint(0, E1.shape[0], args.neg_num * E.shape[0]), axis=0)
    neg_samples = tf.reshape(neg_samples, [E.shape[0], args.neg_num, E.shape[1]])
    neg_samples_hard = (1 - args.hard_rate) * E_t + args.hard_rate * neg_samples
    
    loss2 = tf.reduce_sum(tf.math.log((tf.reduce_sum(tf.math.exp(E * E2), axis=1) / args.tau) / (
                tf.reduce_sum(tf.reduce_sum(tf.math.exp(E_t * neg_samples), axis=2), axis=1) / args.tau + tf.reduce_sum(
            tf.reduce_sum(tf.math.exp(E_t * neg_samples_hard), axis=2), axis=1) / args.tau)))

    loss = (loss1 + loss2) / (E.shape[0]*2*args.neg_num)
    return loss


def read_graph(args):
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    count = 0
    print("开始读取数据! {}".format(datetime.datetime.now()))
    data = pd.DataFrame()
    for file in input_files:
        count = count + 1
        print("当前正在处理第{}个文件,文件路径:{}......".format(count, "s3://" + file))
        # 读取边结构数据
        data = pd.concat([data, pd.read_csv("s3://" + file, sep=',', header=None)], axis=0)
    # 读取属性特征信息
    # 最后一列为节点的类型，所以特征的列数为n-1
    path = args.data_input.split(',')[1]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    data_attr = pd.DataFrame()
    for file in input_files:
        # 读取属性特征数据
        data_attr = pd.concat([data_attr, pd.read_csv("s3://" + file, sep=',', header=None)], axis=0)
    # 读取节点的属性特征数据
    data_attr = tf.convert_to_tensor(data_attr.values, dtype=tf.float32)
    # 定义图结构
    g = dgl.graph((data.iloc[:, 0].to_list(), data.iloc[:, 1].to_list()), num_nodes=data_attr.shape[0],
                  idtype=tf.int32)
    # 转化为无向图
    g = dgl.to_bidirected(g)
    g = dgl.add_self_loop(g)
    g = dgl.to_simple(g)
    g.ndata["feat"] = data_attr
    return g

#采样子图进行训练
def get_subgraph(i, args):
    #计算第i个节点所在子图的embedding向量
    if g.out_degrees(i) > args.sub_graph_nodes_max:
        sub_nodes = [i] + list(g.successors(i).numpy()[0:args.sub_graph_nodes_max + 1])
    else:
        sub_nodes = [i] + list(g.successors(i).numpy())
    sub_nodes = np.array(list(set(sub_nodes)))
    # 取度数最大的前args.sub_graph_nodes_max个节点构成子图
    g_sub = dgl.node_subgraph(g, sub_nodes)
    g_sub = dgl.to_bidirected(g_sub, copy_ndata=True)
    g_sub = dgl.add_self_loop(g_sub)
    g_sub = dgl.to_simple(g_sub)

    if g_sub.number_of_edges() > args.sub_graph_edges_max:
        # 获取当前节点的后继节点
        sub_nodes = [i] + list(g.successors(i).numpy())
        sub_nodes = np.array(list(set(sub_nodes)))
        #边数太大,要大范围截断,以减轻子图抽样过程耗时
        if g_sub.number_of_edges() > 0 and g_sub.number_of_edges() / args.sub_graph_edges_max > 1.2:
            j = max(int((len(sub_nodes) - 1) / 5), 1)
        else:
            j = max(int((len(sub_nodes) - 1) / 2), 1)
        increasenum = 1
        while j > 0:
            j = j - increasenum
            increasenum = increasenum + 1
            if sub_nodes[j] == i:
                j = j - 1
            sub_nodes_temp = sub_nodes[tf.argsort(g.in_degrees(sub_nodes), direction="DESCENDING").numpy()[0:max(j,1)]]
            sub_nodes_temp = [i] + list(sub_nodes_temp)
            sub_nodes_temp = np.array(list(set(sub_nodes_temp)))
            g_sub = dgl.node_subgraph(g, sub_nodes_temp)
            g_sub = dgl.add_self_loop(g_sub)
            g_sub = dgl.to_bidirected(g_sub, copy_ndata=True)
            g_sub = dgl.to_simple(g_sub)
            if g_sub.number_of_edges() <= args.sub_graph_edges_max:
                break
        #只剩下度数最小的节点仍不满足要求
        if g_sub.number_of_edges() > args.sub_graph_edges_max:
            g_sub = ""
    if i < 0:
        print("第{}个节点原子图边数过大,重新采样子图节点数为:{} 边数为{} {}".format(i, g_sub.number_of_nodes(),
                                                            g_sub.number_of_edges(), datetime.datetime.now()))
            
    if g_sub.number_of_nodes() < args.sub_graph_nodes_min or g_sub.number_of_edges() < args.sub_graph_edges_min:
         print("第{}个节点采样子图失败,图的规模太小! 节点数:{} 边数:{} {}".format(i, g_sub.number_of_nodes(),
                                                                g_sub.number_of_edges(), datetime.datetime.now()))
         g_sub = ""
    return g_sub


def write(data, count, args):
    #注意在此业务中data是一个二维list
    #数据的数量
    n = len(data)
    if n > 0:
        start = time.time()
        with open(os.path.join(args.data_output, 'pred_{}.csv'.format(count)), mode="a") as resultfile:
            for i in range(n):
                # 说明此时的data是[[],[],...]的二级list形式
                line = ",".join(map(str, data[i])) + "\n"
                resultfile.write(line)
        cost = time.time() - start
        print("第{}个大数据文件已经写入完成,写入数据的行数{} 耗时:{}  {}".format(count, n, cost, datetime.datetime.now()))


def train(args):
    '''
    该算法上游包含三个tdw表输入
    第一个tdw表:图的边 含有两列分别为左右端点,不一定是双向图,使用双向图需要进行转化
    第二个tdw表:图节点的属性特征,每一行为一个节点的属性特征
    第三个tdw表:计算bpr loss的三元组pair
    '''
    #第一步读取图数据
    global g
    g = read_graph(args)
    #读取训练数据
    path = args.data_input.split(',')[2]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    count = 0
    print("开始读取数据! {}".format(datetime.datetime.now()))

    #读取bpr pair
    data = pd.DataFrame()
    for file in input_files:
        count = count + 1
        print("当前正在处理pair文件的第{}个文件,文件路径:{}......".format(count, "s3://" + file))
        # 读取边结构数据
        data = pd.concat([data, pd.read_csv("s3://" + file, sep=',', header=None)], axis=0).astype('int64')

    #初始化图神经网络模型
    if args.env == "train":
        model = GIBnet(g.ndata["feat"].shape[1], args.output_dim, args.n_heads)
    else:
        #装载之前已训练的模型
        model = GIBnet(g.ndata["feat"].shape[1], args.output_dim, args.n_heads)
        cmd = "s3cmd get -r  " + args.model_output
        os.system(cmd)
        checkpoint_path = "./gibclmodel/gibcl.pb"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)
        print("The weights of GIBCL model are load!")
    optimizer = keras.optimizers.Adagrad(learning_rate=args.lr)
    #图节点的数目
    N = g.number_of_nodes()
    BEFORE_LOSS = 2**23
    stop_num = 0
    loss = 0
    for epoch in range(args.epoch):
        #依据局部embedding无监督更新网络参数
        loss_mutual = 0
        for num in range(args.subgraph_num):
            g_sub = get_subgraph(random.randint(0, N - 1), args)
            if g_sub != "":
                with tf.GradientTape() as tape:
                    if num % 3000 == 0:
                        print("第{}个epoch的第{}节点互信息子图的节点数目为:{} 边的数目为:{}! {}".format(epoch, num,
                                                                                  g_sub.number_of_nodes(),
                                                                                  g_sub.number_of_edges(),
                                                                                  datetime.datetime.now()))
                    h_t, h_s, h_origin = model(g_sub, args, training=True)
                    loss_mutinfor = loss_function_mutual_infor(h_origin, h_t, h_s, args)
                    loss_mutual = loss_mutual + loss_mutinfor / args.subgraph_num
                gradients = tape.gradient(loss_mutinfor, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                if num % 1000 == 0:
                    print("已完成第{}个epoch的第{}节点的loss为:{} {}".format(epoch, num, loss_mutinfor,
                                                                  datetime.datetime.now()))
            del g_sub
        #依据有监督信息更新网络参数
        loss_bpr = 0
        del gradients
        n = data.shape[0]
        for i in range(n):
            with tf.GradientTape() as tape_all:
                embed1 = tf.zeros((1, args.output_dim))[0, :]
                embed2 = tf.zeros((1, args.output_dim))[0, :]
                embed3 = tf.zeros((1, args.output_dim))[0, :]
                embed1_s = tf.zeros((1, args.output_dim))[0, :]
                embed2_s = tf.zeros((1, args.output_dim))[0, :]
                embed3_s = tf.zeros((1, args.output_dim))[0, :]

                t1 = time.time()
                #用多线程的方法进行子图采样,便于节省时间
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(get_subgraph, int(data.iloc[i, s]), args) for s in range(3)]
                    g_sub1, g_sub2, g_sub3 = [f.result() for f in futures]
                t2 = time.time()
                if i % 1000 == 0:
                    print("第{}个epoch的第{}个bpr pair三个子图采样耗时为:{} {}".format(epoch, i, t2 - t1,
                                                                         datetime.datetime.now()))

                #g_sub1 = get_subgraph(int(data.iloc[i, 0]), args)
                #g_sub2 = get_subgraph(int(data.iloc[i, 1]), args)
                #g_sub3 = get_subgraph(int(data.iloc[i, 2]), args)

                if g_sub1 != "":
                    if i % 10000 == 0:
                        print("第{}个epoch的第{}个节点采样的第一个子图节点数为:{} 边数为:{} {}".format(epoch, i,
                                                                                 g_sub1.number_of_nodes(),
                                                                                 g_sub1.number_of_edges(),
                                                                                 datetime.datetime.now()))
                    embed1, embed1_s, _ = model(g_sub1, args, training=True)
                    j = tf.where(g_sub1.ndata[dgl.NID] == data.iloc[i, 0]).numpy()[0, 0]
                    embed1 = embed1[j, :]
                    embed1_s = embed1_s[j, :]
                else:
                    print("第{}个epoch第{}个节点编号:{}第一个子图采样失败! {}".format(epoch, i, int(data.iloc[i, 0]),
                                                                     datetime.datetime.now()))
                if g_sub2 != "":
                    if i % 10000 == 0:
                        print("第{}个epoch的第{}个节点采样的第二个子图节点数为:{} 边数为:{} {}".format(epoch, i,
                                                                                 g_sub2.number_of_nodes(),
                                                                                 g_sub2.number_of_edges(),
                                                                                 datetime.datetime.now()))
                    embed2, embed2_s, _ = model(g_sub2, args, training=True)
                    j = tf.where(g_sub2.ndata[dgl.NID] == data.iloc[i, 1]).numpy()[0, 0]
                    embed2 = embed2[j, :]
                    embed2_s = embed2_s[j, :]
                else:
                    print("第{}个epoch第{}个节点编号:{}第二个子图采样失败! {}".format(epoch, i, int(data.iloc[i, 1]),
                                                                     datetime.datetime.now()))
                if g_sub3 != "":
                    if i % 10000 == 0:
                        print("第{}个epoch的第{}个节点采样的第三个子图节点数为:{} 边数为:{} {}".format(epoch, i,
                                                                                 g_sub3.number_of_nodes(),
                                                                                 g_sub3.number_of_edges(),
                                                                                 datetime.datetime.now()))
                    embed3, embed3_s, _ = model(g_sub3, args, training=True)
                    j = tf.where(g_sub3.ndata[dgl.NID] == data.iloc[i, 2]).numpy()[0, 0]
                    embed3 = embed3[j, :]
                    embed3_s = embed3_s[j, :]
                else:
                    print("第{}个epoch第{}个节点编号:{}第三个子图采样失败! {}".format(epoch, i, int(data.iloc[i, 1]),
                                                                     datetime.datetime.now()))

                #基于teacher网络的bpr loss
                r_pos = tf.reduce_sum(embed1 * embed2)
                r_neg = tf.reduce_sum(embed1 * embed3)

                # 基于student网络的bpr loss
                r_pos1 = tf.reduce_sum(embed1_s * embed2_s)
                r_neg1 = tf.reduce_sum(embed1_s * embed3_s)

                loss_bpr = loss_bpr - 0.5*(tf.math.log(tf.nn.sigmoid(r_pos - r_neg)) -
                                           tf.math.log(tf.nn.sigmoid(r_pos1 - r_neg1)))

            if i % args.batch_size == 0 or i == n - 1:
                gradients1 = tape_all.gradient(loss_bpr, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients1, model.trainable_variables))
                print("bpr pair数目为{}第{}epoch第{}个batch的互信息loss:{} bpr loss:{} {}".format(n, epoch,
                                                                                            int(i/2000), loss_mutual,
                                                                               loss_bpr, datetime.datetime.now()))
                if loss_bpr < BEFORE_LOSS:
                    BEFORE_LOSS = loss_bpr
                    if epoch >= 0:
                        # 保存已训练好的图神经网络模型
                        # model_t.summary()
                        model.save_weights("./gibclmodel/gibcl.pb", save_format="tf")
                        print("gibcl net已保存!")
                        cmd = "s3cmd put -r ./gibclmodel " + args.model_output
                        os.system(cmd)
                    else:
                        print("第{}个epoch的bpr loss:{} best loss:{} 无需再次保存模型! {}".format(epoch, loss_bpr,
                                                                                       BEFORE_LOSS,
                                                                                       datetime.datetime.now()))
                loss_bpr = 0
        if epoch == args.epoch - 1:
            # 保存已训练好的图神经网络模型
            # model_t.summary()
            model.save_weights("./gibclmodel/gibcl.pb", save_format="tf")
            print("gibcl net已保存!")
            cmd = "s3cmd put -r ./gibclmodel " + args.model_output
            os.system(cmd)

#计算节点的embedding
def get_embedding(node_list, model, count, args):
    n = len(node_list)
    result = np.zeros((n, args.output_dim + 1)).astype("str")
    num = -1
    for i in node_list:
        num = num + 1
        g_sub = get_subgraph(i, args)
        if g_sub != "":
            h, _, _ = model(g_sub, args, training=False)
            j =  tf.where(g_sub.ndata[dgl.NID] == i)[0, 0]
            result[num, 0] = str(i)
            result[num, 1:] = h[j, :].numpy().astype("str")
        else:
            result[num, 0] = str(i)
    #把这些结果写入cfs文件系统中
    write(result.tolist(), count, args)



#利用已有的模型对未知数据进行推理,输出embedding vectors
def inference(args):
    # 第一步读取图数据
    global g
    g = read_graph(args)

    #装载之前已训练的模型
    model = GIBnet(g.ndata["feat"].shape[1], args.output_dim, args.n_heads)
    cmd = "s3cmd get -r  " + args.model_output
    os.system(cmd)
    checkpoint_path = "./gibclmodel/gibcl.pb"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    print("The weights of GIBCL model are load!")

    #输出每一个节点的embedding vectors
    N = g.number_of_nodes()
    node_list = []
    count = -1
    n_infer = math.ceil(N / args.file_nodes_max_num)
    for i in range(N):
        node_list.append(i)
        if i > 0 and (i % args.file_nodes_max_num == 0 or i == N -1):
            count = count + 1
            print("一共{}个节点需{}次在线推理开始第{}次在线推理! {}".format(N, n_infer - 1, count, datetime.datetime.now()))
            get_embedding(node_list, model, count, args)
            node_list = []
    print("完成{}个节点的推理过程! {}".format(N, datetime.datetime.now()))


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--env", help="运行的环境(train or test)", type=str, default='train_incremental')
    parser.add_argument("--lr", help="学习率", type=float, default=0.00001)
    parser.add_argument("--n_heads", help="多头自注意力机制的头数", type=int, default=2)
    parser.add_argument("--stop_num", help="执行Early Stopping的最低epoch", type=int, default=10)
    parser.add_argument("--k_max", help="最大采样子图次数", type=int, default=2)
    parser.add_argument("--batch_size", help="bpr loss更新梯度的周期", type=int, default=2000)
    parser.add_argument("--epoch", help="迭代次数", type=int, default=20)
    parser.add_argument("--mul", help="互信息loss的权重", type=float, default=1.0)
    parser.add_argument("--bpr", help="bpr loss的权重", type=float, default=15.0)
    parser.add_argument("--tau", help="在数据增广过程中的temperature参数", type=float, default=1.0)
    parser.add_argument("--hard_rate", help="困难负样本中的正样本信息比例", type=float, default=0.8)
    parser.add_argument("--neg_num", help="负采样的负样本数目", type=int, default=3)
    parser.add_argument("--subgraph_num", help="采用子图的数目", type=int, default=2000)
    parser.add_argument("--input_dim", help="输入特征的维度", type=int, default=14)
    parser.add_argument("--output_dim", help="输出特征的维度", type=int, default=64)
    parser.add_argument("--sub_graph_nodes_max", help="采样子图最大的节点数目", type=int, default=1000)
    parser.add_argument("--sub_graph_edges_max", help="采样子图最大的边数目", type=int, default=10000)
    parser.add_argument("--sub_graph_nodes_min", help="采样子图最小的节点数目", type=int, default=1)
    parser.add_argument("--sub_graph_edges_min", help="采样子图最小的边数目", type=int, default=1)
    parser.add_argument("--k_hop", help="采样子图的跳连数目", type=int, default=1)
    parser.add_argument("--file_nodes_max_num", help="单个csv文件中写入数据的最大行数", type=int, default=150000)
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--model_output", help="模型的输出位置",
                        type=str, default='s3://models/gibcl/')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    if args.env == "train" or args.env == "train_incremental":
        train(args)
    else:
        inference(args)
    end_time = time.time()
    print("算法总共耗时:{}".format(end_time - start_time))
