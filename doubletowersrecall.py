#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/4/17 16:47
# @Author  : Liangliang
# @File    : doubletowersrecall.py
# @Software: PyCharm

import time
import os
import argparse
import s3fs
import math
import random
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# 设置随机种子点
'''
random.seed(921208)
np.random.seed(921208)
tf.random.set_seed(921208)
os.environ['PYTHONHASHSEED'] = "921208"
# 设置GPU随机种子点
os.environ['TF_DETERMINISTIC_OPS'] = '1'
'''

#防止log中出现0
e = 1e-6


#读取文件系统
class S3FileSystemPatched(s3fs.S3FileSystem):
    def __init__(self, *k, **kw):
        super(S3FileSystemPatched, self).__init__(*k,
                                                  key=os.environ['AWS_ACCESS_KEY_ID'],
                                                  secret=os.environ['AWS_SECRET_ACCESS_KEY'],
                                                  client_kwargs={'endpoint_url': 'http://' + os.environ['S3_ENDPOINT']},
                                                  **kw
                                                  )


class BaseLayer(tf.keras.Model):
    def __init__(self, feat):
        super(BaseLayer, self).__init__()
        self.cov = tf.keras.layers.Dense(feat)

    def call(self, inputs, training=None, mask=None):
        inputs = self.cov(inputs)
        inputs = tf.nn.leaky_relu(inputs)
        return inputs


class UserDNN(tf.keras.Model):
    def __init__(self, feat1=200, feat2=100, output_dim=64):
        super(UserDNN, self).__init__()
        self.cov1 = tf.keras.layers.Dense(feat1)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.attent1 = tf.keras.layers.Attention()
        self.cov2 = tf.keras.layers.Dense(feat2)
        self.attent2 = tf.keras.layers.Attention()
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()
        self.cov3 = tf.keras.layers.Dense(output_dim)

    def call(self, inputs, training=None, mask=None):
        h = self.cov1(inputs)
        if training:
            h = self.dropout(h)
            h = self.batch_normalization1(h)
        h = tf.nn.leaky_relu(h)
        h = self.attent1([h, h])
        h = self.cov2(h)
        if training:
            h = self.batch_normalization2(h)
        h = tf.nn.leaky_relu(h)
        h = self.attent2([h, h])
        h = self.cov3(h)
        h = tf.nn.leaky_relu(h)
        h = tf.nn.l2_normalize(h, axis=1)
        return h


class ClubDNN(tf.keras.Model):
    def __init__(self, feat1=300, feat2=200, output_dim=64):
        super(ClubDNN, self).__init__()
        self.cov1 = tf.keras.layers.Dense(feat1)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.attent1 = tf.keras.layers.Attention()
        self.cov2 = tf.keras.layers.Dense(feat2)
        self.attent2 = tf.keras.layers.Attention()
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()
        self.cov3 = tf.keras.layers.Dense(output_dim)

    def call(self, inputs, training=None, mask=None):
        h = self.cov1(inputs)
        if training:
            h = self.dropout(h)
            h = self.batch_normalization1(h)
        h = tf.nn.leaky_relu(h)
        h = self.attent1([h, h])
        h = self.cov2(h)
        if training:
            h = self.batch_normalization2(h)
        h = tf.nn.leaky_relu(h)
        h = self.attent2([h, h])
        h = self.cov3(h)
        h = tf.nn.leaky_relu(h)
        h = tf.nn.l2_normalize(h, axis=1)
        return h


def loss_function(use_data, club_data, batch_label):
    n = use_data.shape[0]
    result = tf.reshape(tf.reduce_sum(use_data * club_data, axis=1)/(tf.norm(use_data, axis=1) *
                                                                     tf.norm(club_data, axis=1)), [-1, 1])
    result = (result + 1.0) / 2.0
    loss = -tf.reduce_sum(batch_label*tf.math.log(result + e) + (1 - batch_label)*tf.math.log(1 - result + e))
    loss = loss / n
    #平均每10个batch打印一次测试结果
    if random.random() <= args.prob:
        auc = roc_auc_score(batch_label.numpy(), result.numpy())
        acc = accuracy_score(batch_label.numpy(), tf.where(result <= 0.5, 0 ,1).numpy())
        print("当前的auc:{} accuracy:{} {}".format(auc, acc, datetime.datetime.now()))
    return loss


def train_step(model_layer_user, model_layer_club, model_layer, model_user, model_club, epoch):
    # 读取训练数据
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    count = 0
    print("开始读取数据! {}".format(datetime.datetime.now()))
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_all = 0
    n = len(input_files)
    for file in input_files:
        data = pd.DataFrame()
        count = count + 1
        print("epoch:{}一共{}个文件,当前正在处理第{}个文件,文件路径:{}......".format(epoch, len(input_files), count,
                                                                  "s3://" + file))
        # 读取训练数据
        data = pd.concat([data, pd.read_csv("s3://" + file, sep=',', header=None)], axis=0).astype('float32')
        label = tf.reshape(tf.convert_to_tensor(data.iloc[:, -1], dtype=tf.float32), [-1, 1])
        data = tf.convert_to_tensor(data.iloc[:, :-1], dtype=tf.float32)
        data = tf.data.Dataset.from_tensor_slices((data, label)).shuffle(100).batch(args.batch_size,
                                                                                    drop_remainder=True)
        count_batch = 0
        loss = 0
        for batch_data, batch_label in data:
            count_batch = count_batch + 1
            with tf.GradientTape(persistent=True) as tape:
                use_data = batch_data[:, 0:args.split_dim]
                club_data = batch_data[:, args.split_dim::]

                use_data = model_layer_user(use_data)
                use_data = model_layer(use_data)
                use_data = model_user(use_data, training=True)

                club_data = model_layer_club(club_data)
                club_data = model_layer(club_data)
                club_data = model_club(club_data, training=True)

                loss = loss_function(use_data, club_data, batch_label)
            grad1 = tape.gradient(loss, model_layer_user.trainable_variables)
            grad2 = tape.gradient(loss, model_layer_club.trainable_variables)
            grad3 = tape.gradient(loss, model_layer.trainable_variables)
            grad4 = tape.gradient(loss, model_user.trainable_variables)
            grad5 = tape.gradient(loss, model_club.trainable_variables)

            optimizer.apply_gradients(zip(grad1, model_layer_user.trainable_variables))
            optimizer.apply_gradients(zip(grad2, model_layer_club.trainable_variables))
            optimizer.apply_gradients(zip(grad3, model_layer.trainable_variables))
            optimizer.apply_gradients(zip(grad4, model_user.trainable_variables))
            optimizer.apply_gradients(zip(grad5, model_club.trainable_variables))
            if count_batch % args.batch_during == 0:
                print("第{}个epoch第{}文件第{}batch的loss:{} {}".format(epoch, count, count_batch, loss,
                                                                 datetime.datetime.now()))
        loss_all = (loss_all + loss) / n
    return model_layer_user, model_layer_club, model_layer, model_user, model_club, loss_all


def train():
    #定义模型
    if args.env == "train":
        model_layer_user = BaseLayer(args.feat1)
        model_layer_club = BaseLayer(args.feat1)
        model_layer = BaseLayer(args.feat2)
        model_user = UserDNN(args.feat3, args.feat4, args.output_dim)
        model_club = ClubDNN(args.feat3, args.feat4, args.output_dim)
    else:
        # 装载训练好的模型
        cmd = "s3cmd get -r  " + args.model_output + "model_layer_user"
        os.system(cmd)
        model_layer_user = tf.keras.models.load_model("./model_layer_user", custom_objects={'tf': tf}, compile=False)
        print("model_layer_user is loaded!")

        cmd = "s3cmd get -r  " + args.model_output + "model_layer_club"
        os.system(cmd)
        model_layer_club = tf.keras.models.load_model("./model_layer_club", custom_objects={'tf': tf}, compile=False)
        print("model_layer_club is loaded!")

        cmd = "s3cmd get -r  " + args.model_output + "model_layer"
        os.system(cmd)
        model_layer = tf.keras.models.load_model("./model_layer", custom_objects={'tf': tf}, compile=False)
        print("model_layer is loaded!")

        cmd = "s3cmd get -r  " + args.model_output + "model_user"
        os.system(cmd)
        model_user = tf.keras.models.load_model("./model_user", custom_objects={'tf': tf}, compile=False)
        print("model_user is loaded!")

        cmd = "s3cmd get -r  " + args.model_output + "model_club"
        os.system(cmd)
        model_club = tf.keras.models.load_model("./model_club", custom_objects={'tf': tf}, compile=False)
        print("model_club is loaded!")

    beforeLoss = 2**23
    stopNum = 0
    for epoch in range(args.epoch):
        model_layer_user, model_layer_club, model_layer, model_user, model_club, loss_all = train_step(
            model_layer_user, model_layer_club, model_layer, model_user, model_club, epoch)
        if beforeLoss > loss_all:
            beforeLoss = loss_all
            stopNum = 0
            # 保存model_layer_user
            # model_layer_user.summary()
            model_layer_user.save("./model_layer_user", save_format="tf")
            cmd = "s3cmd put -r ./model_layer_user " + args.model_output
            os.system(cmd)
            print("epoch:{} model_layer_user模型已保存! {}".format(epoch, datetime.datetime.now()))

            # 保存model_layer_club
            # model_layer_club.summary()
            model_layer_club.save("./model_layer_club", save_format="tf")
            cmd = "s3cmd put -r ./model_layer_club " + args.model_output
            os.system(cmd)
            print("epoch:{} model_user模型已保存! {}".format(epoch, datetime.datetime.now()))

            # 保存model_layer
            # model_layer.summary()
            model_layer.save("./model_layer", save_format="tf")
            cmd = "s3cmd put -r ./model_layer " + args.model_output
            os.system(cmd)
            print("epoch:{} model_layer模型已保存! {}".format(epoch, datetime.datetime.now()))

            # 保存model_user
            #model_user.summary()
            model_user.save("./model_user", save_format="tf")
            cmd = "s3cmd put -r ./model_user " + args.model_output
            os.system(cmd)
            print("epoch:{} model_user模型已保存! {}".format(epoch, datetime.datetime.now()))

            # 保存model_club
            #model_club.summary()
            model_club.save("./model_club", save_format="tf")
            cmd = "s3cmd put -r ./model_club " + args.model_output
            os.system(cmd)
            print("epoch:{} model_club模型已保存! {}".format(epoch, datetime.datetime.now()))
        else:
            stopNum = stopNum + 1
            if stopNum > args.stop_num:
                print("epoch:{} Early stop! {}".format(epoch, datetime.datetime.now()))
                break

#判断是否是二维列表
def is_2d_list(lst):
    return isinstance(lst, list) and all(isinstance(sub_lst, list) for sub_lst in lst)

#写入数据
def write(data, count, id):
    #data是一个二维列表,第一列为ID,之后的列为特征
    n = len(data)
    n_flies = math.ceil(n / args.file_max_num)
    for i in range(n_flies):
        data_temp = data[i * args.file_max_num:min((i + 1) * args.file_max_num, n)]
        flag = is_2d_list(data)
        with open(os.path.join(args.data_output, 'pred_{}_{}_{}.csv'.format(count, id, i)), mode="a") as resultfile:
            # 说明此时的data是[[],[],...]的二级list形式
            if flag:
                n_data = len(data_temp)
                for j in range(n_data):
                    line = ",".join(map(str, data[j])) + "\n"
                    resultfile.write(line)
            else:
                # 说明此时的data是[x,x,...]的list形式
                line = ",".join(map(str, data)) + "\n"
                resultfile.write(line)
        print("第{}个大数据文件的第{}个子文件的第{}次子文件已经写入完成,写入数据的行数{} {}".format(count, id, i, len(data_temp),
                                                            datetime.datetime.now()))


#执行推理过程
def inference():
    # 读取训练好的模型
    model_user = ""
    model_club = ""
    cmd = "s3cmd get -r  " + args.model_output + "model_layer"
    os.system(cmd)
    model_layer = tf.keras.models.load_model("./model_layer", custom_objects={'tf': tf}, compile=False)
    print("model_layer is loaded!")
    if args.env == "inference_user":
        cmd = "s3cmd get -r  " + args.model_output + "model_layer_user"
        os.system(cmd)
        model_layer_user = tf.keras.models.load_model("./model_layer_user", custom_objects={'tf': tf}, compile=False)
        print("model_layer_user is loaded!")

        cmd = "s3cmd get -r  " + args.model_output + "model_user"
        os.system(cmd)
        model_user = tf.keras.models.load_model("./model_user", custom_objects={'tf': tf}, compile=False)
        print("model_user is loaded!")
    if args.env == "inference_club":
        cmd = "s3cmd get -r  " + args.model_output + "model_layer_club"
        os.system(cmd)
        model_layer_club = tf.keras.models.load_model("./model_layer_club", custom_objects={'tf': tf}, compile=False)
        print("model_layer_club is loaded!")

        cmd = "s3cmd get -r  " + args.model_output + "model_club"
        os.system(cmd)
        model_club = tf.keras.models.load_model("./model_club", custom_objects={'tf': tf}, compile=False)
        print("model_club is loaded!")
    #读取数据
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    count = 0
    for file in input_files:
        data = pd.DataFrame()
        count = count + 1
        print("一共{}个文件,当前正在处理第{}个文件,文件路径:{}......".format(len(input_files), count, "s3://" + file))
        # 读取训练数据
        # data的第一列为ID,第二列之后为特征
        data = pd.concat([data, pd.read_csv("s3://" + file, sep=',', header=None)], axis=0).astype('str')
        #数据的实际特征为dim-1维,有一维为ID
        N, dim = data.shape
        n = math.ceil(N / args.file_max_num)
        for i in range(n):
            data_temp = tf.convert_to_tensor(data.iloc[i*args.file_max_num:min((i+1)*args.file_max_num, N), 1:].values,
                                             dtype=tf.float32)
            #防止出现一维tensorflow
            data_temp = tf.reshape(data_temp, [-1, dim - 1])
            result = np.zeros((data_temp.shape[0], args.output_dim + 1)).astype("str")
            result[:, 0] = data.iloc[i*args.file_max_num:min((i+1)*args.file_max_num, N), 0].values
            if args.env == "inference_user":
                data_temp = model_layer_user(data_temp)
                data_temp = model_layer(data_temp)
                data_temp = model_user(data_temp)
            elif args.env == "inference_club":
                data_temp = model_layer_club(data_temp)
                data_temp = model_layer(data_temp)
                data_temp = model_club(data_temp)
            else:
                raise TypeError("args.env必需是inference_user或inference_club！")
            result[:, 1::] = data_temp
            #写入结果
            write(result.tolist(), count, i)


# 第一步读取数据
if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--env", help="运行的环境(train or test)", type=str, default='train_incremental')
    parser.add_argument("--lr", help="学习率", type=float, default=0.00001)
    parser.add_argument("--stop_num", help="early stop机制的触发次数", type=int, default=20)
    parser.add_argument("--batch_size", help="batch的大小", type=int, default=3048)
    parser.add_argument("--batch_during", help="打印batch loss的周期", type=int, default=30)
    parser.add_argument("--epoch", help="迭代次数", type=int, default=200)
    parser.add_argument("--prob", help="打印训练结果auc的概率", type=float, default=0.05)
    parser.add_argument("--sim", help="向量相似度阈值", type=float, default=0.7)
    parser.add_argument("--split_dim", help="玩家特征的维数", type=int, default=88)
    parser.add_argument("--feat1", help="隐含层输出特征的维度1", type=int, default=400)
    parser.add_argument("--feat2", help="隐含层输出特征的维度2", type=int, default=300)
    parser.add_argument("--feat3", help="隐含层输出特征的维度3", type=int, default=200)
    parser.add_argument("--feat4", help="隐含层输出特征的维度4", type=int, default=100)
    parser.add_argument("--output_dim", help="输出特征的维度", type=int, default=64)
    parser.add_argument("--file_max_num", help="单个csv文件中写入数据的最大行数", type=int, default=100000)
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--model_output", help="模型的输出位置",
                        type=str, default='s3://models/gibclv2/')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    if args.env == "train" or args.env == "train_incremental":
        train()
    elif args.env == "inference_user" or args.env == "inference_club":
        inference()
    else:
        raise TypeError("args.env必需是train或train_incremental或inference_user或inference_club！")
    end_time = time.time()
    print("算法总共耗时:{}".format(end_time - start_time))