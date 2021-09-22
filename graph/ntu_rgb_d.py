import sys


sys.path.extend('../')
from graph import tools

from .tools import Graph

#用于创建NTU对应的图结构
num_node = 25
self_link = [(i, i) for i in range(num_node)] #自连接，为每个节点增加自环，使得聚合表征包含自身特征
#因为邻接矩阵的对角都是0，和特征矩阵内积，自身特征会被忽略，所以A=A+I

#关键节点的连接方式,ntu数据集各个节点的连接顺序
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
#将1-->0，为了从0开始
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index] #(i,j)就是inward_ori_index列表里的元组tuple
#构建无向图
outward = [(j, i) for (i, j) in inward] #列表解析后出来的还是以列表形式存在
#定义无向图
neighbor = inward + outward


#分成四部分，头部，手臂，髋部，腿部
head = [(2, 3), (2, 20), (20, 4), (20, 8)]
lefthand = [(4, 5), (5, 6), (6, 7), (7, 22), (22, 21)]
righthand = [(8, 9), (9, 10), (10, 11), (11, 24), (24, 23)]
torso = [(20, 4), (20, 8), (20, 1), (1, 0), (0, 12), (0, 16)]
hands = lefthand + righthand
leftleg = [(0, 12), (12, 13), (13, 14), (14, 15)]
rightleg = [(0, 16), (16, 17), (17, 18), (18, 19)]
legs= leftleg + rightleg








class NTUGraph(Graph):
    def __init__(self,
                 labeling_mode='uniform'):
        super(NTUGraph, self).__init__(num_node=num_node,
                                      inward=inward,
                                      outward=outward,
                                      parts=[hands, legs, head, torso ],
                                      labeling_mode = labeling_mode)





# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import os
#
#     # os.environ['DISPLAY'] = 'localhost:11.0'
#     A = NTUGraph('parts').get_adjacency_matrix()
#     f, i = plt.subplots(1, 3)
#     for i in A: #A[0]=(1,25,25),A[1]=(1,25,25),A3=[1,25,25]
#         plt.imshow(i, cmap='gray')
#         plt.show()
#     print(A)
