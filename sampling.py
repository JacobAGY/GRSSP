# -*- coding: UTF-8 -*-
import numpy as np
import networkx as nx
import random
import Rank
import multiprocessing

# 将图中的边进行排序以元组的形式返回
def linksample(G):
    # print("process name2: " + multiprocessing.current_process().name)

    link_length = len(G)
    if int(G.size()) <= int(G.__len__()):
        samplelink = set(list(G.edges()))
    else:
        samplelink = set()    # sample graph
        start_edge = Rank.edge_rank(G)  # select a start edge
        start_edge = tuple(sorted([start_edge[0], start_edge[1]]))
        samplelink.add(start_edge)   # update sample graph
        select_nodes = set(start_edge)
        while len(samplelink) < link_length:
            node = random.choice(list(select_nodes))
            nextnode = random.choice(list(G.neighbors(node)))
            # if nextnode in select_nodes:
            #     nextnode = random.choice(list(G.neighbors(node)).pop(nextnode))
            samplelink.add(tuple(sorted([node, nextnode])))
            select_nodes.add(nextnode)
    return samplelink


def spanning_tree(G):
    # print("process name3: " + multiprocessing.current_process().name)
    # 生成树是包含图中所有节点的最小连接子结构，如图3所示。通过从不同的节点进行遍历，可以得到不同的生成树。这里我们随机选择一个节点作为初始节点。
    if int(G.size()) <= int(G.__len__()):
        samplelink = set(list(G.edges()))
    else:
        T = nx.maximum_spanning_tree(G)
        samplelink = set(list(T.edges()))
    return samplelink


# node2vec sampling method
class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        # 模拟随机游走模型
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        walk = [start_node]
        # G1 = nx.Graph()
        edge = set()
        while len(edge) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    neb = cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]
                    walk.append(neb)

                    # G1.add_edge(cur, cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                    # edge.add(tuple(sorted([cur, cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]])))
                    edge.add(tuple(sorted([cur, neb])))
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next)

                    if G.has_edge(prev, next):
                        edge.add(tuple(sorted([prev, next])))
                    if G.has_edge(cur, next):
                        # G1.add_edge(prev, next)
                        edge.add(tuple(sorted([cur, next])))
            else:
                break
        # print("walk list:", walk)
        return edge


    def simulate_walks(self, rank_type, walk_length):
        # multiprocessing.get_logger()

        '''
        Repeatedly simulate random walks from each node.Biased Walk
        '''
        # Rank.k_shell_rank(G)和Rank.leader_rank(G)都是拿到起始节点
        G = self.G
        if rank_type == 1:
            node = Rank.k_shell_rank(G)   # node ranking methods. Here is k-shell rank method.
        elif rank_type == 2:
            node = Rank.leader_rank(G)   # node ranking methods. Here is Leader rank method.
        else:
            raise Exception("Invalid rank_type!", rank_type)



        if int(G.size()) <= int(G.__len__()):
            edge = set(list(G.edges()))
        else:
            edge = self.node2vec_walk(walk_length=walk_length, start_node=node)
            edge = list(edge)

        return edge

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)
		

    def preprocess_transition_probs(self):
        # 用于引导biased walks的转移概率的预处理。
        '''
        Preprocessing of transition probabilities for guiding the biased walks.
        '''
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    # 从离散分布中计算非均匀抽样的实用程序列表。
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    # np.zeros(K) 返回来一个给定形状和类型的用0填充的数组;
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)
    # np.random.rand() 通过本函数返回一个[0.1）范围内的随机样本值。
    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
