# -*- coding: UTF-8 -*-
import argparse
import time

import numpy as np
import networkx as nx
import sampling
import classification
from itertools import chain
from sklearn.decomposition import PCA
import os

import hashlib

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging

FORMAT = '%(asctime)s\t %(processName)s %(process)d %(message)s'
# logging.basicConfig(level=logging.INFO, format=FORMAT,filename='mutag_5N_100T.log',
#                     filemode='w',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
#                     #a是追加模式，默认如果不写的话，就是追加模式
#
#                     )
def parse_args():
# def parse_args(num):
	'''
	Parses the S2GN arguments.
	'''
	parser = argparse.ArgumentParser(description="Run RandomSGN.")
	# nargs - 应该读取的命令行参数个数，可以是具体的数字，或者是?号；default - 不指定参数时的默认值；help - 参数的帮助信息，当指定为 argparse.SUPPRESS 时表示不显示该参数的帮助信息.
	parser.add_argument('--input', nargs='?', default="mutag_gexf",
	                    help='Input graph path')

	parser.add_argument('--label', nargs='?', default="mutag.Labels",
	                    help='Input graph path')

	parser.add_argument('--rank_type', type=int, default=1,
	                    help='Type of node rank. Default is 1 or 2.')

	parser.add_argument('--types', type=int, default=0,
	                    help='Type of processing the features. Default is 1 or 2.')

	parser.add_argument('--N', type=int, default=0,
	                    help='Number of convert to line graph. Default is 2.')

	parser.add_argument('--T', type=int, default=1,
	                    help='Number of sampling times. Default is 10.')

	parser.add_argument('--p', type=float, default=4,
	                    help='Return hyperparameter. Default is 4.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	# graph2vec hyper parameters

	parser.add_argument("--dimensions", type=int, default=128,
                        help="Number of dimensions. Default is 128.")

	parser.add_argument("--workers", type=int, default=4,
                        help="Number of workers. Default is 4.")

	parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs. Default is 1.")

	parser.add_argument("--min-count", type=int, default=5,
                        help="Minimal structural feature count. Default is 5.")

	parser.add_argument("--wl-iterations", type=int, default=2,
                        help="Number of Weisfeiler-Lehman iterations. Default is 2.")

	parser.add_argument("--learning-rate", type=float, default=0.025,
                        help="Initial learning rate. Default is 0.025.")

	parser.add_argument("--down-sampling", type=float, default=0.0001,
                        help="Down sampling rate of features. Default is 0.0001.")

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()


class WeisfeilerLehmanMachine:
	"""
    Weisfeiler Lehman feature extractor class.
    """
	def   __init__(self, graph, features, iterations):
		"""
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
		self.iterations = iterations
		self.graph = graph
		self.features = features
		self.nodes = self.graph.nodes()
		self.extracted_features = [str(v) for k, v in features.items()]
		self.do_recursions()

	def do_a_recursion(self):
		""" 返回的是提取出来的节点的度的值
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
		new_features = {}
		for node in self.nodes:
			nebs = self.graph.neighbors(node)
			# degs = [self.features[neb] for neb in nebs]
			# 邻居节点所对应的度以列表形式表示
			degs = [self.features[neb] for neb in nebs]
			features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
			features = "_".join(features)
			hash_object = hashlib.md5(features.encode())
			# hexdigest()返回摘要，作为十六进制数据字符串值
			hashing = hash_object.hexdigest()
			new_features[node] = hashing
		self.extracted_features = self.extracted_features + list(new_features.values())
		return new_features

	def do_recursions(self):
		"""
        The method does a series of WL recursions.
        """
		for _ in range(self.iterations):
			self.features = self.do_a_recursion()


def read_graph(path):
	'''
	Reads the input network in networkx.
	'''
	# weighted：'Boolean specifying (un)weighted. Default is unweighted.'
	if args.weighted:
		G = nx.read_gexf(os.path.join(args.input, path), node_type=int)
	else:
		G = nx.read_gexf(os.path.join(args.input, path), node_type=str)
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1
	if not args.directed:
		G = G.to_undirected()

	return G


def read_label():
	with open(args.label) as f:
		singlelist = [line.strip()[-1] for line in f.readlines()]
		labels = np.array(singlelist)
	return labels


def to_line(graph):
	'''
	:param graph
	:return G_line: line/Subgraph network
	'''
	graph_to_line = nx.line_graph(graph)
	graph_line = nx.convert_node_labels_to_integers(graph_to_line, first_label=0, ordering='default')
	return graph_line #, list(graph_line.edges())


def feature_extractor(graph, rounds, name):
	"""
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
	# print('graph edges:', graph.edges())  提取特征
	# dict()返回一个字典值 拿图顶点的度作为特征提取
	features = dict(nx.degree(graph))
	# print('啊', classification.character(graph))
	# features = dict(nx.density(graph))
	# features = {int(k): v for k, v in features.items()}
	features = {k: v for k, v in features.items()}
	# print("features:", features)
	name = name.split('.')[0]
	machine = WeisfeilerLehmanMachine(graph, features, rounds)
	doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
	return doc

def main(args, fullpath, path):
	'''
	Obtain Substructure for each graph and transform it to different-order SGNs.
	'''
	nx_G = read_graph(fullpath)  ## one network come in
	print('PATH:', path)
	SGN_FEATURES = list()
	# T:'Number of sampling times. Default is 10.'

	for t in range(args.T):      ## 10 times sampling
		print('sampling times:', t)
		document_SGN = list()
		print("\nFeature extraction started.\n")
		# wl_iterations："Number of Weisfeiler-Lehman iterations. Default is 2." 做两次映射
		document_SGN.append(feature_extractor(nx_G, args.wl_iterations, path))
		cur= nx_G
		if not nx.is_connected(cur):
			nodeset = max(nx.connected_components(cur),key=len)
			cur = cur.subgraph(nodeset)
		cur1 = cur
		cur2 = cur
		cur3 = cur
		walk_length = len(cur)
		for n in range(args.N):
			# N:Number of convert to line graph 做两次SGN变成SGN2
			G = sampling.Graph(cur1, args.directed, args.p, args.q)
			# 用于对Biased Walk 进行预处理
			G.preprocess_transition_probs()
			# print('G.alias_edges:', G.alias_edges)
			# graph_edge1 = G.simulate_walks(args.rank_type)  ## sample a subgraph
			graph_edge1 = G.simulate_walks(args.rank_type,walk_length)

			graph_edge2 = sampling.linksample(cur2)  ## sample a subgraph

			graph_edge3 = sampling.spanning_tree(cur3)  ## sample a spanning tree

			print(n, 'th graph sampling is ok!')
			graph1 = nx.Graph()
			graph2 = nx.Graph()
			graph3 = nx.Graph()
			# graph_edge1是一个集合list，graph1是一个类，图
			graph1.add_edges_from(graph_edge1)
			graph2.add_edges_from(graph_edge2)
			graph3.add_edges_from(graph_edge3)
			# print("graph_edge:", graph_edge)
			sgn1 = to_line(graph1)
			sgn2 = to_line(graph2)
			sgn3 = to_line(graph3)
			for edge in sgn1.edges():
				sgn1[edge[0]][edge[1]]['weight'] = 1
			for edge in sgn2.edges():
				sgn2[edge[0]][edge[1]]['weight'] = 1
			for edge in sgn3.edges():
				sgn3[edge[0]][edge[1]]['weight'] = 1
			print(n, 'th convert SGN is ok!') ## SGNAlgorithms(G);

			document_SGN.append(feature_extractor(sgn1, args.wl_iterations, path))  ## get the feature of one network
			document_SGN.append(feature_extractor(sgn2, args.wl_iterations, path))  ## get the feature of one network
			document_SGN.append(feature_extractor(sgn3, args.wl_iterations, path))  ## get the feature of one network

			cur1 = sgn1
			cur2 = sgn2
			cur3 = sgn3

		SGN_FEATURES.append(document_SGN)

	return SGN_FEATURES


def model(document_collections):
	print("\nOptimization started.\n")
	model = Doc2Vec(document_collections, vector_size=args.dimensions, window=0, min_count=args.min_count,
			dm=0, sample=args.down_sampling, workers=args.workers,
			epochs=args.epochs, alpha=args.learning_rate)
	return model


def feature_processing(all_features):
	'''
	:param all_features:
	      1:PCA;
	      2:get the mean value
	:return: the reduced dimension network feature vector.
	'''
	if args.types == 1:
		fea_list = []
		for fea in all_features:
			features = list(chain(*fea))  ## flatten a list
			fea_list.append(features)
		x = np.array(fea_list)
		pca = PCA(n_components=1024)   # decided by the feature abstract method.
		reduced_x = pca.fit_transform(x)

	elif args.types == 2:
		fea_list = []
		for fea in all_features:
			features = np.array(fea)
			fea_list.append(list(np.mean(features, axis=0)))
		reduced_x = np.array(fea_list)

	elif args.types == 3:
		fea_list = []
		for fea in all_features:
			features = list(chain(*fea))  ## flatten a list
			fea_list.append(features)
		reduced_x = np.array(fea_list)

	elif args.types == 0:
		reduced_x = np.array(all_features)

	else:
		raise Exception("Invalid feature processing type!", type)

	return reduced_x


if __name__ == "__main__":
	start = time.time()
	# parse_args()是对定义的参数进行赋值，add_argument()
	args = parse_args()
	# all_feature = []
	all_features = []
	# listdir目录路径，不传参为当前目录，args.input即MUTAG
	files = os.listdir(args.input)
	# 对文件列表进行排序，key=lambda 元素: 元素[字段索引]
	files.sort(key=lambda x: str(x.split('.')[0]))

	class myclass(object):
		def __init__(self):
			pass
	t = myclass()
	# T：Number of sampling times
	for i in range(args.T):
		# N：Number of convert to line graph
		for j in range(args.N*3+1):
			setattr(t, "document_collections"+str(i)+str(j), [])
	Fea = myclass()
	for path in files:
		i = path.split('.')[0]
		setattr(Fea, str(i), [])

	# get a object contained T*(N+1) lists of all files' document, each list is SGN_ij's document of all files.
	for path in files:
		full_path = path
		# print('num_graph:', path)
		document_features = main(args, full_path, path)
		for i in range(args.T):
			for j in range(args.N*3+1):
				logging.info("lai ==================")
				logging.info(len(t.__dict__))
				## 3 is the number of sampling methods. 1-sampling = N+1, 2-sampling = N^2+1, 3-sampling = N^3+1
				string = "document_collections"+str(i)+str(j)
				t.__dict__[string] += [document_features[i][j]]

	for i in range(args.T):
		for j in range(args.N*3+1):
			string1 = "document_collections"+str(i)+str(j)
			print(string1)
			doc = t.__dict__[string1]
			# print(doc)
			mymodel = model(doc)  ##  doc2vec的目标是创建文档的向量化表示，而不管其长度如何。
			print("get the emb of each graph!")
			for path in files:
				path = str(path.split('.')[0])
				Fea.__dict__[path] += list(mymodel.docvecs["g_"+ path])
				# dv:该对象包含从训练数据中学习到的段落向量。 有一个这样的向量训练期间提供的每个独特的文档标签。 可以使用标记单独访问它们作为索引访问键。
				Fea.__dict__[path] += list(mymodel.dv["g_" + path])
				# print(Fea.__dict__[path].values())

				# for m in list(mymodel.dv["g_" + path]):
				# 	features.append(m)
	for key in sorted([int(key) for key in Fea.__dict__.keys()]):
		all_features.append(Fea.__dict__[str(key)])

	# reduced_x = feature_processing(all_features)
	# pca = PCA(n_components=1024)
	reduced_x = feature_processing(all_features)
	labels = read_label()
	classification.result_class(reduced_x, labels)
	end = time.time()
	logging.info("=========time===========")
	logging.info(end-start)