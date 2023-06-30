from pyspark.sql import SparkSession

import argparse
import networkx as nx
import os
import subgraph_random_sampling
import hashlib
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import  numpy as np
import random
def parse_args():
    # def parse_args(num):
    '''
	Parses the S2GN arguments.
	'''
    parser = argparse.ArgumentParser(description="Run RandomSGN.")
    # nargs - 应该读取的命令行参数个数，可以是具体的数字，或者是?号；default - 不指定参数时的默认值；help - 参数的帮助信息，当指定为 argparse.SUPPRESS 时表示不显示该参数的帮助信息.
    parser.add_argument('--input', nargs='?', default="IMDB-BINARY_gexf",
                        help='Input graph path')

    parser.add_argument('--label', nargs='?', default="IMDB-BINARY.Labels",
                        help='Input graph path')
    parser.add_argument('--rank_type', type=int, default=1,
                        help='Type of node rank. Default is 1 or 2.')

    parser.add_argument('--types', type=int, default=0,
                        help='Type of processing the features. Default is 1 or 2.')

    parser.add_argument('--N', type=int, default=2,
                        help='Number of convert to line graph. Default is 2.')

    parser.add_argument('--T', type=int, default=1,
                        help='Number of sampling times. Default is 10.')

    parser.add_argument('--p', type=float, default=4,
                        help='Return hyperparameter. Default is 4.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    # graph2vec hyper parameters

    parser.add_argument("--dimensions", type=int, default=1024,
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


def to_line(graph):
    '''
	:param graph
	:return G_line: line/Subgraph network
	'''
    graph_to_line = nx.line_graph(graph)
    graph_line = nx.convert_node_labels_to_integers(graph_to_line, first_label=0, ordering='default')
    return graph_line  # , list(graph_line.edges())


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

def read_label():
    with open(args.label) as f:
        singlelist = [line.strip()[-1] for line in f.readlines()]
    return singlelist
def RSP_partition(parameters_for_block_size, parameters_for_block_num, graphs):
    RSP_blocks = {}
    selected_RSP_idxes = {}
    for block_size in parameters_for_block_size:
        for num_of_blocks in parameters_for_block_num:
            key = '%d %d' % (block_size, num_of_blocks)
            RSP_blocks[key] = []
            # selected_RSP_idx[(block_size,num_of_blocks)]=[]
            temp = list(range(len(graphs)))
            random.shuffle(temp)
            i = 0
            while i + block_size < len(graphs):
                RSP_blocks[key].append(temp[i:i + block_size])
                i += block_size
            RSP_blocks[key].append(temp[i:len(temp)])
            temp = list(range(len(RSP_blocks[key])))
            random.shuffle(temp)
            selected_RSP_idxes[key] = temp[:num_of_blocks]
    for block_size in parameters_for_block_size:
        for num_of_blocks in parameters_for_block_num:
            key = '%d %d' % (block_size, num_of_blocks)

            RSSP = [[graphs[j] for j in RSP] for RSP in [RSP_blocks[key][i] for i in selected_RSP_idxes[key]]]
    return RSSP


if __name__ == "__main__":
    spark = SparkSession.builder.appName("test").getOrCreate()

    args = parse_args()
    rounds = args.wl_iterations
    all_features = []
    files = os.listdir(args.input)
    # 对文件列表进行排序，key=lambda 元素: 元素[字段索引]
    files.sort(key=lambda x: str(x.split('.')[0]))
    values = []

    graphs = []

    graph_0 = []
    graph_1 = []
    labels_all = []
    label_0 = []

    labels = read_label()
    num = 0
    flag = False
    for path in files:

        nx_G = read_graph(path)  ## one network come in
        path = str(path.split('.')[0])
        label = int(labels[int(path) - 1])
        cur = nx_G
        if not nx.is_connected(cur):
            nodeset = max(nx.connected_components(cur), key=len)
            cur = cur.subgraph(nodeset)
        walk_length = len(cur)

        for n in range(args.N):
            for edge in cur.edges():
                cur[edge[0]][edge[1]]['weight'] = 1
            G = subgraph_random_sampling.Graph(cur, args.directed, args.p, args.q)
            G.preprocess_transition_probs()
            graph_edges1 = G.simulate_walks(walk_length)
            for graph_edge1 in graph_edges1:
                if (len(graph_edge1) > 0):
                    num += 1
                    graph = nx.Graph()
                    graph.add_edges_from(graph_edge1)
                    for edge in graph.edges():
                        graph[edge[0]][edge[1]]['weight'] = 1

                    if (n == 0):
                        graph_0.append(graph)
                        graphs.append(graph)

                        labels_all.append(label)
                        label_0.append(label)

                    elif (n == 1):
                        graph_1.append(graph)
                        graphs.append(graph)
                        labels_all.append(label)

                        graph = to_line(graph)
                        graphs.append(graph)
                        labels_all.append(label)
            graph_edges2 = subgraph_random_sampling.linksample(cur, walk_length)

            for graph_edge2 in graph_edges2:

                if (len(graph_edge2) > 0):
                    num += 1
                    graph = nx.Graph()
                    graph.add_edges_from(graph_edge2)
                    for edge in graph.edges():
                        graph[edge[0]][edge[1]]['weight'] = 1

                    if (n == 0):
                        graph_0.append(graph)
                        graphs.append(graph)

                        labels_all.append(label)
                        label_0.append(label)

                    elif (n == 1):
                        graph_1.append(graph)
                        graphs.append(graph)
                        labels_all.append(label)

                        graph = to_line(graph)
                        graphs.append(graph)
                        labels_all.append(label)

            cur = to_line(cur)

    len_0 = len(graph_0)
    len_1 = len(graph_1)

    tmp_0 = len_0
    while (tmp_0 < len_1):
        array = random.randint(0, len_0 - 1)
        graphs.append(graph_0[array])
        labels_all.append(label_0[array])
        tmp_0 += 1
    print(len(graphs))
    class1 = list()
    class2 = list()
    for i in range(len(graphs)):
        if(labels_all[i] == 1):
            class1.append(nx.density(graphs[i]))
        else:
            class2.append(nx.density(graphs[i]))
    class1 = np.array(class1)
    class2 = np.array(class2)

    class1_mean = np.mean(class1)
    class2_mean = np.mean(class2)

    class1_var = np.var(class1)
    class2_var = np.var(class2)

    f1 = ((class1_mean - class2_mean)*(class1_mean - class2_mean))/(class1_var * class1_var + class2_var * class2_var)
    print(f1)
    s = 1.960 * 1.960 * 5 / 0.01 * (1 - 1/f1)
    # s = 2.576 * 2.576 * 5 / 0.0036 * (1 - 1 / f1)
    print(s)