import numpy as np

#构造I矩阵
def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))#初始化邻接矩阵A：25*25，也就是论文中的Ak：N*N，即代码中的V(num_node)
    for i, j in link:#节点i和节点J相连
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 除以每列的和(归一化)
    Dl = np.sum(A, 0)#对每一列相加  度矩阵
    #Dl = np.sum(A, axis=0)
    h, w = A.shape #即代码中的V*V
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)#D的逆
    AD = np.dot(A, Dn)#A.dot(a,b)两个矩阵的内积
    return AD

#对邻接矩阵A进行归一化处理
def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape  # 即代码中的V*V
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)# DAD--归一化后的邻接矩阵
    return DAD


#neighbor+self_link:A+I
def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A

def get_uniform_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    A = I - N
    return A

def get_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    A = np.stack((I, N))
    return A

def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def get_DAD_graph(num_node, self_link, neighbor):
    A = normalize_undigraph(edge2mat(neighbor + self_link, num_node))
    return A


def get_DLD_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    A = I - normalize_undigraph(edge2mat(neighbor, num_node))
    return A



def get_part_based_spatial_graph(num_node, self_link, parts):
    stack = []
    stack.append(edge2mat(self_link, num_node))
    for p in parts:
        In = normalize_digraph(edge2mat(p, num_node))
        stack.append(In)
        opp = [(y,x) for (x,y) in p]
        Out = normalize_digraph(edge2mat(opp, num_node))
        stack.append(Out)

    A = np.stack(stack)
    return A



def get_part_based_graph(num_node, self_link, parts):
    stack = []
    stack.append(edge2mat(self_link, num_node))
    for p in parts:
        opp = [(y,x) for (x,y) in p]
        p.extend(opp)#list.extend(a)函数会把a中的各个元素分开，a中的内容不再是一个整体。
        stack.append(normalize_undigraph(edge2mat(p, num_node)))

    A = np.stack(stack)
    #print(A.shape[0])
    return A



def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


class Graph(object):
    """ The Graph to model the human skeletons

    Constructor Args:
        labeling_mode (type string) ->
              distance*: Uniform distance partitioning
              distance: Distance partitioning
              spatial: Spatial configuration
              DAD: normalized graph adjacency matrix
              DLD: normalized graph laplacian matrix
              parts: Mid-level part-based partitioning
        num_node (type int) ->
            Number of joints in the tracked skeleton
        inward (type list) ->
            List of tuples having connections for centripetal group
        outward (type list) ->
            List of tuples having connections for centrifugal group
        parts (type list) ->
            List of lists having joints aggregated into four parts,
            namely head, hands, torso and legs.
    """

    def __init__(self,
                 inward,
                 outward,
                 parts,
                 labeling_mode='uniform',
                 num_node=25):
        if isinstance(num_node, list):
            self.self_link = []
            self.A = []
            for i in range(len(num_node)):
                self.num_node = num_node[i]
                self.inward = inward[i]
                self.outward = outward[i]
                self.neighbor = self.inward + self.outward
                self.parts = parts[i]
                self.self_link.append([(x, x) for x in range(self.num_node)])
                self.A.append(self.get_adjacency_matrix(labeling_mode))
            self.num_node = num_node
            self.inward = inward
            self.outward = outward
            self.neighbor = [inward[i] + outward[i] for i in range(len(inward))]
            self.parts = parts
            A = []
            for i in range(len(self.A[0])):
                A.append(self.A[0][i] + self.A[1][i])
            self.A = A
        else:
            self.num_node = num_node
            self.inward = inward
            self.outward = outward
            self.neighbor = inward + outward
            self.parts = parts
            self.self_link = [(x, x) for x in range(num_node)]
            self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'uniform':
            A = get_uniform_graph(self.num_node, self.self_link, self.neighbor)
        elif labeling_mode == 'distance*':
            A = get_uniform_distance_graph(self.num_node, self.self_link, self.neighbor)
        elif labeling_mode == 'distance':
            A = get_distance_graph(self.num_node, self.self_link, self.neighbor)
        elif labeling_mode == 'spatial':
            A = get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        elif labeling_mode == 'DAD':
            A = get_DAD_graph(self.num_node, self.self_link, self.neighbor)
        elif labeling_mode == 'DLD':
            A = get_DLD_graph(self.num_node, self.self_link, self.neighbor)
        elif labeling_mode == 'parts':
            A = get_part_based_graph(self.num_node, self.self_link, self.parts)

        elif labeling_mode == 'parts+spatial':
            A = get_part_based_spatial_graph(self.num_node, self.self_link, self.parts)
        else:
            raise ValueError()
        return A
