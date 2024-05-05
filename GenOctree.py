import numpy as np
# import open3d as o3d

class OctreeNode:
    def __init__(self, node_id=None, parent_id=None, child_points=None,color=None,position=None):
        self.node_id = node_id
        self.parent_id = parent_id
        self.child_points = child_points if child_points is not None else [[] for _ in range(8)]
        self.pos=position
        self.occupancy=None
        self.color=color
        self.octant = None

#存储整个树,暂时用不到，现在考虑位置信息怎么用即可
class AllOctree:
    def __init__(self):
        self.octree=[]
        self.Lmax=None
        self.nodeNum=None
        self.occupancy=[]

    def __len__(self):
        return self.Lmax

    def __getitem__(self, index):
        if index>self.Lmax or index<-self.Lmax:
            raise IndexError("索引超出边界")
        else:
            if index<0:
                index+=self.Lmax
            return self.octree[index]




#存储整棵树的每一层的子树
class Octree:
    def __init__(self):
        self.nodes = []

def dec2bin(n, count=8): 
    """returns the binary of integer n, using count number of digits""" 
    return [int((n >> y) & 1) for y in range(count-1, -1, -1)]

def dec2binAry(x, bits):
    mask = np.expand_dims(2 ** np.arange(bits - 1, -1, -1), 1).T
    return (np.bitwise_and(np.expand_dims(x, 1), mask) != 0).astype(int)

def bin2decAry(x):
    if (x.ndim == 1):
        x = np.expand_dims(x, 0)
    bits = x.shape[1]
    mask = np.expand_dims(2 ** np.arange(bits - 1, -1, -1), 1)
    return x.dot(mask).astype(int)

def Morton(A):
    A = A.astype(int)
    n = np.ceil(np.log2(np.max(A) + 1)).astype(int)
    x = dec2binAry(A[:, 0], n)
    y = dec2binAry(A[:, 1], n)
    z = dec2binAry(A[:, 2], n)
    m = np.stack((x, y, z), 2)
    m = np.transpose(m, (0, 2, 1))
    mcode = np.reshape(m, (A.shape[0], 3 * n), order='F')
    return mcode

#计算指定八叉树节点的位置
def CalPos(parentPos,level,octant,Lmax,qs):
    pos=parentPos+np.array([int(c) for c in list(bin(octant)[2:].zfill(3))],dtype=int)*qs*2**(Lmax-level)
    return pos

def genOctree(points,color, Lmax=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    Codes = []
    mcode = Morton(points)  # 这里将三维映射到一维,具体莫顿码映射规则可以百度，这里不赘述
    if Lmax is None:  # 获取深度
        Lmax = np.ceil(mcode.shape[1] / 3).astype(int)
    pointNum = mcode.shape[0]  # 获取点的数量

    mcode2 = np.zeros((pointNum, Lmax))
    for n in range(Lmax):
        mcode2[:, n:n + 1] = bin2decAry(mcode[:, n * 3:n * 3 + 3])

    pointID = list(range(0, pointNum))
    nodeid = 0
    proot = OctreeNode(child_points=[np.array(pointID)],color=np.mean(color[:],axis=0),position=[0,0,0])
    octree = [Octree() for _ in range(Lmax+1)]
    octree[0].nodes.append(proot)
    for L in range(1, Lmax + 1):
        for node in octree[L-1].nodes:
            for octant, ptid in enumerate(node.child_points):
                if len(ptid) == 0:
                    continue
                nodeid += 1
                child_node = OctreeNode(node_id=nodeid, parent_id=node.node_id)
                idn = mcode2[ptid, L - 1]
                for i in range(8):
                    child_node.child_points[i] = ptid[idn == i] #将xyz同一个层级的点归类到一个子空间中
                child_node.color=np.mean(color[ptid],axis=0)
                child_node.pos=CalPos(node.pos, L, octant, Lmax+1,qs=1)
                # print(child_node.color)
                '''
                点进行莫顿码变换之后，可以用pid访问到这个点的编号，此时编号对应的颜色可以统计直接取平均，这里应该在
                遍历完当前父节点之后进行统计
                '''
                occupancyCode = np.in1d(np.array(range(7, -1, -1)), idn).astype(int)
                #当前节点的
                child_node.occupancy = int(bin2decAry(occupancyCode))
                child_node.octant = octant + 1
                Codes.append(child_node.occupancy)
                octree[L].nodes.append(child_node)

    del octree[0]

    # for L in range(0, Lmax):
    #     for node in octree[L].nodes:
    #         print(node.pos)

    return Codes, octree, Lmax



# def draw_point_with_bbox(vis, position, box_size, color):
#     """
#     参数:
#     - vis: open3d.visualization.Visualizer对象
#     - position: 点的位置，形状为(3,)的NumPy数组
#     - box_size: 边界框的尺寸，形状为(3,)的NumPy数组
#     - color: 点和边界框的颜色，形状为(3,)的NumPy数组
#     - point_size: 点的大小
#     """
#     # 创建一个点云对象，包含单个点
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector([position])
#     point_cloud.paint_uniform_color(color)
#     vis.add_geometry(point_cloud)
    
#     # 计算边界框的最小和最大坐标
#     min_bound = position - np.array(box_size) / 2
#     max_bound = position + np.array(box_size) / 2
    
#     # 创建边界框对象
#     bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
#     vis.add_geometry(bbox)
    
#     # # 设置点的大小
#     # vis.get_render_option().point_size = point_size




def DeOctree(Codes):
    Codes = np.squeeze(Codes)
    occupancyCode = np.flip(dec2binAry(Codes,8),axis=1)  
    codeL = occupancyCode.shape[0]                        
    N = np.ones((30),int) 
    codcal = 0
    L = 0
    while codcal+N[L]<=codeL:
        L +=1
        try:
            N[L+1] = np.sum(occupancyCode[codcal:codcal+N[L],:])
        except:
            assert 0
        codcal = codcal+N[L]
    Lmax = L
    octree = [Octree() for _ in range(Lmax+1)]
    proot = [np.array([0,0,0])]
    Octree[0].node = proot
    codei = 0
    for L in range(1,Lmax+1):
        childNode = []  # the node of next level
        for currentNode in Octree[L-1].node: # bbox of currentNode
            code = occupancyCode[codei,:]
            for bit in np.where(code==1)[0].tolist():
                newnode =currentNode+(np.array(dec2bin(bit, count=3))<<(Lmax-L))# bbox of childnode
                childNode.append(newnode)
            codei+=1
        Octree[L].node = childNode.copy()
    points = np.array(Octree[Lmax].node)
    return points