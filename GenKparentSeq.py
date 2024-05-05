import numpy as np

def genKparentSeq(Octree,K):
    LevelNum =Octree[0].level
    nodeNum = Octree[-1].nodes[-1].node_id
    occupancySeq = np.ones((nodeNum,K),'int')*255
    LevelOctant = np.zeros((nodeNum,K,2),'int') # Level and Octant
    Color = np.zeros((nodeNum,K,3),'float'); #padding 0
    Pos = np.zeros((nodeNum,K,3),'int'); #padding 0
    ChildID = [[] for _ in range(nodeNum)]
    occupancySeq[0,K-1] = Octree[0].nodes[0].occupancy
    LevelOctant[0,K-1,0] = 1
    LevelOctant[0,K-1,1] = 1
    Pos[0,K-1,:] = Octree[0].nodes[0].pos
    Color[0,K-1,:] = Octree[0].nodes[0].color
    Octree[0].nodes[0].parent_id = 1 # set to 1
    n= 0
    for L in range(0,LevelNum):
        for node in Octree[L].nodes:
            occupancySeq[n,K-1] = node.occupancy
            occupancySeq[n,0:K-1] = occupancySeq[node.parent_id-1,1:K]
            LevelOctant[n,K-1,:] = [L+1,node.octant]
            LevelOctant[n,0:K-1] = LevelOctant[node.parent_id-1,1:K,:]
            Color[n,K-1] = node.color
            Color[n,0:K-1,:] = Color[node.parent_id-1,1:K,:]
            Pos[n,K-1] = node.pos
            Pos[n,0:K-1,:] = Pos[node.parent_id-1,1:K,:]
            if (L==LevelNum-1):
                pass
            n+=1
    assert n==nodeNum
    DataStruct = {'occupancySeq':occupancySeq,'Level':LevelOctant,'ChildID':ChildID,'Color':Color,'Pos':Pos}
    return DataStruct