import os
import numpy as np


def write_ply_data(filename, points,attributeName=[],attriType=[]): 
    '''
    write data to ply file.
    e.g pt.write_ply_data('ScanNet_{:5d}.ply'.format(idx), np.hstack((point,np.expand_dims(label,1) )) , attributeName=['intensity'], attriType =['uint16'])
    '''
    # if os.path.exists(filename):
    #   os.system('rm '+filename)
    if type(points) is list:
      points = np.array(points)

    attrNum = len(attributeName)
    assert points.shape[1]>=(attrNum+3)

    if os.path.dirname(filename)!='' and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename)) 

    plyheader = ['ply\n','format ascii 1.0\n'] + ['element vertex '+str(points.shape[0])+'\n'] + ['property float x\n','property float y\n','property float z\n']
    for aN,attrName in enumerate(attributeName):
      plyheader.extend(['property '+attriType[aN]+' '+ attrName+'\n'])
    plyheader.append('end_header')
    typeList = {'uint16':"%d",'float':"float",'uchar':"%d"}

    np.savetxt(filename, points, newline="\n",fmt=["%f","%f","%f"]+[typeList[t] for t in attriType],header=''.join(plyheader),comments='')

    return