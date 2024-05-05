import os
import numpy as np
import subprocess

PCERRORPATH = "D:\PythonFilesForPycharm\Octattention-mywork\octattention-final\Mywork\dataprepare\\file\pc_error"
TEMPPATH = "D:\PythonFilesForPycharm\Octattention-mywork\octattention-final\Mywork\\temp\data/"

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

def pcerror(pcRefer,pc,pcReferNorm,pcerror_cfg_params, pcerror_result,pcerror_path=PCERRORPATH):
  '''
  Options:
          --help=0            This help text
    -a,   --fileA=""          Input file 1, original version
    -b,   --fileB=""          Input file 2, processed version
    -n,   --inputNorm=""      File name to import the normals of original point
                              cloud, if different from original file 1n
    -s,   --singlePass=0      Force running a single pass, where the loop is
                              over the original point cloud
    -d,   --hausdorff=0       Send the Haursdorff metric as well
    -c,   --color=0           Check color distortion as well
    -l,   --lidar=0           Check lidar reflectance as well
    -r,   --resolution=0      Specify the intrinsic resolution
          --dropdups=2        0(detect), 1(drop), 2(average) subsequent points
                              with same coordinates
          --neighborsProc=1   0(undefined), 1(average), 2(weighted average),
                              3(min), 4(max) neighbors with same geometric
                              distance
          --averageNormals=1  0(undefined), 1(average normal based on neighbors
                              with same geometric distance)
          --mseSpace=1        Colour space used for PSNR calculation
                              0: none (identity) 1: ITU-R BT.709 8: YCgCo-R
          --nbThreads=1       Number of threads used for parallel processing
  '''
  if pcerror_result is not None:
    pcLabel =os.path.basename(pcerror_result).split(".")[0]
  else:
    pcLabel = "pt0"
  if type(pc) is not str:
    write_ply_data(TEMPPATH + pcLabel + "pc.ply",pc)
    pc = TEMPPATH +pcLabel + "pc.ply"
  if type(pcRefer) is not str:
    write_ply_data(TEMPPATH+pcLabel+"pcRefer.ply",pcRefer)
    pcRefer = TEMPPATH + pcLabel + "pcRefer.ply"
  if pcerror_result is not None:
    f = open(pcerror_result, 'a+')
  else:
    import sys
    f = sys.stdout
  if type(pcerror_cfg_params) is str:
    pcerror_cfg_params = pcerror_cfg_params.split(' ')
  if pcReferNorm==None:
    return subprocess.Popen([pcerror_path,
            '-a', pcRefer, '-b', pc] + pcerror_cfg_params,
            stdout=f, stderr=f)
  return subprocess.Popen([pcerror_path,
                '-a', pcRefer, '-b', pc, '-n', pcReferNorm] + pcerror_cfg_params,
                stdout=f, stderr=f)