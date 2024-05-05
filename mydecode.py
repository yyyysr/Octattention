import numpy as np
import torch


'''
一共是三步:
1.主函数中一个总的解码建树的函数
  传入参数为:
    二进制文件路径列表，命名分别为Occupancy，color0,color1,color2
    原始的Occupancy、color0、color1、color2
    加了参数的模型
    窗口大小
  
  主要过程:
    首先拿到根节点的occupancy，然后构建初始的解码器实例，调用decodeNode(下面函数2)
    
  传出参数为:
    出来的OccupancyCode和对应的颜色

2.解码器类
  init:
    传入参数为:
      # byte_stream: the bin file stream.(Node)
      sysNum: the Number of symbols that you are going to decode. This value should be 
              saved in other ways.    需要解码的数量，这个在别的地方存储
      sysDim: the Number of the possible symbols.   符号种类
      binfile: bin file path, if it is Not None, 'byte_stream' will read from this file
              and copy to Cpp backend Class 'InCacheString'   二进制文件路径
    
    主要过程：
      self.byte_stream根据binfile进行读取
      self.decode=初始化AC实例，传入参数为二进制串、节点数量
'''