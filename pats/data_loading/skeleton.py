import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import pandas as pd

from common import Modality, MissingData


"""
专门用于处理 2D 骨骼数据（如人体姿态数据），并实现了Modality类中定义的抽象方法和一些特定功能
"""
class Skeleton2D(Modality):
    def __init__(self, path2data='../data',
               path2outdata='../data',
               speaker='oliver',
               preprocess_methods=['data']):
        super(Skeleton2D, self).__init__(path2data=path2data)
        self.path2data = path2data
        self.df = pd.read_csv(Path(self.path2data) / 'cmu_intervals_df.csv', dtype=object)
        self.df.loc[:, 'delta_time'] = self.df['delta_time'].apply(float)
        self.df.loc[:, 'interval_id'] = self.df['interval_id'].apply(str)
        self.path2outdata = path2outdata
        self.speaker = speaker
        self.preprocess_methods = preprocess_methods
        self.missing = MissingData(self.path2data)

        """
        parents function定义了骨骼关节的父子关系(树状层次结构)。每个关节都有一个父关节，这种关系通常用于运动捕捉、骨骼动画或姿态估计任务中
        
        - 返回值返回一个列表，表示每个关节的父关节索引：
            • -1：   表示根关节（没有父关节）
            • 其他值：表示当前关节的父关节索引。
            
        这种父子关系通常用于构建骨骼树结构，便于计算关节的相对运动或全局位置
        
        Root: Neck (Index 0)
        ├── RShoulder (1)
        │   └── RElbow (2)
        │       └── RWrist (3)
        │           └── RHandRoot (31)  
        │               ├── RHandThumb1 (32)
        │               │   └── RHandThumb2 (33)
        │               │       └── RHandThumb3 (34)
        │               │           └── RHandThumb4 (35)
        │               ├── RHandIndex1 (36)
        │               │   └── RHandIndex2 (37)
        │               │       └── RHandIndex3 (38)
        │               │           └── RHandIndex4 (39)
        │               ├── RHandMiddle1 (40)
        │               │   └── RHandMiddle2 (41)
        │               │       └── RHandMiddle3 (42)
        │               │           └── RHandMiddle4 (43)
        │               ├── RHandRing1 (44)
        │               │   └── RHandRing2 (45)
        │               │       └── RHandRing3 (46)
        │               │           └── RHandRing4 (47)
        │               └── RHandLittle1 (48)
        │                   └── RHandLittle2 (49)
        │                       └── RHandLittle3 (50)
        │                           └── RHandLittle4 (51)
        ├── LShoulder (4)
        │   └── LElbow (5)
        │       └── LWrist (6)
        │           └── LHandRoot (10)
        │               ├── LHandThumb1 (11)
        │               │   └── LHandThumb2 (12)
        │               │       └── LHandThumb3 (13)
        │               │           └── LHandThumb4 (14)
        │               ├── LHandIndex1 (15)
        │               │   └── LHandIndex2 (16)
        │               │       └── LHandIndex3 (17)
        │               │           └── LHandIndex4 (18)
        │               ├── LHandMiddle1 (19)
        │               │   └── LHandMiddle2 (20)
        │               │       └── LHandMiddle3 (21)
        │               │           └── LHandMiddle4 (22)
        │               ├── LHandRing1 (23)
        │               │   └── LHandRing2 (24)
        │               │       └── LHandRing3 (25)
        │               │           └── LHandRing4 (26)
        │               └── LHandLittle1 (27)
        │                   └── LHandLittle2 (28)
        │                       └── LHandLittle3 (29)
        │                           └── LHandLittle4 (30)
        └── Nose (7)
            ├── REye (8)
            └── LEye (9)
        """
    @property
    def parents(self):
        return [-1,
                0, 1, 2,
                0, 4, 5,
                0, 7, 7,
                6,
                10, 11, 12, 13,
                10, 15, 16, 17,
                10, 19, 20, 21,
                10, 23, 24, 25,
                10, 27, 28, 29,
                3,
                31, 32, 33, 34,
                31, 36, 37, 38,
                31, 40, 41, 42,
                31, 44, 45, 46,
                31, 48, 49, 50]

    """
     joint_subset属性定义了一个关节子集，用于选择与任务相关的骨骼关键点。在这个例子中，它排除了一些不必要的关节（如鼻子和眼睛）。
     
        - range(7)：选择前 7 个关节（索引 0 到 6）
        - range(10, len(self.parents))：选择从索引10开始的所有关节，直到self.parents的长度。
        - np.r_  ：将两个范围合并为一个 NumPy 数组。
        
    返回值: 返回一个 NumPy 数组，表示选择的关节索引。
    """
    @property
    def joint_subset(self):
        # choose only relevant skeleton key-points (removed nose and eyes)
        return np.r_[range(7), range(10, len(self.parents))]

    @property
    def root(self):
        return 0

    @property
    def joint_names(self):
        return ['Neck',
                'RShoulder', 'RElbow', 'RWrist',
                'LShoulder', 'LElbow', 'LWrist',
                'Nose', 'REye', 'LEye',
                'LHandRoot',
                'LHandThumb1', 'LHandThumb2', 'LHandThumb3', 'LHandThumb4',
                'LHandIndex1', 'LHandIndex2', 'LHandIndex3', 'LHandIndex4',
                'LHandMiddle1', 'LHandMiddle2', 'LHandMiddle3', 'LHandMiddle4',
                'LHandRing1', 'LHandRing2', 'LHandRing3', 'LHandRing4',
                'LHandLittle1', 'LHandLittle2', 'LHandLittle3', 'LHandLittle4',
                'RHandRoot',
                'RHandThumb1', 'RHandThumb2', 'RHandThumb3', 'RHandThumb4',
                'RHandIndex1', 'RHandIndex2', 'RHandIndex3', 'RHandIndex4',
                'RHandMiddle1', 'RHandMiddle2', 'RHandMiddle3', 'RHandMiddle4',
                'RHandRing1', 'RHandRing2', 'RHandRing3', 'RHandRing4',
                'RHandLittle1', 'RHandLittle2', 'RHandLittle3', 'RHandLittle4'
                ]

    def fs(self, modality):
        return 15

    @property
    def h5_key(self):
        return 'pose'

