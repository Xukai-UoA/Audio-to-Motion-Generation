import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
- __file__                是一个内置变量，表示当前 Python 脚本的文件路径。
- os.path.abspath(path)   是一个函数，用于将路径转换为绝对路径。
- os.path.dirname(path)   是一个函数，用于获取路径中文件的目录部分。它会去掉路径中的文件名，只保留目录路径。
- sys.path.append(path)   是一个方法，用于将新的路径添加到 sys.path 的末尾。
"""
import h5py
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..argsUtils import *


# HDF5 File Operator
class HDF5:
    # HDF5类本身不需要在初始化时执行任何特定的逻辑
    def __init__(self):
        pass

    @staticmethod
    def h5_open(filename, mode):
        """
        • filename：HDF5 file path
        • mode    ：The way to open .h5 files（'r'、'w'、'a'）
        """
        # Make sure parent directory exist, if not then create one
        os.makedirs(Path(filename).parent, exist_ok=True)
        return h5py.File(filename, mode)

    @staticmethod
    def h5_close(h5):
        h5.close()

    @staticmethod
    def add_dataset(h5, key, data, exist_ok=False):
        if key in h5:
            # if key not exit, create a new dataset
            if exist_ok:
                warnings.warn('dataset {} not exists. Updating data...'.format(key))
                del h5[key]
                h5.create_dataset(key, data=data)
            # if the key already exit, skip.
            else:
                warnings.warn('dataset {} already exists. Skipping...'.format(key))
        else:
            h5.create_dataset(key, data=data)

    @staticmethod
    def update_dataset(h5, key, data):
        HDF5.add_dataset(h5, key, data, exist_ok=True)

    @staticmethod
    def load(filename, key):
        h5 = HDF5.h5_open(filename, 'r')
        data = h5[key]
        return data, h5

    @staticmethod
    def isDatasetInFile(filename, key):
        h5 = HDF5.h5_open(filename, 'r')
        if key in h5:
            h5.close()
            return True
        else:
            h5.close()
            return False

    # Create a .h5 file if it's not exist, else appends extra data to key
    @staticmethod
    def append(filename, key, data):
        h5 = HDF5.h5_open(filename, 'a')
        try:
            HDF5.update_dataset(h5, key, data)
        except:
            # pdb.set_trace()
            warnings.warn('could not update dataset {} with filename {}'.format(key, filename))
        HDF5.h5_close(h5)

    @staticmethod
    def del_dataset(h5, key):
        '''
        Delete a dataset(key) in h5 file
        - True: the key is found and deleted
        - False: The key not exit
        '''
        if key in h5:
            del h5[key]
            return True
        else:
            warnings.warn('Key not found. Skipping...')
            return False

    @staticmethod
    def add_key(base_key, sub_keys=[]):
        if isinstance(sub_keys, str):
            # 如果 sub_keys 是字符串，将其转换为列表。
            sub_keys = [sub_keys]

        sub_keys = '/'.join(sub_keys)
        new_key = (Path(base_key) / Path(sub_keys)).as_posix()
        return new_key

"""
Modality is inherited from HDF5 class

It adds processing logic for specific data modalities (such as audio, gestures, etc.) based on HDF5 file operations.
"""
class Modality(HDF5):
    def __init__(self, path2data='../data',
               path2outdata='../data',
               speaker='oliver',
               preprocess_methods=['log_mel']):
        # 根据当前类的继承链找到父类，调用父类的__init__方法，确保父类的初始化逻辑被执行。
        super(Modality, self).__init__()
        self.path2data = path2data
        # 指定 DataFrame 中所有列的数据类型为object，通常用于处理混合数据类型或字符串数据。
        self.df = pd.read_csv(Path(self.path2data) / 'cmu_intervals_df.csv', dtype=object)
        # 将 self.df 中的 delta_time 列的值转换为浮点数类型。
        # .loc 是 Pandas 提供的基于标签的索引器，用于选择数据。
        self.df.loc[:,'delta_time'] = self.df['delta_time'].apply(float)
        # 将 self.df 中的 interval_id 列的值转换为字符串类型。
        self.df.loc[:,'interval_id'] = self.df['interval_id'].apply(str)

        self.path2outdata = path2outdata
        self.speaker = speaker
        self.preprocess_methods = preprocess_methods

    def preprocess(self):
        # 定义一个预处理方法的占位符。留给子类去具体实现,它提示子类需要实现自己的预处理逻辑。
        raise NotImplementedError

    # 删除 HDF5 文件中指定的键（数据集）
    def del_keys(self, h5_key):
        if self.speaker != 'all':
            speakers = [self.speaker]
        else:
            speakers = self.speaker

        # 显示进度条
        # - desc='speakers'：进度条描述信息
        # - leave=False    ：控制进度条在循环结束后是否保留
        for speaker in tqdm(speakers, desc='speakers', leave=False):
            # 将信息打印在进度条上方
            tqdm.write('Speaker: {}'.format(speaker))
            # 从 DataFrame 中筛选出特定speaker的子集
            df_speaker = self.get_df_subset("speaker", speaker)
            interval_ids = df_speaker['interval_id'].unique()
            for preprocess_method in self.preprocess_methods:
                for interval_id in tqdm(interval_ids, desc='intervals'):
                    filename = Path(self.path2outdata) / 'processed' / speaker / '{}.h5'.format(interval_id)
                    key = self.add_key(h5_key[0], [preprocess_method])

                    ## delete dataset
                    h5 = self.h5_open(filename.as_posix(), 'a')
                    key_flag = self.del_dataset(h5, key)
                    if not key_flag:
                        break  ## ignore files of a speaker if the first file does not have ``key``
                    self.h5_close(h5)

    def get_df_subset(self, column, value):
        # 检查 value 是否是一个列表
        if isinstance(value, list):
            # 如果是列表，说明需要筛选出列column中值属于这个列表的行
            return self.df[self.df[column].isin(value)]
        else:
            return self.df[self.df[column] == value]

    @property
    def speakers(self):
        return [
            'oliver',  # TV sitting high_freq
            'jon',  # TV sitting
            'conan',  # TV standing high_freq
            'rock',  # lec sitting
            'chemistry',  # lec sitting
            'ellen',  # TV standing
            'almaram',  # eval sitting
            'angelica',  # eval sitting
            'seth',  # TV sitting low frequency
            'shelly',  # TV sitting
            'colbert',  # TV standing high_freq
            'corden',  # TV standing
            'fallon',  # TV standing
            'huckabee',  # TV standing
            'maher',  # TV standing
            'lec_cosmic',  # lec sitting
            'lec_evol',  # lec sitting
            'lec_hist',  # lec sitting
            'lec_law',  # lec sitting
            'minhaj',  # TV standing
            'ytch_charisma',  # yt sitting
            'ytch_dating',  # yt sitting
            'ytch_prof',  # yt sitting
            'bee',  # TV standing
            'noah'  # TV sitting
        ]

    # 遍历speakers列表，将每个说话者名称映射到一个唯一的索引。
    @property
    def inv_speakers(self):
        # dictionary
        dc = {}
        for i, speaker in enumerate(self.speakers):
            dc[speaker] = i
        return dc

    # 获取指定说话者的索引。
    def speaker_id(self, speaker):
        return self.inv_speakers[speaker]


"""
管理缺失数据的记录和存储。通过 HDF5 文件来保存和管理缺失数据的区间
"""
class MissingData(HDF5):
    def __init__(self, path2data):
        super(MissingData, self).__init__()
        self.path2file = Path(path2data)/'missing_intervals.h5'
        # 如果文件不存在，使用HDF5.h5_open以追加模式（'a'）打开文件，会自动创建文件。
        if not os.path.exists(self.path2file):
            h5 = HDF5.h5_open(self.path2file, 'a')
            HDF5.h5_close(h5)

        # set key name and missing data list
        self.key = 'intervals'
        self.missing_data_list = []

    # 将缺失的数据区间添加到临时列表中
    def append_interval(self, data):
        self.missing_data_list.append(data)
        warnings.warn('interval_id: {} not found.'.format(data))

    # Append missing data to the missing_intervals.h5 file
    def save_intervals(self, missing_data_list):
        # 定义一个可变长度字符串的数据类型，用于存储 HDF5 数据
        dt = h5py.special_dtype(vlen=str)
        # Check if the dataset in the .h5 file
        if HDF5.isDatasetInFile(self.path2file, self.key):
            # load the dataset
            intervals, h5 = HDF5.load(self.path2file, self.key)
            # convert to set format
            intervals = set(intervals[()])
            h5.close()
            # 将新传入的 missing_data_list（去除None值）与现有数据集合并
            intervals.update(set(missing_data_list) - {None})
            intervals = np.array(list(intervals), dtype=dt)
        else:
            intervals = np.array(list(set(missing_data_list) - {None}), dtype=dt)
        # 将 missing_data_list（去除None值）转换为 NumPy 数组
        HDF5.append(self.path2file, self.key, intervals)

    # Add new missing data from the current set to the missing_intervals.h5 file
    def save(self, missing_data_list):
        dt = h5py.special_dtype(vlen=str)
        intervals = np.array(list(set(missing_data_list) - {None}), dtype=dt)
        # don't consider whether the data already in the .h5 file
        HDF5.append(self.path2file, self.key, intervals)

    # 从 HDF5 文件中加载缺失数据区间
    def load_intervals(self):
        # 如果存在，加载数据并将其转换为集合
        if HDF5.isDatasetInFile(self.path2file, self.key):
            intervals, h5 = HDF5.load(self.path2file, self.key)
            intervals = set(intervals[()])
            h5.close()
        else:
            # 返回一个空集合
            intervals = set()
        return intervals

'''
Delete keys from the processed dataset stored in hdf5 files

preprocess_methods: list of preprocess_methods to delete
modalities: modality to delete. 
            deleting keys of different modalities must be deleted separately
            eg: 'audio', 'pose' etc.
'''
def delete_keys(args, exp_num):
    modality = Modality(args.path2data, args.path2outdata,
                        args.speaker, args.preprocess_methods)
    modality.del_keys(args.modalities)


if __name__ == '__main__':
    arg_parse_n_loop(delete_keys)