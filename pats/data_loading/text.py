import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data._utils.collate import default_collate

from common import Modality, MissingData


# 对一批数据进行填充（padding），使它们在指定维度上的长度一致
def pad(datasets, key, dim):
    sizes = []
    for data in datasets:
        data = data[key]
        sizes.append(data.shape[dim])
    max_length = max(sizes)
    new_datasets = []
    lengths = []
    for data in datasets:
        data = data[key]
        length = data.shape[dim]
        zero_shape = list(data.shape)
        zero_shape[dim] = max_length - length
        new_datasets.append(np.concatenate([data, np.zeros(zero_shape)], axis=dim))
        lengths.append(length)
    return default_collate(new_datasets), default_collate(lengths)


# 一个自定义的collate_fn，用于在 PyTorch 数据加载器中处理批量数据。它对指定键（pad_key）的数据进行填充，并将其他数据使用默认的collate_fn处理
def collate_fn_pad(batch, pad_key='text/meta', dim=0):
    if isinstance(batch[0], dict):
        data_dict = {}
        for key in batch[0]:
            if key in pad_key:
                padded_outs = pad(batch, key, dim=dim)
                if key == pad_key[-1]:  ## TODO hardcoded to use the last key which is text/token_duration
                    data_dict[key], data_dict['text/token_count'] = padded_outs[0], padded_outs[1]
                else:
                    data_dict[key] = padded_outs[0]
            else:
                data_dict[key] = default_collate([d[key] for d in batch])
        return data_dict
    else:
        return default_collate(batch)


class Text(Modality):
    def __init__(self, path2data='../data',
                 path2outdata='../data',
                 speaker='oliver',
                 preprocess_methods=['w2v'],
                 text_aligned=0):
        super(Text, self).__init__(path2data=path2data)
        self.path2data = path2data
        self.df = pd.read_csv(Path(self.path2data) / 'cmu_intervals_df.csv', dtype=object)
        self.df.loc[:, 'delta_time'] = self.df['delta_time'].apply(float)
        self.df.loc[:, 'interval_id'] = self.df['interval_id'].apply(str)
        self.path2outdata = path2outdata
        self.speaker = speaker
        self.preprocess_methods = preprocess_methods
        self.missing = MissingData(self.path2data)

        # list of word2-vec models
        # Word2Vec 模型：Word2Vec 是一种流行的词嵌入（word embedding）技术，用于将单词转换为高维向量。这些向量可以捕捉单词之间的语义关系，广泛应用于自然语言处理（NLP）任务
        self.w2v_models = []
        self.text_aligned = text_aligned

    def fs(self, modality):
        return 15

    @property
    def h5_key(self):
        return 'text'
