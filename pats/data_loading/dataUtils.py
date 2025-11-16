import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import Modality
from nltk.corpus import stopwords
# BertTokenizer是Hugging Face的transformers库提供的工具，用于将文本分割成适合BERT模型的词汇单元（tokens）
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Sampler
import bisect
from functools import partial

from common import *
from skeleton import *
from audio import *
from text import *
from ..data import h5_loader


"""
Wrapper for Data

Parameter:
- path2data (str): path to dataset.
- speaker (str): speaker name. 
- modalities (list of str): list of modalities to wrap in the dataloader. These modalities are basically keys of the hdf5 files which were preprocessed earlier (default: ``['pose/data', 'audio/log_mel']``)
- fs_new (list, optional): new frequency of modalities, to which the data is up/down sampled to. (default: ``[15, 15]``).
- time (float, optional): time of snippet length in seconds. (default: ``4.3``).
- split (tuple or None, optional): split fraction of train and dev sets. Must add up to less than 1. If ``None``, use ``dataset`` columns in the master dataframe (loaded in self.df) to decide train, dev and test split. (default: ``None``).
- batch_size (int, optional): batch size of the dataloader. (default: ``100``).
- shuffle (boolean, optional): set to ``True`` to have the data reshuffled at every epoch (default: ``False``).
- num_workers (int, optional): set to values >0 to have more workers to load the data. argument for torch.utils.data.DataLoader. (default: ``15``). 

### Problem: 定义训练数据的采样器（Sampler） ###
"""
class Data_Loader(Modality):
    def __init__(self, path2data, speaker,
                 modalities=['pose/data', 'audio/log_mel_512'],
                 fs_new=[15, 15],
                 time=4.3,
                 split=None,      # True: need to split dataset
                 batch_size=100,
                 shuffle=True,
                 num_workers=0,
                 window_hop=0,
                 load_data=True,  # Load data samples for checking, default value: False
                 style_iters=0,
                 num_training_sample=None,
                 sample_all_styles=0,
                 repeat_text=1,
                 quantile_sample=None,
                 quantile_num_training_sample=None,
                 weighted=0,
                 filler=False,  # original code set to 0
                 num_training_iters=None):
        # super().__init__() 是一个用于调用父类构造函数的方法。它的作用是确保在子类中初始化时，父类的初始化逻辑也被正确执行。
        super().__init__(path2data=path2data)
        self.path2data = path2data
        self.speaker = speaker
        self.modalities = modalities
        self.fs_new = fs_new
        self.time = time
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.window_hop = window_hop
        self.load_data = load_data
        self.style_iters = style_iters  ## used to call a train sampler
        self.num_training_sample = num_training_sample
        self.sample_all_styles = sample_all_styles
        self.repeat_text = repeat_text
        self.quantile_sample = quantile_sample
        self.quantile_num_training_sample = quantile_num_training_sample
        self.weighted = weighted
        self.filler = filler
        self.num_training_iters = num_training_iters
        self.text_in_modalities = False
        self.missing = MissingData(self.path2data)

        # Check if the model use NLP tools
        if self.filler:
            # 使用stopwords.words('english')加载英文的停用词列表
            # 包含常见的停用词(如 "the"、"is"、"and" 等). 这些停用词通常在文本处理中被过滤掉，因为它们对语义贡献较小
            self.stopwords = stopwords.words('english')
            # 加载预训练的 BERT 分词器
            #  'bert-base-uncased'是一个预训练的 BERT 模型，不区分大小写。
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.stopwords, self.tokenizer = None, None

        # Check if text as an input
        for modality in self.modalities:
            if 'text' in modality:
                self.text_in_modalities = True

        print("-------- Processing and Loading Modality data --------")
        # Add all speakers in list
        if isinstance(self.speaker, str):
            self.speaker = [self.speaker]

        # Load all modality classes
        self.modality_classes = self._load_modality_classes()
        # {'pose/data': <skeleton.Skeleton2D object at 0x71035719ef30>, 'audio/log_mel_512': <audio.Audio object at 0x71038b244e60>}
        print("All modality Model Load successfully ...")

        # Load the master table
        # as_posix()方法将路径对象转换为一个字符串，其中路径分隔符统一为（/）
        self.df = pd.read_csv((Path(self.path2data) / 'cmu_intervals_df.csv').as_posix())  # [84289 rows x 8 columns]
        # load file with evil twins
        self.df = self.df._append(pd.read_csv((Path(self.path2data) / 'cmu_intervals_df_transforms.csv').as_posix()))  # [87153 rows x 8 columns]
        self.df.loc[:,'interval_id'] = self.df['interval_id'].apply(str)

        if speaker[0] == 'all':
            # Load all speakers name
            self.speaker = self.speakers
        else:
            pass

        # Get the specific speaker data rows list from .csv file for training
        self.df = self.get_df_subset('speaker', self.speaker)

        # Create Style Dictionary
        # 创建一个字典（self.speaker_dict），用于将不同的说话者（speaker）映射到唯一的整数索引（i）
        #   -  sp:i：这是字典中的键值对，sp是键，i是值。
        self.speaker_dict = {sp: i for i, sp in enumerate(self.speaker)}
        # Output: self.speaker_dict = {'oliver': 0, 'lec_evol': 1}

        # 检查 self.df.values 的长度是否大于零
        assert len(self.df.values), 'speaker `{}` not found'.format(speaker)

        # get train-dev-test datasets split
        # {'train': <pats.data_loading.dataUtils.ConcatDatasetIndex object at 0x7efc9d5e2b70>,
        #  'dev': <pats.data_loading.dataUtils.ConcatDatasetIndex object at 0x7efc9d505b20>,
        #  'test': <pats.data_loading.dataUtils.ConcatDatasetIndex object at 0x7efc9d5e3ad0>}
        self.datasets = self.tdt_split()
        print("Pre-processing train, validation, test dataset successfully ...")

        self.dataLoader_kwargs = {'batch_size': batch_size,
                                  'shuffle': shuffle,
                                  'num_workers': num_workers,
                                  'pin_memory': False}
        # if not repeat_text, do not repeat the word vectors to match the fs
        # if True: #not self.repeat_text:
        if self.text_in_modalities:
            ## always keep text/token_duration at the end to comply with the collate_fn_pad
            pad_keys = ['text/w2v', 'text/bert', 'text/filler', 'text/tokens', 'text/token_duration']
            self.dataLoader_kwargs.update({'collate_fn': partial(collate_fn_pad, pad_key=pad_keys, dim=0)})

        # 更新数据加载器（DataLoader）中所有数据集的时间窗口索引列表
        self.update_dataloaders(time, window_hop)

    # ----------------------------------------------------------------
    # ---------------------- Support Function ------------------------
    # ----------------------------------------------------------------

    def _load_modality_classes(self):
        modality_map = {}
        for modality in self.modalities:
            mod = modality.split('/')[0]
            modality_map[modality] = self.mod_map(mod)
        return modality_map


    # Called by _load_modality_classes(self)
    def mod_map(self, mod):
        mod_map_dict = {
            'pose': Skeleton2D,
            'audio': Audio,
            'text': Text
        }
        return mod_map_dict[mod](path2data=self.path2data, speaker=self.speaker)


    def tdt_split(self):
        # 检查 self.split 是否为“假值”（如 None、False、0、空字符串 '' 或空列表 [] 等）
        if not self.split:
            df_train = self.get_df_subset('dataset', 'train')
            df_dev = self.get_df_subset('dataset', 'dev')
            df_test = self.get_df_subset('dataset', 'test')
        else:
            # get the dataset length
            length = self.df.shape[0]
            # range of train dataset
            end_train = int(length * self.split[0])
            # range of validation dataset
            start_dev = end_train
            end_dev = int(start_dev + length * self.split[1])
            # range of test dataset
            start_test = end_dev

            # split dataset
            df_train = self.df[:end_train]
            df_dev = self.df[start_dev:end_dev]
            df_test = self.df[start_test:]

        # get missing intervals
        missing_intervals = self.missing.load_intervals()
        missing_intervals = self.get_transforms_missing_intervals(missing_intervals)

        """
        对输入的 Pandas DataFrame 或 Series 进行处理，提取其中的 interval_id 列的唯一值，并排除那些在 missing_intervals 中的值
        
        - lambda x                  : 定义一个匿名函数. x 通常是一个 Pandas DataFrame 或 Series
        - x['interval_id'].unique() : 提取 x 中的 interval_id 列，并调用 .unique() 返回一个数组，包含 interval_id 列中的所有唯一值
        - set(...)                  : 将 .unique() 返回的数组转换为集合（set）.集合是一种无序的数据结构，不允许重复元素，因此可以用来去重。
        - list(...)                 : 将差集操作的结果（一个集合）转换为列表（list）
        - sorted(...)               : 对列表中的值进行排序。
        """
        get_intervals = lambda x: sorted(list(set(x['interval_id'].unique()) - missing_intervals))
        # get new train/dev/test intervals
        second_missing_intervals = []
        for speaker in self.speaker:
            new_missing = h5_loader.check_log_mel(speaker)
            second_missing_intervals.extend(new_missing)

        # train_intervals = get_intervals(df_train)
        # 列表推导式: 遍历 get_intervals(df_train) 中的每个元素，并检查它是否不在 almaram_missing 中。如果不在，则保留该元素。
        train_intervals = [interval for interval in get_intervals(df_train) if interval not in second_missing_intervals]
        dev_intervals = [interval for interval in get_intervals(df_dev) if interval not in second_missing_intervals]
        test_intervals = [interval for interval in get_intervals(df_test) if interval not in second_missing_intervals]
        #test_intervals = get_intervals(df_test)

        """
        Test Function, Only load a few sample of the dataset
        
        - For official Training, change self.load_data to False.
        """
        if not self.load_data:
            train_intervals = train_intervals[:5]
            # print(train_intervals)
            # Output: ['100905', '100912', '100913', '100915', '100937']
            dev_intervals = dev_intervals[:5]
            # print(dev_intervals)
            test_intervals = test_intervals[:5]

        # update_train_intervals
        train_intervals, dev_intervals, test_intervals, train_intervals_dict = self.update_intervals(train_intervals, dev_intervals, test_intervals)

        # ConcatDataset 的主要作用是将多个 Dataset 对象合并成一个单一的 Dataset ，使得它们可以被统一地访问和迭代。
        # ConcatDataset: self.get_minidata_list()
        print("Loading train dataset ...")
        dataset_train = ConcatDatasetIndex(self.get_minidata_list(train_intervals))
        print("Loading validation dataset ...")
        dataset_dev = ConcatDatasetIndex(self.get_minidata_list(dev_intervals))
        print("Loading test dataset ...")
        dataset_test = ConcatDatasetIndex(self.get_minidata_list(test_intervals))

        # self.train_sampler = self.get_train_sampler(dataset_train, train_intervals_dict)

        return {'train': dataset_train,
                'dev': dataset_dev,
                'test': dataset_test}


    # Called by tdt_split(self)
    def get_transforms_missing_intervals(self, missing_intervals):
        transforms = []
        # if speaker = "speaker1|speaker2"
        for speaker in self.speaker:
            if '|' in speaker:
                transforms.append(speaker.split('|')[-1])

        transforms = sorted(list(set(transforms)))
        new_missing_intervals = set()
        for transform in transforms:
            for interval in missing_intervals:
                new_missing_intervals.update({'{}|{}'.format(interval, transform)})
        missing_intervals.update(new_missing_intervals)
        return missing_intervals


    # Called by tdt_split(self)
    """
    可以根据以下两种策略对数据进行处理：
    1. 按风格采样（sample_all_styles）：从每个风格（speaker）中采样固定数量的区间，或者采样全部区间。
    2. 交替风格采样（style_iters）：使用AlternateClassSampler对训练集进行采样，以确保不同风格的数据在训练过程中交替出现。
    """
    def update_intervals(self, train_intervals, dev_intervals, test_intervals):
        def subsample_intervals(x):
            temp = []
            for x_ in x:
                # 如果self.sample_all_styles > 0，则从每个风格中采样固定数量的区间（self.sample_all_styles）
                if self.sample_all_styles > 0:
                    temp.extend(x_[1][:self.sample_all_styles])
                # 如果self.sample_all_styles == -1，则采样全部区间
                elif self.sample_all_styles == -1:
                    temp.extend(x_[1])
            return temp

        # 将区间按风格分组并排序
        if self.sample_all_styles != 0:
            train_intervals_dict, train_intervals = self.order_intervals(train_intervals)
            dev_intervals_dict, dev_intervals = self.order_intervals(dev_intervals)
            test_intervals_dict, test_intervals = self.order_intervals(test_intervals)

            train_intervals = subsample_intervals(train_intervals_dict)
            dev_intervals = subsample_intervals(dev_intervals_dict)
            test_intervals = subsample_intervals(test_intervals_dict)
        # using AlternateClassSampler：对训练集进行采样，以确保不同风格的数据在训练过程中交替出现
        elif self.style_iters > 0:
            train_intervals_dict, train_intervals = self.order_intervals(train_intervals)
        else:
            train_intervals_dict = None
        return train_intervals, dev_intervals, test_intervals, train_intervals_dict


    # Called by update_intervals
    def order_intervals(self, intervals):
        # self.speaker_dict = {'oliver': 0, 'lec_evol': 1}
        # dictionary comprehension 字典推导式
        interval_dict = {i: [] for i in self.speaker_dict}  # {'oliver': [], 'lec_evol': []}
        for interval in intervals:
            interval_dict[self.getSpeaker(interval)].append(interval)
            print(interval_dict)
        intervals_dict = [(k, interval_dict[k]) for k in interval_dict]
        ordered_intervals = []
        for tup in intervals_dict:
            ordered_intervals += tup[1]
        return intervals_dict, ordered_intervals

    # Called by order_intervals()
    def getSpeaker(self, x):
        return self.get_df_subset('interval_id', x)['speaker'].values[0]

    # Called by tdt_split()
    def get_minidata_list(self, intervals):
        """
        根据给定的时间区间列表（intervals），为每个时间区间创建一个 MiniData 对象，并将这些对象存储在一个列表中。
            - tqdm(intervals): 将 intervals 包装起来，使其在迭代时显示进度条
        """
        return [MiniData(self.getPath2file(interval_id), style=self.getStyle(interval_id), **self.minidataKwargs) for interval_id in tqdm(intervals)]


    # Called by get_minidata_list()
    def getPath2file(self, x):
        return (Path(self.path2data) / 'processed' / self.getSpeaker(x) / str(x)).as_posix() + '.h5'


    # Called by get_minidata_list()
    def getStyle(self, interval_id):
        df_subset = self.get_df_subset('interval_id', interval_id)
        speaker = df_subset.speaker.iloc[0]
        try:
            style = self.speaker_dict[speaker]
        except:
            raise 'speaker style for {} not found'.format(speaker)
        return style


    # Called by get_minidata_list()
    @property
    def minidataKwargs(self):
        minidataKwargs = {'modalities': self.modalities,
                          'fs_new': self.fs_new,
                          'time': self.time,
                          'modality_classes': self.modality_classes,
                          'window_hop': self.window_hop,
                          'repeat_text': self.repeat_text,
                          'text_in_modalities': self.text_in_modalities,
                          'filler': self.filler,
                          'stopwords': self.stopwords,
                          'tokenizer': self.tokenizer}
        return minidataKwargs


    def update_dataloaders(self, time, window_hop):
        # update index list for all minidata
        # key: train, dev, test
        for key in self.datasets:
            # d_: MiniData object
            for d_ in self.datasets[key].datasets:
                d_.update_idx_list(time, window_hop)

        # renew data_loader parameters
        train_data_loader_kwargs = self.dataLoader_kwargs.copy()

        # self.train_sampler：如果定义了训练数据的采样器（Sampler），则将 shuffle 参数设置为 False，因为采样器会处理数据的随机性。
        #if self.train_sampler:
        #    train_data_loader_kwargs['shuffle'] = False

        self.train = DataLoader(ConcatDatasetIndex(self.datasets['train'].datasets), **train_data_loader_kwargs)
        # self.train = DataLoader(ConcatDatasetIndex(self.datasets['train'].datasets), sampler=self.train_sampler, **train_data_loader_kwargs)
        self.dev = DataLoader(ConcatDatasetIndex(self.datasets['dev'].datasets), **self.dataLoader_kwargs)
        self.test = DataLoader(ConcatDatasetIndex(self.datasets['test'].datasets), **self.dataLoader_kwargs)


    # Called by tdt_split(self)
    def get_train_sampler(self, dataset_train, train_intervals_dict):
        # Style iterations with AlternateClassSampler
        if self.style_iters > 0 and self.sample_all_styles == 0:
            train_sampler = self.get_alternate_class_sampler(dataset_train, train_intervals_dict, self.style_iters)
        # Sampler with lesser number of samples for few-shot learning.
        elif self.num_training_sample is not None:
            subset_idx = torch.randperm(len(dataset_train))
            train_sampler = torch.utils.data.SubsetRandomSampler(subset_idx[:self.num_training_sample])
        elif self.quantile_sample is not None:
            subset_idx, kind = self.get_quantile_sample(dataset_train, self.quantile_sample)
            if kind in ['above', 'tail']:
                train_sampler = torch.utils.data.SubsetRandomSampler(subset_idx)
            elif kind in ['rebalance'] and self.quantile_num_training_sample is not None:
                subset_idx = [torch.LongTensor(li) for li in subset_idx]
                train_sampler = BalanceClassSampler(subset_idx,
                                                    int(self.quantile_num_training_sample) * self.batch_size)
        elif self.weighted:
            train_sampler = torch.utils.data.WeightedRandomSampler([1] * len(dataset_train),
                                                                   self.weighted * self.batch_size)
        elif self.num_training_iters is not None:
            train_sampler = torch.utils.data.RandomSampler(dataset_train,
                                                           num_samples=self.num_training_iters * self.batch_size,
                                                           replacement=True)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)
            # train_sampler = None

        return train_sampler

    def get_alternate_class_sampler(self, dataset, intervals_dict, num_samples):
        class_count = []
        interval_offset = 0
        for tup in intervals_dict:
            count = 0
            for i in range(len(tup[1])):
                count += len(dataset.datasets[i + interval_offset])
            class_count.append(count)
            interval_offset += len(tup[1])

        return AlternateClassSampler(class_count, num_samples * self.batch_size)

    def get_quantile_sample(self, data, q):
        pose_modality = None
        for key in self.modalities:
            if 'pose' in key:
                pose_modality = key
                break
        assert pose_modality is not None, "can't find pose modality"
        if isinstance(q, float) or isinstance(q, int):
            if q < 1:
                kind = 'above'
            elif q > 1:
                kind = 'rebalance'
                q = int(q)
            else:
                raise 'q can\'t be 1 or negative'
        elif isinstance(q, list):
            assert np.array([q_ <= 1 and q_ >= 0 for q_ in q]).all(), 'quantile_sample is in [0,1]'
            assert len(q) == 2
            kind = 'tail'

        ## get distribution of velocities
        diff = lambda x, idx: x[1:, idx] - x[:-1, idx]
        vel = lambda x, idx: (((diff(x, idx)) ** 2).sum(-1) ** 0.5).mean()
        samples = []
        for batch in tqdm(data, desc='quantile_sample_calc'):
            pose = batch[pose_modality]
            pose = pose.reshape(pose.shape[0], 2, -1).transpose(0, 2, 1)
            samples.append(vel(pose, list(range(1, pose.shape[1]))))

        min_sample, max_sample = min(samples), max(samples)
        if kind == 'above':
            v0 = np.quantile(np.array(samples, dtype=np.float), q)
            print('above {} percentile'.format(v0))
        elif kind == 'tail':
            v0 = [np.quantile(np.array(samples, dtype=np.float), q[0]),
                  np.quantile(np.array(samples, dtype=np.float), q[1])]
            print('below {} and above {} percentile'.format(*v0))
        elif kind == 'rebalance':
            v0 = torch.arange(min_sample, max_sample + 1e-5, (max_sample - min_sample) / q)
            print('rebalaced data' + (' {}' * len(v0)).format(*v0))

        def in_subset(v, v0):
            if kind == 'above':
                return v > v0
            elif kind == 'tail':
                return (v > v0[1]) or (v < v0[0])
            elif kind == 'rebalance':
                starts, ends = v0[:-1], v0[1:]
                interval = ((starts <= v) * (v <= ends))
                if interval.any():
                    return interval.int().argmax().item()
                else:
                    raise 'incorrect interval'

        if kind in ['tail', 'above']:
            subset_idx = []
        elif kind == 'rebalance':
            subset_idx = [[] for _ in range(len(v0) - 1)]

        for i, batch in tqdm(enumerate(data), desc='quantile subset'):
            pose = batch[pose_modality]
            pose = pose.reshape(pose.shape[0], 2, -1).transpose(0, 2, 1)
            v = vel(pose, list(range(1, pose.shape[1])))
            if kind in ['tail', 'above']:
                if in_subset(v, v0):
                    subset_idx.append(i)
            elif kind == 'rebalance':
                subset_idx[in_subset(v, v0)].append(i)

        return subset_idx, kind


"""
加载和处理单个数据片段（一个时间窗口内的多模态数据，如姿势、音频和文本），并将其转换为适合模型训练的格式

- Dataset：PyTorch 的 Dataset 类，提供了数据加载和迭代的基础功能
- HDF5：之前定义的类，提供了 HDF5 文件操作的功能
"""
class MiniData(Dataset, HDF5):
    def __init__(self, path2h5, modalities, fs_new, time, modality_classes, window_hop, style=0, repeat_text=1,
                 text_in_modalities=False, filler=0, **kwargs):
        super(MiniData, self).__init__()
        self.path2h5 = path2h5
        self.modalities = modalities
        self.fs_new = fs_new
        self.time = time
        self.modality_classes = modality_classes
        self.window_hop = window_hop
        self.style = style
        self.repeat_text = repeat_text
        self.text_in_modalities = text_in_modalities
        self.filler = filler

        ## load modality shapes and data
        self.shapes = []
        self.data = []
        for modality in self.modalities:
            try:
                data, h5 = self.load(self.path2h5, modality)
            except:
                print(self.path2h5, modality)
                sys.exit(1)

            self.shapes.append(data.shape)
            self.data.append(data[()])
            h5.close()

        if self.text_in_modalities:
            try:
                self.text_df = pd.read_hdf(self.path2h5, key='text/meta')
            except:
                self.text_df = None
        if self.filler:
            self.stopwords = kwargs['stopwords']
            self.tokenizer = kwargs['tokenizer']

        ## create idx lists
        self.idx_start_list_dict = {}
        self.idx_end_list_dict = {}
        self.idx_interval_dict = {}

        self.update_idx_list(self.time, self.window_hop)


    # 根据给定的时间长度（time）和滑动步长（window_hop）计算每个模态（如姿势、音频等）的时间窗口索引。
    # 这些索引用于后续从数据中提取特定时间窗口的数据片段。
    def update_idx_list(self, time, window_hop=0):
        # pose/data 15 (148, 104)
        # audio/log_mel_512 15 (849, 128)
        for modality, fs_new, shape in zip(self.modalities, self.fs_new, self.shapes):
            # Call fs() function in Audio, Skeleton, Text Class in condition
            # 获取当前模态对应的类（如 Skeleton2D、Audio）, 调用该类的 fs 方法，获取当前模态的原始采样频率
            fs = self.modality_classes[modality].fs(modality)
            # 将时间长度（秒）乘以采样频率（Hz），得到时间窗口的大小（单位：采样点数）, 将结果转换为整数。
            window = int(time * fs)
            # 断言 window_hop 必须小于 window
            assert window_hop < window, 'hop size {} must be less than window size {}'.format(window_hop, int(time * fs))
            # 计算原始采样频率与新采样频率的比值。将结果四舍五入为整数
            fs_ratio = round(fs / fs_new)
            # 一个字典，用于存储每个模态的时间窗口索引间隔。
            self.idx_interval_dict[modality] = fs_ratio

            """
            根据是否设置滑动步长（window_hop）来计算时间窗口的起始索引。它决定了如何在数据的时间维度上划分窗口
            
            :parameter
                - shape[0]         ：当前模态的数据长度（时间维度）。
                - shape[0] - window：计算最后一个有效窗口的起始索引。
                - int(window)      ：窗口大小（单位：采样点数）。
                - int(window_hop * fs_ratio)：滑动步长（单位：采样点数）。
                - np.r_[range(...)]         ：将 range 的结果转换为 NumPy 数组，包含所有时间窗口的起始索引。
            """
            if not window_hop:
                # 如果 shape[0] = 1000，window = 100，则 time_splits 为 [0, 100, 200, ..., 900]
                time_splits = np.r_[range(0, shape[0] - window, int(window))]
            else:
                # 如果 shape[0] = 1000，window = 100，window_hop = 20，fs_ratio = 2，则 time_splits 为 [0, 40, 80, ..., 960]
                time_splits = np.r_[range(0, shape[0] - window, int(window_hop * fs_ratio))]

            # [:] 是一个切片操作，用于复制整个数组
            self.idx_start_list_dict[modality] = time_splits[:]
            self.idx_end_list_dict[modality] = time_splits + window

        # len_starts = [len(self.idx_start_list_dict[modality]) for modality in self.idx_start_list_dict]
        # raise len_starts[0] == len_starts[1], 'number of idxes are not consistent in file {}'.format(self.path2h5)


    # __len__：当使用 len(obj) 获取对象长度时，会调用该方法。
    def __len__(self):
        return min([len(self.idx_start_list_dict[modality]) for modality in self.modalities])
        # return len(self.idx_start_list_dict[self.modalities[0]])

    """
    __getitem__ 方法被定义为返回特定索引对应的数据项。
    
    当 MiniData 类的实例被用在需要索引访问的上下文中时，比如在一个循环中迭代该实例，或者直接通过索引获取数据时，__getitem__ 方法就会被自动调用。
    
    在Python中，__getitem__ 是一个特殊方法，它允许类的实例在被索引时自动调用。具体来说，当一个对象被当作序列或映射来访问时，比如使用 obj[index] 的语法，Python会自动调用该对象的 __getitem__ 方法，并将索引值作为参数传递给它。
    """
    def __getitem__(self, idx):
        item = {}
        ## args.modalities = ['pose/normalize', 'text/w2v']
        for i, modality in enumerate(self.modalities):
            ## read from loaded data
            data = self.data[i]

            ## open h5 file
            # data, h5 = self.load(self.path2h5, modality)

            start = self.idx_start_list_dict[modality][idx]
            end = self.idx_end_list_dict[modality][idx]
            interval = self.idx_interval_dict[modality]

            item[modality] = data[start:end:interval].astype(np.float64)
            start_time = data[0:start:interval].shape[0] / self.fs_new[-1]

            if 'text' in modality:
                vec = item[modality]
                indices = [0]  ## starts in 64 frames
                if self.text_df is None or modality == 'text/tokens':  ## to be used with self.repeat_text = 0
                    for t in range(1, vec.shape[0]):
                        if (vec[t] - vec[indices[-1]]).sum() != 0:
                            indices.append(t)
                else:
                    text_df_ = self.text_df[(start <= self.text_df['end_frame']) & (end > self.text_df['start_frame'])]
                    starts_ = text_df_['start_frame'].values - start
                    starts_[0] = 0
                    indices = list(starts_.astype(np.int))
                if not self.repeat_text:
                    item.update({modality: vec[indices]})  ## if self.repeat_text == 0, update the text modality

                ## add filler masks
                if self.filler:
                    filler = np.zeros((len(indices),))
                    if self.text_df is None:
                        pass  ## if text_df is not available, assume no word is filler
                    else:
                        words = self.text_df[
                            (start <= self.text_df['end_frame']) & (end > self.text_df['start_frame'])].Word.values
                        words = [word.lower() for word in words]
                        if 'bert' in modality or 'tokens' in modality:
                            words = self.tokenizer.tokenize(' '.join(words))

                        for i, word in enumerate(words[:len(indices)]):
                            if word in self.stopwords:
                                filler[i] = 1
                    if self.repeat_text:
                        filler_ = np.zeros((vec.shape[0],))
                        end_indices = indices[1:] + [vec.shape[0]]
                        for i, (st, en) in enumerate(zip(indices, end_indices)):
                            filler_[st:en] = filler[i]
                        filler = filler_
                    item.update({'text/filler': filler})

                ## duration of each word
                indices_arr = np.array(indices).astype(np.int)
                length_word = np.zeros_like(indices_arr)
                length_word[:-1] = indices_arr[1:] - indices_arr[:-1]
                duration = (end - start) / interval
                length_word[-1] = duration - indices_arr[-1]
                item.update({'text/token_duration': length_word.astype(np.int)})

            ## close h5 file
            # h5.close()

        # start and end times of audio in the interval
        # start_time = self.fs_new[-1] * data[0:start:interval].shape[0]
        duration = item[self.modalities[0]].shape[0] / self.fs_new[-1]
        # duration = ((end-start)/interval)/self.fs_new[-1]
        end_time = start_time + duration

        item.update({'meta': {'interval_id': Path(self.path2h5).stem,
                              'start': start_time,
                              'end': end_time,
                              'idx': idx}})

        item['style'] = np.zeros(item[self.modalities[0]].shape[0]) + self.style

        return item

    def close_h5_files(self, files):
        for h5 in files:
            h5.close()

"""
将多个 MiniData 对象合并成一个大的 Dataset，使得这些数据集可以被统一地访问和迭代。

    - Input [<pats.data_loading.dataUtils.MiniData object at 0x7a0cc242db80>, <pats.data_loading.dataUtils.MiniData object at 0x7a0cd6f34bc0>]
"""
# Input [<pats.data_loading.dataUtils.MiniData object at 0x7a0cc242db80>, <pats.data_loading.dataUtils.MiniData object at 0x7a0cd6f34bc0>]
class ConcatDatasetIndex(ConcatDataset):
    def __init__(self, datasets):
      super().__init__(datasets)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        batch = self.datasets[dataset_idx][sample_idx]
        if isinstance(batch, dict):
            batch.update({'idx': idx})
        return batch


class AlternateClassSampler(Sampler):
    def __init__(self, class_count, num_samples, replacement=True):
        self.num_samples_per_class = num_samples // len(class_count)
        self.num_samples = self.num_samples_per_class * len(class_count)
        self.class_count = class_count
        self.starts = [0]
        self.ends = []
        for counts in self.class_count:
            self.starts.append(self.starts[-1] + counts)
            self.ends.append(self.starts[-1])
        self.starts = self.starts[:-1]

    """
    __iter__ 和 __next__：当对象被用在 for 循环中，或者需要迭代对象时，会调用 __iter__ 方法返回一个迭代器对象，然后在每次迭代时调用 __next__ 方法获取下一个值。
    """
    def __iter__(self):
        return iter(torch.stack([torch.randint(start, end, size=(self.num_samples_per_class,)) for start, end in
                                 zip(self.starts, self.ends)], dim=1).view(-1).tolist())

    def __len__(self):
        return self.num_samples


class BalanceClassSampler(Sampler):
    def __init__(self, classes, num_samples, replacement=True):
        self.classes = classes
        self.update_classes()
        self.num_samples_per_class = num_samples // len(self.classes)
        self.num_samples = self.num_samples_per_class * len(self.classes)

    def update_classes(self):
        cl_list = []
        for cl in self.classes:
            if cl.shape[0] > 0:
                cl_list.append(cl)
        self.classes = cl_list

    def __iter__(self):
        return iter(torch.stack(
            [class_idx[torch.randint(0, len(class_idx), size=(self.num_samples_per_class,))] for class_idx in
             self.classes], dim=1).view(-1).tolist())

    def __len__(self):
        return self.num_samples