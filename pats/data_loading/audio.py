import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import webrtcvad
import pdb

from common import Modality, MissingData


"""
Audio类，用于处理音频数据。实现了以下功能：

1. 初始化：加载音频数据的元信息，并初始化路径、说话者和预处理方法。
2. 特征提取：
  - log_mel_512和log_mel_400：计算对数梅尔频谱。
  - silence：检测音频中的静音段。
3. 采样频率映射：通过fs_map和fs方法管理不同预处理方法的采样频率。
4. HDF5键名：定义音频数据在 HDF5 文件中的键名。
"""
class Audio(Modality):
    def __init__(self, path2data='../data',
               path2outdata='../data',
               speaker='oliver',
               preprocess_methods=['log_mel_512']):
        super(Audio, self).__init__(path2data=path2data)
        self.path2data = path2data
        self.df = pd.read_csv(Path(self.path2data) / 'cmu_intervals_df.csv', dtype=object)
        self.df.loc[:, 'delta_time'] = self.df['delta_time'].apply(float)
        self.df.loc[:, 'interval_id'] = self.df['interval_id'].apply(str)
        self.path2outdata = path2outdata
        self.speaker = speaker
        self.preprocess_methods = preprocess_methods
        self.missing = MissingData(self.path2data)

    """
    计算音频信号的对数梅尔频谱（Log-Mel Spectrogram），使用 512 的 Hop Length
    
    parameter
        - y  ：音频信号时间序列（通常是 NumPy 数组）
        - sr ：音频的采样率（采样频率）
        - eps：一个小的常数，用于避免对零取对数。
        - n_fft=2048    ：FFT 的窗口大小，决定了频谱的频率分辨率。
        - hop_length=512：每次移动的样本数，决定了频谱的时间分辨率。
        
    Logic:
        - 使用 librosa.feature.melspectrogram 计算梅尔频谱
        - 使用一个小的掩码（mask）将频谱中的零值替换为一个小的常数（eps）
        - 对频谱取对数，并转置结果。
        
    返回值: 返回对数梅尔频谱的转置版本
    """
    def log_mel_512(self, y, sr, eps=1e-10):
        # calculate melspectrogram, output is a 2-D array
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)

        # 标记频谱中的零值
        #   - spec == 0：创建一个布尔数组，表示频谱中哪些位置的值为零
        #   - .astype(np.float)：将布尔数组转换为浮点数组，True转换为1.0，False转换为0.0
        mask = (spec==0).astype(np.float)

        # 将零值替换为eps，非零值保持不变
        #   - mask * eps：将mask中的零值位置替换为小常数eps
        #   - (1 - mask)：创建一个与mask形状相同的数组，其中零值位置为1.0，非零值位置为0.0
        #   - (1 - mask) * spec：保留spec中的非零值
        spec = mask * eps + (1 - mask) * spec

        # np.log(spec)    ：对处理后的频谱取自然对数
        # .transpose(1, 0)：将频谱的形状从(n_mels, t)转置为(t, n_mels)，即时间步数在前，频带数量在后
        return np.log(spec).transpose(1, 0)

    """
    计算音频信号的对数梅尔频谱，使用 400 的 Hop Length，并将音频重采样到 16kHz。
    
    Logic:
        1. 将音频信号重采样到 16kHz。
        2. 使用 librosa.core.stft 计算短时傅里叶变换（STFT）。
        3. 计算梅尔频谱，并使用掩码避免对零取对数。
        4. 返回对数梅尔频谱的转置版本。
    """
    def log_mel_400(self, y, sr, eps=1e-6):
        # 将音频信号从原始采样率（orig_sr）重采样到目标采样率（target_sr）
        y = librosa.core.resample(y, orig_sr=sr, target_sr=16000)  ## resampling to 16k Hz
        # pdb.set_trace()

        # define parameter
        sr = 16000        # 重采样后的采样率（16kHz）
        n_fft = 512       # FFT 的窗口大小（512）
        hop_length = 160  # 每次移动的样本数（160）
        win_length = 400  # 窗口长度（400）

        # 计算短时傅里叶变换(STFT), 输出stft是一个复数数组，表示音频信号在频域中的表示
        stft = librosa.core.stft(y=y.reshape((-1)),   # 将音频信号y转换为一维数组（确保输入是正确的形状）
                              n_fft=n_fft,            # FFT 的窗口大小
                              hop_length=hop_length,  # 每次移动的样本数
                              win_length=win_length,  # 窗口长度
                              center=False)           # 不进行时间轴上的中心化处理

        # 提取音频信号的频谱幅度信息
        #   - np.abs：计算 STFT 结果的幅度（模），将复数转换为非负实数
        stft = np.abs(stft)

        # 计算梅尔频谱, 输出spec是一个二维数组，表示梅尔频谱
        spec = librosa.feature.melspectrogram(S=stft,
                                              sr=sr,
                                              n_fft=n_fft,
                                              hop_length=hop_length,
                                              power=1,      # 表示频谱的功率为 1（即幅度频谱）
                                              n_mels=64,    # 梅尔频带的数量
                                              fmin=125.0,   # 最低频率（125 Hz）
                                              fmax=7500.0,  # 最高频率（7500 Hz）
                                              norm=None)    # 不进行归一化处理
        mask = (spec == 0).astype(np.float)
        spec = mask * eps + (1 - mask) * spec
        return np.log(spec).transpose(1, 0)


    """
    检测音频信号中的静音段（silence detection）。它使用了 WebRTC VAD（Voice Activity Detection）库来判断音频片段是否包含语音。
    
    - VAD 用于检测音频片段是否包含语音. 灵敏度越高，VAD 对语音的检测越敏感
    - 这里将音频重采样到 16kHz，因为 VAD 通常在 16kHz 下工作效果最佳
    """
    def silence(self, y, sr, eps=1e-6):
        # 创建一个 VAD 对象，灵敏度设置为 3（最敏感级别）
        vad = webrtcvad.Vad(3)
        # 将音频信号从原始采样率（orig_sr）重采样到目标采样率（target_sr）
        y = librosa.core.resample(y, orig_sr=sr, target_sr=16000)  ## resampling to 16k Hz
        # pdb.set_trace()
        fs_old = 16000  # 重采样后的采样率（16kHz）
        fs_new = 15     # 目标采样率（15Hz），表示每秒检测 15 次
        # 生成一个时间范围数组，表示每个检测窗口的起始点
        ranges = np.arange(0, y.shape[0], fs_old / fs_new)
        # 每个检测窗口的起始和结束点
        starts = ranges[0:-1]  # 取数组从第一个元素到倒数第二个元素的所有值
        ends = ranges[1:]      # 取数组从第二个元素到最后一个元素的所有值。

        is_speeches = []
        # zip() 将多个可迭代对象（如列表、元组等）打包成一个迭代器，每次迭代返回一个元组
        # 生成(starts[0], ends[0]), (starts[1], ends[1]), ...
        for start, end in zip(starts, ends):
            # 在每个检测窗口内，进一步划分更细的时间范围（每秒 100 次检测）
            sub_ranges = np.arange(start, end, fs_old / 100)
            sub_is_speech = []
            for s, e, in zip(sub_ranges[:-1], sub_ranges[1:]):
                try:
                    # vad.is_speech(y[int(s):int(e)].tobytes(), fs_old)
                    #   - vad.is_speech(...)：调用 VAD 检测该子窗口是否包含语音
                    #   - y[int(s):int(e)]  ：提取子窗口内的音频数据
                    #   - tobytes()         ：将音频数据转换为字节序列，因为 VAD 需要字节输入
                    sub_is_speech.append(vad.is_speech(y[int(s):int(e)].tobytes(), fs_old))
                except:
                    # 它会在代码中插入一个断点。当程序执行到这行代码时，会暂停运行，并进入交互式调试模式。
                    pdb.set_trace()

            """
            np.array(sub_is_speech, dtype=np.int).mean()
                - 将布尔值（True/False）转换为整数（1/0），并计算平均值
                - 如果平均值小于或等于 0.5，表示该检测窗口内大部分时间是静音的
                
            is_speeches.append(int(...))
                - 将检测结果（1表示有语音，0 表示静音）存储到is_speeches
            """
            is_speeches.append(int(np.array(sub_is_speech, dtype=np.int).mean() <= 0.5))
            # 在每个检测窗口后追加一个静音标志（0），可能是为了对齐时间轴
            is_speeches.append(0)
        return np.array(is_speeches, dtype=np.int)

    @property
    def fs_map(self):
        return {
            'log_mel_512': int(45.6 * 1000 / 512),  # int(44.1*1000/512) #112 #round(22.5*1000/512)
            'log_mel_400': int(16.52 * 1000 / 160),
            'silence': 15
        }

    # split('/')[-1]   用于提取字符串的最后一部分，即预处理方法的名称
    # 例如，modality是'audio/log_mel_512'，则提取出'log_mel_512'
    def fs(self, modality):
        modality = modality.split('/')[-1]
        return self.fs_map[modality]

    @property
    def h5_key(self):
        return 'audio'
