import h5py
import pathlib
import numpy as np
import pandas as pd

"""
HDF5 文件中的结构分为两种：
    - Group：类似文件夹，用于组织多个数据集或子组。
    - Dataset：实际存储数据的数组。
"""

# Check all the h5 file content
def inspect_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        stack = [(f, '  ')]  # (当前对象, 缩进层级)

        while stack:
            current, indent = stack.pop()

            # 打印当前对象的类型和名称
            print(f"{indent}- {type(current).__name__}: {current.name}")

            # 如果是 Group，继续遍历子对象
            if isinstance(current, h5py.Group):
                # 按逆序压栈，保证顺序遍历
                for key in reversed(current.keys()):
                    child = current[key]
                    stack.append((child, indent + '  '))
            # 如果是 Dataset，输出详细信息
            elif isinstance(current, h5py.Dataset):
                print(f"{indent}  Shape: {current.shape}")
                print(f"{indent}  Dtype: {current.dtype}")
                # 打印前1个样本
                if np.issubdtype(current.dtype, np.number):
                    sample = current[:1]  # 展平后更易阅读
                    print(f"{indent}  Sample data:\n{sample}")
                else:
                    print(f"{indent}  Sample data not shown (non-numeric dtype).")
            # 其他类型（如元组）跳过
            else:
                pass


# Check specific keys
def data_reader(file_path):
    # open .h5 file
    h5_file = h5py.File(file_path, 'r')
    # define .h5 file key
    # print(h5_file.keys())  # <KeysViewHDF5 ['audio', 'pose', 'text']>
    key = ['audio', 'pose', 'text']

    group = h5_file[key[1]]
    pose_dataset = group['data']

    print(f"Shape: {pose_dataset.shape}")
    print(f"Dtype: {pose_dataset.dtype}")

    if np.issubdtype(pose_dataset.dtype, np.number):
        sample = pose_dataset[:1]  # 展平后更易阅读
        print(f"Sample data:\n{sample}")
    else:
        print(f"Sample data not shown (non-numeric dtype).") smelly poos 


# Check whether the h5 file have log_mel_512 or log_mel_400
def check_log_mel(speaker):
    data_frame = pd.read_csv('/home/xukai/Documents/Motion_mimicking/Arm-and-Hand-Model/Code/Speech_to_Gesture/pats/data/cmu_intervals_df.csv')
    #data_frame = pd.read_csv('/home/myuser/workspace/pats/data/cmu_intervals_df.csv')  # [84289 rows x 8 columns]

    if isinstance(speaker, list):
        # 如果是列表，说明需要筛选出列column中值属于这个列表的行
        data_intervals = data_frame[data_frame['speaker'].isin(speaker)]
    else:
        data_intervals = data_frame[data_frame['speaker'] == speaker]

    get_intervals = lambda x: sorted(list(set(x['interval_id'].unique())))
    # get new train/dev/test intervals
    all_intervals = get_intervals(data_intervals)

    missing_list = []
    for interval in all_intervals:
        # Local check
        #base_path = "./processed/" + speaker
        # Local model training
        base_path = "./pats/data/processed/" + speaker
        # Remote GPU training
        #base_path = "/home/myuser/workspace/pats/data/processed/" + speaker
        file_path = f"{base_path}/{interval}.h5"

        try:
            with h5py.File(file_path, "r") as h5_file:
                # Check log_mel_512
                if 'audio/log_mel_512' not in h5_file:
                    # print(f"[MISSING] {interval} does NOT contain audio/log_mel_512")
                    missing_list.append(interval)
                if 'pose/data' not in h5_file:
                    # print(f"[MISSING] {interval} does NOT contain audio/log_mel_512")
                    missing_list.append(interval)
        except Exception as e:
            print(f"[ERROR] Failed to open {file_path}: {str(e)}")

    print(missing_list)

    return missing_list




if __name__ == '__main__':
    path = "./processed/almaram/120145.h5"  # 100905.h5  103699
    #path = "./missing_intervals.h5"

    #inspect_h5(path)

    check_log_mel("almaram")

    #data_reader(path)

"""
- File: /
  - Group: /audio
    - Dataset: /audio/log_mel_400
      Shape: (990, 64)
      Dtype: float64
      Sample data:
        [-4.01213539 -4.98865723 -5.13060738 -6.08112011 -5.73557633 -5.89106042
         -5.91147154 -6.27881099 -6.57486068 -5.96086752 -5.85459047 -6.31393508]
         
    - Dataset: /audio/silence
          Shape: (296,)
          Dtype: int64
          Sample data: [0 0 0]
          
  - Group: /pose
    - Dataset: /pose/data
      Shape: (149, 104)
      Dtype: float64
      Sample data:
        [ 377. -139. -200. -106.  135.  231.  219.   19.  -10.   39.  218.  240.
        212.  202.  182.  201.  192.  185.  179.  210.  207.  200.  193.  221.]
    - Dataset: /pose/normalize
      Shape: (149, 104)
      Dtype: float64
      Sample data:
        [ 377.         -166.56949815 -239.66834266 -127.02422161  161.7761313
          276.81693577  262.43683521   22.76849255  -11.98341713   46.73532682]

  - Group: /text
      - Dataset: /text/bert
        Shape: (478, 768)
        Dtype: float32
        Sample data:
        [ 2.88496524e-01  1.27960742e-01  1.51469335e-01  9.01399180e-04
        6.77053183e-02 -2.70105422e-01  2.27116212e-01 -1.02153614e-01]

- Group: /text/meta
      - Dataset: /text/meta/axis0
        Shape: (3,)
        Dtype: |S11
        Sample data not shown (non-numeric dtype).
      - Dataset: /text/meta/axis1
        Shape: (74,)
        Dtype: int64
        Sample data:
[0]
      - Dataset: /text/meta/block0_items
        Shape: (1,)
        Dtype: |S4
        Sample data not shown (non-numeric dtype).
      - Dataset: /text/meta/block0_values
        Shape: (1,)
        Dtype: object
        Sample data not shown (non-numeric dtype).
      - Dataset: /text/meta/block1_items
        Shape: (2,)
        Dtype: |S11
        Sample data not shown (non-numeric dtype).
      - Dataset: /text/meta/block1_values
        Shape: (74, 2)
        Dtype: float64
        Sample data:
[0. 0.]
    - Dataset: /text/tokens
      Shape: (478,)
      Dtype: int64
      Sample data:
[6366]
    - Dataset: /text/w2v
      Shape: (478, 300)
      Dtype: float64
      Sample data:
[-2.34375000e-01  1.60156250e-01 -8.10546875e-02  4.15039062e-02
 -4.12109375e-01 -1.45507812e-01  9.22851562e-02  4.93164062e-02

"""
