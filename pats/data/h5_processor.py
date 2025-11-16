import h5py
import pathlib
import numpy as np
import pandas as pd

"""
HDF5 文件中的结构分为两种：
    - Group：类似文件夹，用于组织多个数据集或子组。
    - Dataset：实际存储数据的数组。
"""

""" ---------------------- Pose data format modify -------------------------- """
# [N, 6] 改为 [N, 2, 3]
def modify_dataset_shape(file_path):
    # 1. 以读写模式打开HDF5文件
    with h5py.File(file_path, 'r+') as h5_file:
        # 2. 定位目标Dataset
        pose_group = h5_file['pose']
        pose_dataset = pose_group['data']

        # 3. 读取数据到内存（必须复制一份，因为HDF5数据是只读的）
        data_in_memory = pose_dataset[:]
        print(f"Original dtype: {pose_dataset.dtype}, Original shape: {data_in_memory.shape}")

        # 4. 修改数据形状（示例：将数据从 [N, 6] 改为 [N, 2, 3]）
        new_shape = (data_in_memory.shape[0], 2, -1)
        data_reshaped = data_in_memory.reshape(new_shape)
        print(f"New dtype: {data_reshaped.dtype}, New shape: {data_reshaped.shape}")
        sample = data_reshaped[:1]  # 展平后更易阅读
        print(f"Sample data:\n{sample}")


        # 5. 删除原Dataset（必须先删除才能重建）
        #del h5_file['pose/data']
        del pose_group['data']

        # 6. 创建新Dataset（指定压缩、数据类型等可选参数）
        pose_group.create_dataset(
            'data',
            data=data_reshaped
            #dtype=data_reshaped.dtype,
            #shape=new_shape,
            # 可选参数：compression="gzip", compression_opts=9
        )

        print("HDF5文件已更新！")


# [N, 2, 3] to [N, 6]
def restore_dataset_shape(file_path):
    with h5py.File(file_path, 'r+') as h5_file:
        # 1. 定位目标 Dataset
        pose_dataset = h5_file['pose/data']

        # 2. 读取数据到内存
        data_in_mem = pose_dataset[:]
        print(f"Original restored shape (before modification): {data_in_mem.shape}")  # (302, 2, 52)
        print(f"Dtype: {data_in_mem.dtype}")

        # 3. 计算逆变换的形状：将 (N, 2, 52) → (N, 104)
        new_shape = (data_in_mem.shape[0], data_in_mem.shape[1] * data_in_mem.shape[2])
        print(f"New target shape: {new_shape}")  # (302, 2 * 52=104)

        # 4. 断言总元素数一致（可选但推荐）
        assert data_in_mem.size == np.prod(new_shape), \
            "Total elements mismatch! Check data integrity."

        # 5. 重塑数据
        data_reshaped = data_in_mem.reshape(new_shape)
        print(f"Reshaped data sample (52 → 104 columns):")
        print(data_reshaped[:1])  # 展示前两行数据

        # 6. 删除旧 Dataset 并创建新的
        del h5_file['pose/data']
        h5_file.create_dataset(
            'pose/data',
            data=data_reshaped
        )
        print("HDF5文件已更新！")


# -------------------- Process all the pose data in intervals --------------------------------
def restore_single_interval_shape(interval, speaker):
    base_path = "./processed/" + speaker
    file_path = base_path + '/' + interval + '.h5'
    h5_file = h5py.File(file_path, 'r+')
    # 1. 定位目标 Dataset
    if 'pose/data' not in h5_file:
        pass
    else:
        pose_dataset = h5_file['pose/data']

        # 2. 读取数据到内存
        data_in_mem = pose_dataset[:]
        if data_in_mem.shape == (data_in_mem.shape[0], 2, 52):
            # 3. 计算逆变换的形状：将 (N, 2, 52) → (N, 104)
            #new_shape = (data_in_mem.shape[0], data_in_mem.shape[1] * data_in_mem.shape[2])
            #print(f"New target shape: {new_shape}")  # (302, 2 * 52=104)
            N = data_in_mem.shape[0]
            new_shape = (N, 104)

            # 4. 提取 x 和 y 坐标
            x = data_in_mem[:, 0, :]  # (N, 52) 所有关节的 x 坐标
            y = data_in_mem[:, 1, :]  # (N, 52) 所有关节的 y 坐标

            # 5. 重塑数据为交替格式 [x1, y1, x2, y2, ..., x52, y52]
            data_reshaped = np.empty(new_shape)
            data_reshaped[:, 0::2] = x  # 偶数索引放 x
            data_reshaped[:, 1::2] = y  # 奇数索引放 y

            # 6. 重塑数据
            #data_reshaped = data_in_mem.reshape(new_shape)
            #print(f"Reshaped data sample (52 → 104 columns):")
            #print(data_reshaped[:1])  # 展示前两行数据

            # 6. 删除旧 Dataset 并创建新的
            del h5_file['pose/data']
            h5_file.create_dataset(
                'pose/data',
                data=data_reshaped
            )
            print(f"{interval} - HDF5文件已更新！")
        else:
            pass


def restore_all_intervals(speaker):
    data_frame = pd.read_csv('/home/xukai/Documents/Motion_mimicking/Arm-and-Hand-Model/Code/Speech_to_Gesture/pats/data/cmu_intervals_df.csv')  # [84289 rows x 8 columns]

    if isinstance(speaker, list):
        # 如果是列表，说明需要筛选出列column中值属于这个列表的行
        data_intervals = data_frame[data_frame['speaker'].isin(speaker)]
    else:
        data_intervals = data_frame[data_frame['speaker'] == speaker]

    get_intervals = lambda x: sorted(list(set(x['interval_id'].unique())))
    # get new train/dev/test intervals
    all_intervals = get_intervals(data_intervals)

    for interval in all_intervals:
        restore_single_interval_shape(interval, speaker)

    print("All intervals preprocess successfully ...")


if __name__ == '__main__':
    #path = "./processed/noah/cmu0000034062.h5"  # 100905.h5  103699
    #path = "./missing_intervals.h5"

    #modify_dataset_shape(path)
    #restore_dataset_shape(path)

    restore_all_intervals("chemistry")