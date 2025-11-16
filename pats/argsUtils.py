import argparse
import itertools
# 用于安全地将字符串解析为 Python 数据结构（如列表、字典等）
from ast import literal_eval


def get_args_perm():
    # 创建一个ArgumentParser对象，用于定义和解析命令行参数
    parser = argparse.ArgumentParser()

    # Dataset Parameters
    # - nargs='+'： 表示该参数可以接受一个或多个值，返回一个列表
    parser.add_argument('-path2data', nargs='+', type=str, default=['pats/data/'],
                        help='path to data')
    parser.add_argument('-speaker', nargs='+', type=literal_eval, default=['bee'],
                        help='choose speaker or `all` to use all the speakers available')
    parser.add_argument('-modalities', nargs='+', type=literal_eval, default=[['pose/data', 'audio/log_mel_512']],
                        help='choose a set of modalities to be loaded by the dataloader')
    parser.add_argument('-split', nargs='+', type=literal_eval, default=[None],
                        help='(train,dev percentage) split of data. default=None')
    parser.add_argument('-batch_size', nargs='+', type=int, default=[32],
                        help='minibatch size. Use batch_size=1 when using time=0')
    parser.add_argument('-shuffle', nargs='+', type=int, default=[1],
                        help='shuffle the data after each epoch. default=True')
    parser.add_argument('-time', nargs='+', type=int, default=[4.3],
                        help='time for each sample (in seconds)')
    parser.add_argument('-fs_new', nargs='+', type=literal_eval, default=[[15, 15]],
                        help='subsample to the new frequency')

    args, unknown = parser.parse_known_args()  # unknown don't actually use

    # 将args对象转换为字典，键为参数名，值为参数值。
    args_dict = args.__dict__
    # sort by parameter name
    args_keys = sorted(args_dict)
    # 参数组合列表，每个元素是一个字典，表示一组参数
    # - *(args_dict[names] for names in args_keys)：将每个参数的所有可能值展开为多个可迭代对象。
    # - itertools.product(...)      ：生成这些可迭代对象的笛卡尔积。
    # - dict(zip(args_keys, prod))  ：将每个排列组合转换为字典，键为参数名，值为当前排列的值。
    args_perm = [dict(zip(args_keys, prod)) for prod in itertools.product(*(args_dict[names] for names in args_keys))]

    return args, args_perm


def arg_parse_n_loop(loop):
    args, args_perm = get_args_perm()

    for i, perm in enumerate(args_perm):
        # 将perm字典中的键值对更新到args.__dict__中
        args.__dict__.update(perm)
        print(args)
        loop(args, i)

if __name__ == '__main__':
    pass
    # get_args_perm()
