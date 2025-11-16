import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from pats.data_loading import Data_Loader
from speech2gesture_model import *
from normalization_tools import get_mean_std, get_mean_std_necksub


"""
This part of code is used to check the strange shape data
"""

# Set basic parameter
SPEAKER = 'oliver'
PATS_PATH = './pats/data'

# Save training model files
ROOT_PATH = './save/' + SPEAKER + '/'
MODEL_PATH_G = ROOT_PATH + 'gen'
MODEL_PATH_D = ROOT_PATH + 'dis'
LOSS_PATH = ROOT_PATH + 'loss.npy'

# Hyperparameter
lr = 10e-4
n_epochs = 6
lambda_d = 1.
lambda_gan = 1.


# Set log file
error_log_path = "training_dataset_errors.log"

# Loading data
common_kwargs = dict(path2data=PATS_PATH,
                     speaker=[SPEAKER],
                     modalities=['pose/data', 'audio/log_mel_512'],
                     fs_new=[15, 15],
                     batch_size=4,
                     window_hop=5)


if __name__ == '__main__':

    # Load speaker data
    dataloader = Data_Loader(**common_kwargs)

    dataloader = dataloader.train
    validate_dataloader = dataloader
    iterator = iter(validate_dataloader)

    total_batches = len(dataloader)
    error_batch_num = 0

    try:
        # enumerate(iterable, start=0)  start: 索引的起始值，默认为 0
        for batch_num, batch in enumerate(dataloader, 1):
            print(f"\n=== Processing Batch {batch_num}/{total_batches} ===")
            batch_error = next(iterator)

            #print("Batch:", batch)
            batch_dataset = ['pose/data', 'audio/log_mel_512', 'meta', 'style', 'idx']
            print("Pose:",batch[batch_dataset[0]].shape)    # Pose: torch.Size([4, 64, 104])
            print("Audio:", batch[batch_dataset[1]].shape)  # Audio: torch.Size([4, 64, 128])
            print("style:", batch[batch_dataset[3]].shape)  # style: torch.Size([4, 64])
            print("meta:", batch[batch_dataset[2]])
            print("idx:", batch[batch_dataset[4]])
            print("idx:", batch_error[batch_dataset[4]])
            print("-" * 20)
    except RuntimeError as e:
        print(f"Error in batch {batch_num+1}:")
        batch_error = next(iterator)
        batch_dataset = ['pose/data', 'audio/log_mel_512', 'meta', 'style', 'idx']
        print("Pose:", batch_error[batch_dataset[0]].shape)  # Pose: torch.Size([4, 64, 104])
        print("Audio:", batch_error[batch_dataset[1]].shape)  # Audio: torch.Size([4, 64, 128])
        print("style:", batch_error[batch_dataset[3]].shape)  # style: torch.Size([4, 64])
        print("meta:", batch_error[batch_dataset[2]])
        print("idx:", batch_error[batch_dataset[4]])
        print("-" * 20)






"""
    except RuntimeError as e:
        print(f"Error in batch {batch_num}:")
        print(e)
        print("Pose:",batch[batch_dataset[0]].shape)
        torch.save(batch, f"error_batch_{batch_num}.pt")  # 保存问题 Batch
        raise
"""