import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pats.data_loading import Data_Loader
from real_motion_model import *
from normalization_tools import get_mean_std, get_mean_std_necksub


class DynamicGANTraining:
    """动态GAN训练策略类"""

    def __init__(self, g_lr=5e-6, d_lr=10e-6):
        self.g_lr_initial = g_lr
        self.d_lr_initial = d_lr
        self.g_lr_current = g_lr
        self.d_lr_current = d_lr

        # record every batch loss
        self.d_loss_history = []
        self.g_loss_history = []

        # 动态调整参数
        self.d_strong_threshold = 0.20  # 判别器过强阈值
        self.g_weak_threshold = 0.80  # 生成器过弱阈值
        self.g_strong_threshold = 0.10

        # 训练频率控制
        self.d_train_freq = 1
        self.g_train_freq = 3
        self.min_d_freq = 1
        self.max_d_freq = 2
        self.min_g_freq = 2
        self.max_g_freq = 6

        # 标签平滑参数
        self.real_label_smooth = 0.98
        self.fake_label_smooth = 0.02
        self.dynamic_smooth = False

    def update_loss_history(self, d_loss, g_loss):
        """更新损失历史"""
        self.d_loss_history.append(d_loss)
        self.g_loss_history.append(g_loss)

        # 保持历史长度不超过100
        if len(self.d_loss_history) > 100:
            self.d_loss_history.pop(0)
            self.g_loss_history.pop(0)

    def get_recent_avg_loss(self, window=10):
        """获取最近的平均损失"""
        if len(self.d_loss_history) < window:
            return np.mean(self.d_loss_history), np.mean(self.g_loss_history)

        recent_d = np.mean(self.d_loss_history[-window:])
        recent_g = np.mean(self.g_loss_history[-window:])
        return recent_d, recent_g

    def should_train_discriminator(self):
        """判断是否应该训练判别器"""
        if len(self.d_loss_history) == 0:
            return True

        recent_d, recent_g = self.get_recent_avg_loss()

        # 如果判别器过强，减少训练
        if recent_d < self.d_strong_threshold and recent_g > self.g_weak_threshold:
            return False

        # 如果生成器太强，增加判别器训练
        if recent_d > 0.7 and recent_g < 0.4:
            return True

        return True

    def adjust_training_frequency(self, epoch):
        """动态调整训练频率"""
        if len(self.d_loss_history) < 10:
            return self.g_train_freq, self.d_train_freq

        recent_d, recent_g = self.get_recent_avg_loss()

        # 计算损失比值
        loss_ratio = recent_d / (recent_g + 1e-8)

        # 判别器过强
        if loss_ratio < 0.15 or recent_d < 0.1:
            # 减少判别器训练，增加生成器训练
            self.d_train_freq = max(1, self.d_train_freq - 1)
            self.g_train_freq = min(self.max_g_freq, self.g_train_freq + 1)
            print(f"判别器过强，调整频率: G={self.g_train_freq}, D={self.d_train_freq}")

        elif loss_ratio > 2.5:  # 生成器过强
            # 增加判别器训练，减少生成器训练
            self.d_train_freq = min(self.max_d_freq, self.d_train_freq + 1)
            self.g_train_freq = max(self.min_g_freq, self.g_train_freq - 1)
            print(f"生成器过强，调整频率: G={self.g_train_freq}, D={self.d_train_freq}")

        return self.g_train_freq, self.d_train_freq

    def adjust_learning_rates(self, optimizer_g, optimizer_d, epoch):
        """动态调整学习率"""
        if len(self.d_loss_history) < 10:
            for param_group in optimizer_g.param_groups:
                param_group['lr'] = self.g_lr_initial
            for param_group in optimizer_d.param_groups:
                param_group['lr'] = self.d_lr_initial

        else:
            recent_d, recent_g = self.get_recent_avg_loss()

            # 判别器过强时
            if recent_d < self.d_strong_threshold:
                # 降低判别器学习率，提高生成器学习率
                self.d_lr_current *= 0.9
                self.g_lr_current *= 1.05
                print(f"调整学习率: G_lr={self.g_lr_current:.2e}, D_lr={self.d_lr_current:.2e}")

            # 生成器过强时
            elif recent_d > 0.65 and recent_g < 0.3:
                # 提高判别器学习率，降低生成器学习率
                self.d_lr_current *= 1.05
                self.g_lr_current *= 0.9
                print(f"调整学习率: G_lr={self.g_lr_current:.2e}, D_lr={self.d_lr_current:.2e}")

            # 应用新的学习率
            for param_group in optimizer_g.param_groups:
                param_group['lr'] = self.g_lr_current
            for param_group in optimizer_d.param_groups:
                param_group['lr'] = self.d_lr_current


    # Generate dynamic smooth labels
    def get_smooth_labels(self, epoch, batch_size, device, is_real=True):
        # Noise annealing
        max_noise_std = 0.01
        min_noise_std = 0.002
        anneal_start_epoch = 0  # start anneal from epoch 0
        anneal_end_epoch = 60
        max_smooth_offset = 0.05  # extra smoothing at early stage

        if epoch < anneal_start_epoch:
            progress = 0.0
            base_noise_std = max_noise_std
        elif epoch > anneal_end_epoch:
            progress = 1.0
            base_noise_std = min_noise_std
        else:
            # Linear Annealing
            progress = (epoch - anneal_start_epoch) / (anneal_end_epoch - anneal_start_epoch)
            base_noise_std = max_noise_std - progress * (max_noise_std - min_noise_std)

        # Smooth value annealing： stronger smoothness in the early stage
        base_real_smooth = self.real_label_smooth - max_smooth_offset * (1 - progress) if is_real else self.fake_label_smooth + max_smooth_offset * (1 - progress)

        # generate labels
        recent_d, recent_g = self.get_recent_avg_loss() if len(self.d_loss_history) >= 10 else (0.5, 0.5)
        if is_real:
            smooth_val = base_real_smooth
            if self.dynamic_smooth and recent_d < self.d_strong_threshold:
                smooth_val = max(0.97, smooth_val - 0.1)  # 判别器过强，增加平滑
                noise_std = base_noise_std + 0.01  # 额外噪声
            else:
                noise_std = base_noise_std
            labels = torch.ones(batch_size, 4, device=device).fill_(smooth_val)
            labels = torch.clamp(labels + torch.normal(0, noise_std, labels.shape, device=device), 0.85, 1.0)
        else:
            smooth_val = base_real_smooth
            if self.dynamic_smooth and recent_g < self.g_strong_threshold:
                smooth_val = min(0.03, smooth_val + 0.1)  # 生成器过强，增加假标签平滑
                noise_std = base_noise_std + 0.01
            else:
                noise_std = base_noise_std
            labels = torch.zeros(batch_size, 4, device=device).fill_(smooth_val)
            labels = torch.clamp(labels + torch.normal(0, noise_std, labels.shape, device=device), 0.0, 0.15)

        return labels.requires_grad_(False)


# Set basic parameter
SPEAKER = 'multi_speaker'
PATS_PATH = './pats/data'

# Save training model files
ROOT_PATH = './save/' + SPEAKER + '/'
MODEL_PATH_G = ROOT_PATH + 'gen'
MODEL_PATH_D = ROOT_PATH + 'dis'
LOSS_PATH = ROOT_PATH + 'loss.npy'

# Hyperparameter
lr = 10e-4
n_epochs = 500
lambda_d = 1.
lambda_gan = 1.

# Loading data
common_kwargs = dict(path2data=PATS_PATH,
                     speaker=['oliver', 'noah', 'seth', 'shelly', 'ellen', 'angelica', 'almaram', 'chemistry'],
                     modalities=['pose/data', 'audio/log_mel_512'],
                     fs_new=[15, 15],  # Unify same sampling rate of modalities
                     batch_size=128,
                     window_hop=5)


def pos_to_motion(pose_batch):
    # shape = pose_batch.shape()
    # reshaped = pose.reshape(shape[0], shape[1], 2, -1)
    # diff = pose_batch[:, 1:] - pose_batch[:, :-1]
    diff = torch.diff(pose_batch, n=1, dim=1)
    return diff


if __name__ == '__main__':
    # Load speaker data
    dataloader = Data_Loader(**common_kwargs)

    # Initialize hardware
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"-------- Using GPU Device {torch.cuda.get_device_name(0)} to Train the model --------")
    cuda = True if torch.cuda.is_available() else False

    # Initialize the dynamic training strategy
    dynamic_trainer = DynamicGANTraining(g_lr=lr/2, d_lr=lr)

    # Define loss function
    motion_reg_loss = torch.nn.L1Loss()
    # Mean Squared Error Loss
    g_loss = torch.nn.MSELoss()
    d_loss1 = torch.nn.MSELoss()
    d_loss2 = torch.nn.MSELoss()

    # Initialize generator and discriminator
    generator = SelfAttention_G()
    discriminator = SelfAttention_D(out_channels=64)
    print("Generator and Discriminator model Initialized successfully ...")

    # Move the models and loss functions on GPU
    if cuda:
        generator.cuda()
        discriminator.cuda()
        motion_reg_loss.cuda()
        g_loss.cuda()
        d_loss1.cuda()
        d_loss2.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------
    g_loss_list = []
    d_loss_list = []

    pose_mean, pose_std = get_mean_std_necksub(dataloader)
    # Normalize the pose
    norm_pose_list = []
    for batch in dataloader.train:
        pose = batch['pose/data']  # torch.Size([129, 64, 104])
        pose = pose.reshape(pose.shape[0], pose.shape[1], 2, -1)  # torch.Size([129, 64, 2, 52])
        neck = pose[:, :, :, 0].reshape(pose.shape[0], pose.shape[1], 2, 1)
        pose = torch.sub(pose, neck)
        pose = pose.reshape(pose.shape[0], pose.shape[1], -1)  # torch.Size([129, 64, 104])
        pose = torch.sub(pose, pose_mean)
        pose = torch.div(pose, pose_std)
        norm_pose_list.append(pose)

    # 验证集归一化（使用训练集的均值和标准差）
    norm_pose_list_dev = []
    for batch_val in dataloader.dev:
        pose_val = batch_val['pose/data']
        pose_val = pose_val.reshape(pose_val.shape[0], pose_val.shape[1], 2, -1)
        neck_val = pose_val[:, :, :, 0].reshape(pose_val.shape[0], pose_val.shape[1], 2, 1)
        pose_val = torch.sub(pose_val, neck_val)
        pose_val = pose_val.reshape(pose_val.shape[0], pose_val.shape[1], -1)
        pose_val = torch.sub(pose_val, pose_mean)  # 关键：使用训练集的统计量
        pose_val = torch.div(pose_val, pose_std)
        norm_pose_list_dev.append(pose_val)

    # store val loss
    val_g_loss_list = []
    val_d_loss_list = []

    for epoch in range(n_epochs):
        # dynamic tune the lr and training times
        g_freq, d_freq = dynamic_trainer.adjust_training_frequency(epoch)
        dynamic_trainer.adjust_learning_rates(optimizer_G, optimizer_D, epoch)

        for i, batch in enumerate(dataloader.train, 0):
            #print("Batch %d strat training" % (i))
            audio = batch['audio/log_mel_512']  # torch.Size([129, 64, 128])
            audio = audio.to(device)
            audio = audio.type(torch.cuda.FloatTensor)
            real_pose = norm_pose_list[i]
            real_pose = real_pose.to(device)
            real_pose = real_pose.type(torch.cuda.FloatTensor)
            total_batches = len(dataloader.train)

            # Adversarial ground truths
            # valid = torch.ones(real_pose.size(0), 11, device=device).fill_(1.0).requires_grad_(False)
            valid = dynamic_trainer.get_smooth_labels(epoch, real_pose.size(0), device, is_real=True)
            fake = dynamic_trainer.get_smooth_labels(epoch, real_pose.size(0), device, is_real=False)

            # 生成真实和虚假motion（在循环外计算，避免重复计算）
            real_motion = pos_to_motion(real_pose)
            # -----------------
            #  Train Generator (dynamic)
            # -----------------
            for gen_step in range(g_freq):
                optimizer_G.zero_grad()

                # Using audio as generator input
                fake_pose, internal_losses = generator(audio, real_pose=real_pose)

                # Generate motions
                fake_motion = pos_to_motion(fake_pose)

                # # Generate accelerations
                # real_acceleration = pos_to_motion(real_motion)
                # fake_acceleration = pos_to_motion(fake_motion)

                # discriminator
                fake_d, _ = discriminator(fake_motion)

                # Loss measures generator's ability to fool the discriminator
                G_loss = motion_reg_loss(real_motion, fake_motion) + lambda_gan * g_loss(fake_d, valid)

                # Add internal losses
                for loss in internal_losses:
                    G_loss += loss

                G_loss.backward()
                optimizer_G.step()

            # ---------------------
            #  Train Discriminator (Dynamic)
            # ---------------------
            # Check whether the discriminator should be trained
            if dynamic_trainer.should_train_discriminator():

                for dis_step in range(d_freq):
                    optimizer_D.zero_grad()

                    # 固定生成器输出（防止梯度干扰）
                    with torch.no_grad():
                        fake_pose_detached, _ = generator(audio)
                        fake_motion_detached = pos_to_motion(fake_pose_detached)


                    fake_d, _ = discriminator(fake_motion_detached.detach())
                    real_d, _ = discriminator(real_motion)

                    # Measure discriminator's ability to classify real from generated samples
                    real_loss = d_loss1(real_d, valid)
                    fake_loss = d_loss2(fake_d, fake)
                    D_loss = real_loss + lambda_d * fake_loss

                    D_loss.backward()
                    optimizer_D.step()

            else:
                # Use the last time loss value, if skip the D training
                # D_loss = torch.tensor(d_loss_list[-1] if d_loss_list else 1.0)
                D_loss = torch.tensor(d_loss_list[-1])
                print(f"跳过判别器训练 - 判别器过强")

            # Update the loss history
            dynamic_trainer.update_loss_history(D_loss.item(), G_loss.item())

            recent_d, recent_g = dynamic_trainer.get_recent_avg_loss()
            if i % 200 == 199:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Recent D: %f] [Recent G: %f] [G_freq: %d] [D_freq: %d]"
                    % (epoch, n_epochs, i + 1, total_batches, D_loss.item(), G_loss.item(), recent_d, recent_g, g_freq, d_freq)
                )
                g_loss_list.append(G_loss.item())
                d_loss_list.append(D_loss.item())

        # ===================== 验证阶段 =====================
        # Change to evaluation mode
        generator.eval()
        discriminator.eval()

        val_g_loss = 0.0
        val_d_loss = 0.0
        val_bone_loss = 0.0
        val_angle_loss = 0.0
        val_steps = 0

        with torch.no_grad():  # 禁用梯度计算
            for j, batch_val in enumerate(dataloader.dev, 0):
                audio_val = batch_val['audio/log_mel_512'].to(device).type(torch.cuda.FloatTensor)
                real_pose_val = norm_pose_list_dev[j].to(device).type(torch.cuda.FloatTensor)

                # 生成器推理
                fake_pose_val, internal_losses = generator(audio_val, real_pose=real_pose_val)

                # 提取bone_loss和angle_loss
                bone_loss = internal_losses[0] if internal_losses else torch.tensor(0.0)  # 如果无损失，默认0
                angle_loss = internal_losses[1] if len(internal_losses) > 1 else torch.tensor(0.0)
                val_bone_loss += bone_loss.item()  # 累加
                val_angle_loss += angle_loss.item()

                real_motion_val = pos_to_motion(real_pose_val)
                fake_motion_val = pos_to_motion(fake_pose_val)

                # create dynamic batch size
                val_batch_size = real_pose_val.size(0)  # get current batch size
                valid_val = torch.ones(val_batch_size, 4, device=device).requires_grad_(False)
                fake_val = torch.zeros(val_batch_size, 4, device=device).requires_grad_(False)

                # 验证生成器损失
                motion_reg_loss_val = motion_reg_loss(real_motion_val, fake_motion_val)
                fake_d_val, _ = discriminator(fake_motion_val)
                g_loss_val = motion_reg_loss_val + lambda_gan * g_loss(fake_d_val, valid_val)

                # 验证判别器损失
                real_d_val, _ = discriminator(real_motion_val)
                fake_d_val, _ = discriminator(fake_motion_val.detach())
                real_loss_val = d_loss1(real_d_val, valid_val)
                fake_loss_val = d_loss2(fake_d_val, fake_val)
                d_loss_val = real_loss_val + lambda_d * fake_loss_val

                val_g_loss += g_loss_val.item()
                val_d_loss += d_loss_val.item()
                val_steps += 1

        # 计算平均验证损失
        val_g_loss /= val_steps
        val_d_loss /= val_steps
        val_g_loss_list.append(val_g_loss)
        val_d_loss_list.append(val_d_loss)

        # 计算平均bone_loss, angle_loss
        avg_bone_loss = val_bone_loss / val_steps
        avg_angle_loss = val_angle_loss / val_steps
        
        print(f"[Validation] Epoch {epoch}/{n_epochs} | G_loss: {val_g_loss:.4f} | D_loss: {val_d_loss:.4f} | Avg Bone Loss: {avg_bone_loss:.4f} | Avg Angle Loss: {avg_angle_loss:.4f}")

        # switch back to training mode
        generator.train()
        discriminator.train()

        # ===================== 保存策略 =====================
        # 创建父目录，如果已存在则不报错
        # MODEL_PATH_G = ROOT_PATH + 'gen'
        os.makedirs(MODEL_PATH_G, exist_ok=True)
        os.makedirs(MODEL_PATH_D, exist_ok=True)

        # 保存最佳模型（基于验证损失）
        if val_g_loss < min(val_g_loss_list[:-1], default=float('inf')):
            print(f"New best G model at epoch {epoch}, saving...")
            torch.save(generator.state_dict(), os.path.join(MODEL_PATH_G, 'Best_Gen'))

        # 常规保存（每个epoch）
        print('epoch ', epoch, ': ', 'saving generators')
        torch.save(generator.state_dict(), os.path.join(MODEL_PATH_G, 'epoch_'+str(epoch)))
        print('epoch ', epoch, ': ', 'saving discriminators')
        torch.save(discriminator.state_dict(), os.path.join(MODEL_PATH_D, 'epoch_'+str(epoch)))
        print('epoch ', epoch, ': ', 'saving losses')

        # ===================== 损失记录 =====================
        loss_dict = {
            'train_g': g_loss_list,
            'train_d': d_loss_list,
            'val_g': val_g_loss_list,
            'val_d': val_d_loss_list,
            'dynamic_stats': {
                'g_lr_history': [dynamic_trainer.g_lr_current],
                'd_lr_history': [dynamic_trainer.d_lr_current],
                'g_freq_history': [g_freq],
                'd_freq_history': [d_freq]
            }
        }
        torch.save(loss_dict, LOSS_PATH)  # 保存为字典格式

