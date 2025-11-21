import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练

from pats.data_loading import Data_Loader
from real_motion_model import *
from normalization_tools import get_mean_std, get_mean_std_necksub


class CurriculumGANTraining:
    """渐进式/课程学习GAN训练策略类 - 继承并扩展动态训练"""

    def __init__(self, g_lr=5e-6, d_lr=10e-6):
        self.g_lr_initial = g_lr
        self.d_lr_initial = d_lr
        self.g_lr_current = g_lr
        self.d_lr_current = d_lr

        # 课程学习参数
        self.curriculum_epochs = 50  # 前50个epoch使用渐进式策略
        self.warmup_epochs = 10  # 前10个epoch为预热阶段

        # 渐进式损失权重
        self.initial_detail_weight = 0.3  # 初始细节损失权重（低）
        self.final_detail_weight = 1.0    # 最终细节损失权重（高）

        # 渐进式物理约束权重
        self.initial_physics_weight = 0.5
        self.final_physics_weight = 2.0

        # record every batch loss
        self.d_loss_history = []
        self.g_loss_history = []

        # 动态调整参数 - 融合基准分支的严格策略
        self.d_strong_threshold = 0.08  # 基准分支的严格阈值 (从0.25→0.08)
        self.g_weak_threshold = 0.90    # 基准分支的严格阈值 (从0.75→0.90)
        self.g_strong_threshold = 0.03  # 基准分支的严格阈值 (从0.12→0.03)

        # 训练频率控制 - 平衡两种策略
        self.d_train_freq = 1
        self.g_train_freq = 4  # 初始增加生成器训练频率
        self.min_d_freq = 1
        self.max_d_freq = 2  # 基准分支的限制（从3→2）
        self.min_g_freq = 3  # 提高最小生成器训练频率
        self.max_g_freq = 8  # 允许更高的生成器训练频率

        # 判别器强度控制 - 使用基准分支的强制训练机制
        self.skip_d_counter = 0  # 跳过判别器训练的次数
        self.max_skip_count = 20  # 基准分支的值（从5→20次）

        # 标签平滑参数 - 融合基准分支的强平滑策略
        self.real_label_smooth = 0.90  # 基准分支的强平滑 (从0.98→0.90)
        self.fake_label_smooth = 0.10  # 基准分支的强平滑 (从0.02→0.10)
        self.dynamic_smooth = True     # 启用动态平滑调整（基准分支）

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
        """
        判断是否应该训练判别器（融合版）
        使用基准分支的严格策略 + 课程学习
        """
        if len(self.d_loss_history) == 0:
            self.skip_d_counter = 0
            return True

        recent_d, recent_g = self.get_recent_avg_loss()

        # 强制训练机制（基准分支）：如果连续跳过次数过多，必须训练一次
        if self.skip_d_counter >= self.max_skip_count:
            print(f"强制训练判别器 (连续跳过{self.skip_d_counter}次)")
            self.skip_d_counter = 0
            return True

        # 判别器过强的条件（基准分支的严格阈值）：D_loss < 0.08 且 G_loss > 0.90
        if recent_d < self.d_strong_threshold and recent_g > self.g_weak_threshold:
            self.skip_d_counter += 1
            return False

        # 如果生成器太强，增加判别器训练
        if recent_d > 0.7 and recent_g < 0.4:
            self.skip_d_counter = 0
            return True

        # 默认训练，重置跳过计数
        self.skip_d_counter = 0
        return True

    def adjust_training_frequency(self, epoch):
        """
        动态调整训练频率（融合版）
        使用基准分支的激进策略 + 课程学习
        """
        if len(self.d_loss_history) < 10:
            return self.g_train_freq, self.d_train_freq

        recent_d, recent_g = self.get_recent_avg_loss()

        # 计算损失比值
        loss_ratio = recent_d / (recent_g + 1e-8)

        # 判别器过强 - 使用基准分支的更宽松触发条件
        if loss_ratio < 0.3 or recent_d < 0.2:
            # 减少判别器训练，大幅增加生成器训练（基准分支策略）
            self.d_train_freq = max(self.min_d_freq, self.d_train_freq - 1)
            self.g_train_freq = min(self.max_g_freq, self.g_train_freq + 2)  # +2 更激进
            print(f"判别器过强，调整频率: G={self.g_train_freq}, D={self.d_train_freq}")

        elif loss_ratio > 2.5:  # 生成器过强
            # 增加判别器训练，减少生成器训练
            self.d_train_freq = min(self.max_d_freq, self.d_train_freq + 1)
            self.g_train_freq = max(self.min_g_freq, self.g_train_freq - 1)
            print(f"生成器过强，调整频率: G={self.g_train_freq}, D={self.d_train_freq}")

        # 平衡状态（基准分支）：保持合理的训练比例
        elif 0.5 <= loss_ratio <= 2.0 and self.g_train_freq < 4:
            # 如果在平衡范围内，确保生成器至少训练4次
            self.g_train_freq = min(4, self.g_train_freq + 1)

        return self.g_train_freq, self.d_train_freq

    def adjust_learning_rates(self, optimizer_g, optimizer_d, epoch):
        """
        动态调整学习率（融合版）
        使用基准分支的激进策略 + 学习率下限保护
        """
        if len(self.d_loss_history) < 10:
            for param_group in optimizer_g.param_groups:
                param_group['lr'] = self.g_lr_initial
            for param_group in optimizer_d.param_groups:
                param_group['lr'] = self.d_lr_initial

        else:
            recent_d, recent_g = self.get_recent_avg_loss()
            loss_ratio = recent_d / (recent_g + 1e-8)

            # 判别器过强时 - 使用基准分支的激进调整
            if loss_ratio < 0.3 or recent_d < 0.2:
                # 大幅降低判别器学习率，适度提高生成器学习率（基准分支策略）
                self.d_lr_current *= 0.8  # 从0.9改为0.8，更激进
                self.g_lr_current = min(self.g_lr_initial * 1.5, self.g_lr_current * 1.1)
                print(f"D过强，调整学习率: G_lr={self.g_lr_current:.2e}, D_lr={self.d_lr_current:.2e}")

            # 生成器过强时
            elif recent_d > 0.65 and recent_g < 0.3:
                # 提高判别器学习率，降低生成器学习率
                self.d_lr_current = min(self.d_lr_initial, self.d_lr_current * 1.1)
                self.g_lr_current *= 0.9
                print(f"G过强，调整学习率: G_lr={self.g_lr_current:.2e}, D_lr={self.d_lr_current:.2e}")

            # 设置学习率下限（基准分支）：避免过度降低
            self.d_lr_current = max(self.d_lr_initial * 0.1, self.d_lr_current)
            self.g_lr_current = max(self.g_lr_initial * 0.5, self.g_lr_current)

            # 应用新的学习率
            for param_group in optimizer_g.param_groups:
                param_group['lr'] = self.g_lr_current
            for param_group in optimizer_d.param_groups:
                param_group['lr'] = self.d_lr_current


    # Generate dynamic smooth labels
    def get_smooth_labels(self, epoch, batch_size, device, is_real=True):
        """
        生成动态平滑标签（融合版）
        使用基准分支的减少噪声策略 + 课程学习
        """
        # Noise annealing - 使用基准分支的减少噪声策略
        max_noise_std = 0.005  # 基准分支：从0.01→0.005
        min_noise_std = 0.001  # 基准分支：从0.002→0.001
        anneal_start_epoch = 0
        anneal_end_epoch = 60
        max_smooth_offset = 0.02  # 基准分支：从0.05→0.02

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

        # generate labels
        recent_d, recent_g = self.get_recent_avg_loss() if len(self.d_loss_history) >= 10 else (0.5, 0.5)

        if is_real:
            # 基准分支的策略：区分 real 和 fake 标签
            base_smooth = self.real_label_smooth - max_smooth_offset * (1 - progress)
            smooth_val = base_smooth

            if self.dynamic_smooth and recent_d < self.d_strong_threshold:
                smooth_val = max(0.90, smooth_val - 0.05)  # 基准分支：0.90, 0.05
                noise_std = base_noise_std + 0.005  # 基准分支：从0.01→0.005
            else:
                noise_std = base_noise_std

            labels = torch.ones(batch_size, 4, device=device).fill_(smooth_val)
            labels = torch.clamp(labels + torch.normal(0, noise_std, labels.shape, device=device), 0.85, 1.0)
        else:
            # 基准分支：修复了fake标签使用正确的base_smooth
            base_smooth = self.fake_label_smooth + max_smooth_offset * (1 - progress)
            smooth_val = base_smooth

            if self.dynamic_smooth and recent_g < self.g_strong_threshold:
                smooth_val = min(0.10, smooth_val + 0.05)  # 基准分支：0.10, 0.05
                noise_std = base_noise_std + 0.005
            else:
                noise_std = base_noise_std

            labels = torch.zeros(batch_size, 4, device=device).fill_(smooth_val)
            labels = torch.clamp(labels + torch.normal(0, noise_std, labels.shape, device=device), 0.0, 0.15)

        return labels.requires_grad_(False)

    def get_curriculum_weight(self, epoch, weight_type='detail'):
        """
        渐进增加任务难度的权重调度
        Args:
            epoch: 当前epoch
            weight_type: 'detail' 或 'physics'
        Returns:
            当前epoch对应的权重
        """
        if weight_type == 'detail':
            initial_weight = self.initial_detail_weight
            final_weight = self.final_detail_weight
        elif weight_type == 'physics':
            initial_weight = self.initial_physics_weight
            final_weight = self.final_physics_weight
        else:
            return 1.0

        # 预热阶段：使用极低权重
        if epoch < self.warmup_epochs:
            return initial_weight * 0.5

        # 课程学习阶段：线性增加权重
        if epoch < self.curriculum_epochs:
            progress = (epoch - self.warmup_epochs) / (self.curriculum_epochs - self.warmup_epochs)
            return initial_weight + progress * (final_weight - initial_weight)

        # 正常训练阶段：使用最终权重
        return final_weight

    def apply_curriculum_to_loss(self, loss_dict, epoch):
        """
        应用课程学习策略到各种损失
        Args:
            loss_dict: 包含各种损失的字典
            epoch: 当前epoch
        Returns:
            调整后的总损失
        """
        detail_weight = self.get_curriculum_weight(epoch, 'detail')
        physics_weight = self.get_curriculum_weight(epoch, 'physics')

        # 基础motion reconstruction loss - 始终全权重
        total_loss = loss_dict.get('motion_reg_loss', 0)

        # GAN loss - 渐进增加权重
        total_loss += detail_weight * loss_dict.get('gan_loss', 0)

        # 物理约束损失 - 渐进增加权重（更重要，权重更大）
        total_loss += physics_weight * loss_dict.get('bone_loss', 0)
        total_loss += physics_weight * loss_dict.get('angle_loss', 0)

        # 时序平滑损失 - 渐进增加
        total_loss += detail_weight * loss_dict.get('smoothness_loss', 0)
        total_loss += detail_weight * loss_dict.get('jerk_loss', 0)

        return total_loss

    def should_use_mixed_precision(self, epoch):
        """
        判断是否应该使用混合精度训练
        前期（预热阶段）不使用，中后期使用以加速训练
        """
        return epoch >= self.warmup_epochs


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


def compute_temporal_smoothness_loss(motion_seq):
    """
    计算运动序列的平滑度损失，确保生成的动作连贯。
    motion_seq: [B, T-1, features] - 帧间差分（速度）
    返回: 标量损失
    """
    # 速度已经是motion_seq（一阶导数）
    # 计算加速度（二阶导数）
    acceleration = motion_seq[:, 1:] - motion_seq[:, :-1]  # [B, T-2, features]

    # 加速度的L2范数作为平滑度度量
    # 惩罚急剧的加速度变化，使动作更自然
    smoothness_loss = torch.mean(torch.norm(acceleration, dim=-1))

    return smoothness_loss


def compute_jerk_loss(motion_seq):
    """
    计算jerk损失（三阶导数），进一步提高运动平滑度。
    motion_seq: [B, T-1, features] - 帧间差分（速度）
    返回: 标量损失
    """
    # 加速度（二阶导数）
    acceleration = motion_seq[:, 1:] - motion_seq[:, :-1]  # [B, T-2, features]

    # Jerk（三阶导数）
    jerk = acceleration[:, 1:] - acceleration[:, :-1]  # [B, T-3, features]

    # Jerk的L2范数
    jerk_loss = torch.mean(torch.norm(jerk, dim=-1))

    return jerk_loss


if __name__ == '__main__':
    # Load speaker data
    dataloader = Data_Loader(**common_kwargs)

    # Initialize hardware
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"-------- Using GPU Device {torch.cuda.get_device_name(0)} to Train the model --------")
    cuda = True if torch.cuda.is_available() else False

    # Initialize the curriculum/progressive training strategy
    dynamic_trainer = CurriculumGANTraining(g_lr=lr/2, d_lr=lr)

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

    # 混合精度训练 - 初始化GradScaler
    scaler_G = GradScaler() if cuda else None
    scaler_D = GradScaler() if cuda else None
    print("Mixed precision training enabled with GradScaler")

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

        # 打印课程学习状态
        detail_w = dynamic_trainer.get_curriculum_weight(epoch, 'detail')
        physics_w = dynamic_trainer.get_curriculum_weight(epoch, 'physics')
        amp_enabled = dynamic_trainer.should_use_mixed_precision(epoch)
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{n_epochs} - Curriculum Learning Status:")
        print(f"  Detail Weight: {detail_w:.3f} | Physics Weight: {physics_w:.3f}")
        print(f"  Mixed Precision: {'ENABLED' if amp_enabled else 'DISABLED (Warmup)'}")
        print(f"  Training Frequency: G={g_freq}, D={d_freq}")
        print(f"{'='*80}\n")

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
            #  Train Generator (dynamic with mixed precision)
            # -----------------
            use_amp = cuda and dynamic_trainer.should_use_mixed_precision(epoch)

            for gen_step in range(g_freq):
                optimizer_G.zero_grad()

                # 使用混合精度训练（预热后启用）
                if use_amp:
                    with autocast():
                        # Using audio as generator input
                        fake_pose, internal_losses = generator(audio, real_pose=real_pose)
                        # Generate motions
                        fake_motion = pos_to_motion(fake_pose)
                        # discriminator
                        fake_d, _ = discriminator(fake_motion)

                        # ============ 应用课程学习策略计算生成器损失 ============
                        loss_dict = {
                            'motion_reg_loss': motion_reg_loss(real_motion, fake_motion),
                            'gan_loss': lambda_gan * g_loss(fake_d, valid),
                            'smoothness_loss': 0.1 * compute_temporal_smoothness_loss(fake_motion),
                            'jerk_loss': 0.05 * compute_jerk_loss(fake_motion),
                            'bone_loss': internal_losses[0] if len(internal_losses) > 0 else torch.tensor(0.0).to(device),
                            'angle_loss': internal_losses[1] if len(internal_losses) > 1 else torch.tensor(0.0).to(device)
                        }
                        G_loss = dynamic_trainer.apply_curriculum_to_loss(loss_dict, epoch)

                    # 使用scaler进行反向传播
                    scaler_G.scale(G_loss).backward()
                    scaler_G.step(optimizer_G)
                    scaler_G.update()
                else:
                    # 标准精度训练（预热阶段）
                    fake_pose, internal_losses = generator(audio, real_pose=real_pose)
                    fake_motion = pos_to_motion(fake_pose)
                    fake_d, _ = discriminator(fake_motion)

                    loss_dict = {
                        'motion_reg_loss': motion_reg_loss(real_motion, fake_motion),
                        'gan_loss': lambda_gan * g_loss(fake_d, valid),
                        'smoothness_loss': 0.1 * compute_temporal_smoothness_loss(fake_motion),
                        'jerk_loss': 0.05 * compute_jerk_loss(fake_motion),
                        'bone_loss': internal_losses[0] if len(internal_losses) > 0 else torch.tensor(0.0).to(device),
                        'angle_loss': internal_losses[1] if len(internal_losses) > 1 else torch.tensor(0.0).to(device)
                    }
                    G_loss = dynamic_trainer.apply_curriculum_to_loss(loss_dict, epoch)

                    G_loss.backward()
                    optimizer_G.step()

            # ---------------------
            #  Train Discriminator (Dynamic with mixed precision)
            # ---------------------
            # Check whether the discriminator should be trained
            if dynamic_trainer.should_train_discriminator():

                for dis_step in range(d_freq):
                    optimizer_D.zero_grad()

                    # 固定生成器输出（防止梯度干扰）
                    with torch.no_grad():
                        if use_amp:
                            with autocast():
                                fake_pose_detached, _ = generator(audio)
                                fake_motion_detached = pos_to_motion(fake_pose_detached)
                        else:
                            fake_pose_detached, _ = generator(audio)
                            fake_motion_detached = pos_to_motion(fake_pose_detached)

                    # 使用混合精度训练判别器
                    if use_amp:
                        with autocast():
                            fake_d, _ = discriminator(fake_motion_detached.detach())
                            real_d, _ = discriminator(real_motion)

                            # Measure discriminator's ability to classify real from generated samples
                            real_loss = d_loss1(real_d, valid)
                            fake_loss = d_loss2(fake_d, fake)
                            D_loss = real_loss + lambda_d * fake_loss

                        scaler_D.scale(D_loss).backward()
                        scaler_D.step(optimizer_D)
                        scaler_D.update()
                    else:
                        fake_d, _ = discriminator(fake_motion_detached.detach())
                        real_d, _ = discriminator(real_motion)

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
                # 获取课程学习权重
                detail_weight = dynamic_trainer.get_curriculum_weight(epoch, 'detail')
                physics_weight = dynamic_trainer.get_curriculum_weight(epoch, 'physics')
                amp_status = "ON" if use_amp else "OFF"

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f] [Recent D: %.4f] [Recent G: %.4f]"
                    % (epoch, n_epochs, i + 1, total_batches, D_loss.item(), G_loss.item(), recent_d, recent_g)
                )
                print(
                    "  [G_freq: %d] [D_freq: %d] [Detail_W: %.2f] [Physics_W: %.2f] [AMP: %s]"
                    % (g_freq, d_freq, detail_weight, physics_weight, amp_status)
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
        val_smoothness_loss = 0.0
        val_jerk_loss = 0.0
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

                # 计算时序平滑损失
                smoothness = compute_temporal_smoothness_loss(fake_motion_val)
                jerk = compute_jerk_loss(fake_motion_val)
                val_smoothness_loss += smoothness.item()
                val_jerk_loss += jerk.item()

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

        # 计算平均bone_loss, angle_loss, smoothness_loss
        avg_bone_loss = val_bone_loss / val_steps
        avg_angle_loss = val_angle_loss / val_steps
        avg_smoothness_loss = val_smoothness_loss / val_steps
        avg_jerk_loss = val_jerk_loss / val_steps

        print(f"[Validation] Epoch {epoch}/{n_epochs} | G_loss: {val_g_loss:.4f} | D_loss: {val_d_loss:.4f}")
        print(f"  Bone Loss: {avg_bone_loss:.4f} | Angle Loss: {avg_angle_loss:.4f} | Smoothness Loss: {avg_smoothness_loss:.4f} | Jerk Loss: {avg_jerk_loss:.4f}")

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

