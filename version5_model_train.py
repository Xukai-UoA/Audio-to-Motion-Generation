import os
import time
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

        # 动态调整参数 - 更激进的平衡策略
        self.d_strong_threshold = 0.25  # 判别器过强阈值（提高以减少误判）
        self.g_weak_threshold = 0.75  # 生成器过弱阈值
        self.g_strong_threshold = 0.12

        # 训练频率控制 - 更大的调整范围
        self.d_train_freq = 1
        self.g_train_freq = 4  # 初始增加生成器训练频率
        self.min_d_freq = 1
        self.max_d_freq = 3  # 允许更高的判别器训练频率
        self.min_g_freq = 3  # 提高最小生成器训练频率
        self.max_g_freq = 8  # 允许更高的生成器训练频率

        # 判别器强度控制
        self.d_skip_count = 0  # 跳过判别器训练的次数
        self.max_d_skip = 5    # 最多连续跳过5次

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
        """判断是否应该训练判别器（增强版）"""
        if len(self.d_loss_history) == 0:
            self.d_skip_count = 0
            return True

        recent_d, recent_g = self.get_recent_avg_loss()

        # 如果判别器过强，减少训练（但不能连续跳过太多次）
        if recent_d < self.d_strong_threshold and recent_g > self.g_weak_threshold:
            if self.d_skip_count < self.max_d_skip:
                self.d_skip_count += 1
                return False
            else:
                # 已经跳过太多次，强制训练一次后重置计数
                self.d_skip_count = 0
                return True

        # 如果生成器太强，必须训练判别器
        if recent_d > 0.7 and recent_g < 0.4:
            self.d_skip_count = 0
            return True

        # 正常情况下训练，重置跳过计数
        self.d_skip_count = 0
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
        base_smooth_val = self.real_label_smooth - max_smooth_offset * (1 - progress) if is_real else self.fake_label_smooth + max_smooth_offset * (1 - progress)

        # generate labels
        recent_d, recent_g = self.get_recent_avg_loss() if len(self.d_loss_history) >= 10 else (0.5, 0.5)
        if is_real:
            smooth_val = base_smooth_val
            if self.dynamic_smooth and recent_d < self.d_strong_threshold:
                smooth_val = max(0.97, smooth_val - 0.1)  # 判别器过强，增加平滑
                noise_std = base_noise_std + 0.01  # 额外噪声
            else:
                noise_std = base_noise_std
            labels = torch.ones(batch_size, 4, device=device).fill_(smooth_val)
            labels = torch.clamp(labels + torch.normal(0, noise_std, labels.shape, device=device), 0.85, 1.0)
        else:
            smooth_val = base_smooth_val
            if self.dynamic_smooth and recent_g < self.g_strong_threshold:
                # 生成器过强时，减少假标签平滑度，让判别器更容易区分（帮助判别器）
                smooth_val = max(0.0, smooth_val - 0.05)  # 降低smooth_val使假标签更接近0
                noise_std = base_noise_std + 0.01
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

# WGAN-GP Hyperparameters
lr = 10e-4
n_epochs = 500
lambda_gp = 10.0  # Gradient penalty coefficient (standard value)
n_critic = 5  # Train discriminator 5 times per generator update (WGAN-GP standard)

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


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """
    计算WGAN-GP的梯度惩罚项

    Args:
        discriminator: 判别器模型
        real_samples: 真实样本 [B, T, features]
        fake_samples: 生成样本 [B, T, features]
        device: 设备

    Returns:
        gradient_penalty: 梯度惩罚损失（标量）
    """
    batch_size = real_samples.size(0)

    # 随机采样插值系数 epsilon ~ Uniform[0, 1]
    epsilon = torch.rand(batch_size, 1, 1, device=device)
    epsilon = epsilon.expand_as(real_samples)

    # 计算插值样本: x_hat = epsilon * real + (1 - epsilon) * fake
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated.requires_grad_(True)

    # 判别器对插值样本的评分
    d_interpolated, _ = discriminator(interpolated)

    # 计算梯度 ∇D(x_hat)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated, device=device),
        create_graph=True,  # 允许二阶导数（用于反向传播）
        retain_graph=True,
        only_inputs=True
    )[0]

    # 展平梯度: [B, T, features] → [B, T*features]
    gradients = gradients.view(batch_size, -1)

    # 计算L2范数: ||∇D(x_hat)||_2
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Gradient Penalty: E[(||∇D(x_hat)||_2 - 1)^2]
    gradient_penalty = torch.mean((gradients_norm - 1) ** 2)

    return gradient_penalty


if __name__ == '__main__':
    # Load speaker data
    dataloader = Data_Loader(**common_kwargs)

    # Initialize hardware
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"-------- Using GPU Device {torch.cuda.get_device_name(0)} to Train the model --------")
    cuda = True if torch.cuda.is_available() else False

    # Initialize the curriculum/progressive training strategy
    # WGAN-GP: Use higher D learning rate for better critic training
    dynamic_trainer = CurriculumGANTraining(g_lr=lr/2, d_lr=lr)

    # Define loss function
    motion_reg_loss = torch.nn.L1Loss()
    # WGAN-GP: No need for MSELoss, use direct mean for Wasserstein distance

    # Initialize generator and discriminator (critic in WGAN-GP)
    generator = SelfAttention_G()
    discriminator = SelfAttention_D(out_channels=64)  # Now acts as critic
    print("Generator and Critic (WGAN-GP Discriminator) model Initialized successfully ...")

    # Move the models and loss functions on GPU
    if cuda:
        generator.cuda()
        discriminator.cuda()
        motion_reg_loss.cuda()

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

            # WGAN-GP: No need for adversarial ground truth labels
            # WGAN-GP uses Wasserstein distance directly

            # 生成真实motion（在循环外计算，避免重复计算）
            real_motion = pos_to_motion(real_pose)
            # -----------------
            #  Train Generator (dynamic with mixed precision)
            # -----------------
            use_amp = cuda and dynamic_trainer.should_use_mixed_precision(epoch)

            # WGAN-GP: Train Generator (after n_critic discriminator updates)
            for gen_step in range(g_freq):
                optimizer_G.zero_grad()

                # 使用混合精度训练（预热后启用）
                if use_amp:
                    with autocast():
                        # Using audio as generator input
                        fake_pose, internal_losses = generator(audio, real_pose=real_pose)
                        # Generate motions
                        fake_motion = pos_to_motion(fake_pose)
                        # Critic score
                        fake_d, _ = discriminator(fake_motion)

                        # ============ WGAN-GP Generator Loss ============
                        # WGAN-GP: G_loss = -E[D(fake)] (maximize D(fake))
                        wgan_g_loss = -torch.mean(fake_d)

                        # 应用课程学习策略计算生成器损失
                        loss_dict = {
                            'motion_reg_loss': motion_reg_loss(real_motion, fake_motion),
                            'gan_loss': wgan_g_loss,  # WGAN-GP Wasserstein loss
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

                    # WGAN-GP Generator Loss
                    wgan_g_loss = -torch.mean(fake_d)

                    loss_dict = {
                        'motion_reg_loss': motion_reg_loss(real_motion, fake_motion),
                        'gan_loss': wgan_g_loss,  # WGAN-GP Wasserstein loss
                        'smoothness_loss': 0.1 * compute_temporal_smoothness_loss(fake_motion),
                        'jerk_loss': 0.05 * compute_jerk_loss(fake_motion),
                        'bone_loss': internal_losses[0] if len(internal_losses) > 0 else torch.tensor(0.0).to(device),
                        'angle_loss': internal_losses[1] if len(internal_losses) > 1 else torch.tensor(0.0).to(device)
                    }
                    G_loss = dynamic_trainer.apply_curriculum_to_loss(loss_dict, epoch)

                    G_loss.backward()
                    optimizer_G.step()

            # ---------------------
            #  Train Critic/Discriminator (WGAN-GP with Gradient Penalty)
            # ---------------------
            # WGAN-GP: Train discriminator multiple times (n_critic) per generator update
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

                            # ============ WGAN-GP Discriminator Loss ============
                            # Wasserstein distance: -E[D(real)] + E[D(fake)]
                            wasserstein_d = -torch.mean(real_d) + torch.mean(fake_d)

                            # Gradient Penalty
                            gp = compute_gradient_penalty(discriminator, real_motion, fake_motion_detached, device)

                            # Total discriminator loss: W-distance + λ*GP
                            D_loss = wasserstein_d + lambda_gp * gp

                        scaler_D.scale(D_loss).backward()
                        scaler_D.step(optimizer_D)
                        scaler_D.update()
                    else:
                        fake_d, _ = discriminator(fake_motion_detached.detach())
                        real_d, _ = discriminator(real_motion)

                        # WGAN-GP Discriminator Loss
                        wasserstein_d = -torch.mean(real_d) + torch.mean(fake_d)
                        gp = compute_gradient_penalty(discriminator, real_motion, fake_motion_detached, device)
                        D_loss = wasserstein_d + lambda_gp * gp

                        D_loss.backward()
                        optimizer_D.step()

            else:
                # Use the last time loss value, if skip the D training
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

                # WGAN-GP validation loss (no labels needed)
                val_batch_size = real_pose_val.size(0)

                # 验证生成器损失 (WGAN-GP)
                motion_reg_loss_val = motion_reg_loss(real_motion_val, fake_motion_val)
                fake_d_val, _ = discriminator(fake_motion_val)
                wgan_g_loss_val = -torch.mean(fake_d_val)
                g_loss_val = motion_reg_loss_val + wgan_g_loss_val

                # 验证判别器损失 (WGAN-GP)
                real_d_val, _ = discriminator(real_motion_val)
                fake_d_val, _ = discriminator(fake_motion_val.detach())
                wasserstein_d_val = -torch.mean(real_d_val) + torch.mean(fake_d_val)
                gp_val = compute_gradient_penalty(discriminator, real_motion_val, fake_motion_val, device)
                d_loss_val = wasserstein_d_val + lambda_gp * gp_val

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

