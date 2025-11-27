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
    """
    渐进式/课程学习GAN训练策略类

    核心改进（基于真实训练结果）：
    1. 学习率反转：G_lr > D_lr（D学得太快，需要限制）
    2. 固定合理训练比例：G=2, D=1（不浪费计算资源）
    3. 激进的学习率动态调整（而非频率调整）
    """

    def __init__(self, g_lr=1e-4, d_lr=3e-5):
        """
        初始化训练策略

        新设计的学习率（基于epoch1训练崩溃的分析）：
        - g_lr = 1e-4 (0.0001): 生成器学习率
        - d_lr = 3e-5 (0.00003): 判别器学习率（是G的30%）

        旧设计的问题：
        - g_lr = 0.0005, d_lr = 0.001 (D是G的2倍！)
        - 导致D在400 batches内崩溃（D loss: 0.0301 → 0.0022）

        新设计原理：
        - D通常比G更容易学习（分类比生成简单）
        - D学习率应该显著低于G
        - 通过学习率平衡，而非训练频率（更高效）
        """
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

        # 动态调整参数
        self.d_strong_threshold = 0.08  # D过强阈值
        self.g_weak_threshold = 0.90    # G过弱阈值（保留兼容性，实际不太使用）
        self.g_strong_threshold = 0.03  # G过强阈值

        # ============ 训练频率控制 - 改进版 ============
        # 核心思想：用学习率平衡，而非极端的训练频率
        self.d_train_freq = 1
        self.g_train_freq = 2  # 初始值：2:1比例（用户需求）
        self.min_d_freq = 1
        self.max_d_freq = 2    # D最多训练2次（不再允许更多）
        self.min_g_freq = 2    # G最少训练2次（保持2:1比例）
        self.max_g_freq = 3    # G最多训练3次（不再使用6, 8这种极端值）

        # 判别器强度控制
        self.skip_d_counter = 0  # 跳过判别器训练的次数
        self.max_skip_count = 20  # 最多连续跳过20次

        # 标签平滑参数
        self.real_label_smooth = 0.90  # Real标签平滑值
        self.fake_label_smooth = 0.10  # Fake标签平滑值
        self.dynamic_smooth = True     # 启用动态平滑调整

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

        # 使用nanmean忽略NaN值，提高数值稳定性
        recent_d = np.nanmean(self.d_loss_history[-window:])
        recent_g = np.nanmean(self.g_loss_history[-window:])

        # 如果所有值都是NaN，使用默认值
        if np.isnan(recent_d):
            recent_d = 1.0
        if np.isnan(recent_g):
            recent_g = 1.0

        return recent_d, recent_g

    def should_train_discriminator(self):
        """
        判断是否应该训练判别器（改进版）
        修复判别器过强检测逻辑，使用更合理的条件
        """
        if len(self.d_loss_history) == 0:
            self.skip_d_counter = 0
            return True

        recent_d, recent_g = self.get_recent_avg_loss()

        # 强制训练机制：如果连续跳过次数过多，必须训练一次
        if self.skip_d_counter >= self.max_skip_count:
            print(f"🔄 强制训练判别器 (连续跳过{self.skip_d_counter}次)")
            self.skip_d_counter = 0
            return True

        # ============ 改进的判别器过强检测逻辑 ============
        # 计算损失比率（更直观的平衡指标）
        loss_ratio = recent_d / (recent_g + 1e-8)

        # 条件1：判别器极度过强 - D_loss < 0.05（极低）
        if recent_d < 0.05:
            self.skip_d_counter += 1
            print(f"⚠️ 判别器极度过强 (D={recent_d:.4f} < 0.05)，跳过训练 [{self.skip_d_counter}/{self.max_skip_count}]")
            return False

        # 条件2：损失比率极端不平衡 - ratio < 0.1（判别器远强于生成器）
        if loss_ratio < 0.1:
            self.skip_d_counter += 1
            print(f"⚠️ GAN严重失衡 (ratio={loss_ratio:.4f} < 0.1)，跳过判别器训练 [{self.skip_d_counter}/{self.max_skip_count}]")
            return False

        # 条件3：原有条件（兼容性保留） - D_loss < 0.08 且 G_loss > 0.90
        if recent_d < self.d_strong_threshold and recent_g > self.g_weak_threshold:
            self.skip_d_counter += 1
            print(f"⚠️ 判别器过强 (D={recent_d:.4f}, G={recent_g:.4f})，跳过训练 [{self.skip_d_counter}/{self.max_skip_count}]")
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
        动态调整训练频率（改进版 - 保守策略）

        核心改进：
        - 不再使用极端频率（6:1, 8:1）→ 只在2:1和3:1之间微调
        - 主要依靠学习率调整来平衡G/D（更高效）
        - 频率调整作为辅助手段
        """
        if len(self.d_loss_history) < 10:
            return self.g_train_freq, self.d_train_freq

        recent_d, recent_g = self.get_recent_avg_loss()

        # 计算损失比值
        loss_ratio = recent_d / (recent_g + 1e-8)

        # ============ 保守的频率调整（不浪费计算资源）============
        # 条件1：判别器极度/严重过强 - ratio < 0.1，调到3:1
        if loss_ratio < 0.1 or recent_d < 0.05:
            self.d_train_freq = 1
            self.g_train_freq = 3  # 最大值3（不再是8）
            print(f"⚠️ 判别器严重过强 (ratio={loss_ratio:.4f}, D={recent_d:.4f})，调整频率: G=3, D=1")

        # 条件2：判别器过强 - ratio < 0.3，微调到2:1或3:1
        elif loss_ratio < 0.3 or recent_d < 0.2:
            self.d_train_freq = 1
            # 只增加1（不再+2或+3）
            self.g_train_freq = min(self.max_g_freq, self.g_train_freq + 1)
            print(f"📉 判别器过强 (ratio={loss_ratio:.4f})，微调: G={self.g_train_freq}, D=1")

        # 生成器过强 - ratio > 2.5
        elif loss_ratio > 2.5:
            # 可能增加D训练或减少G训练
            self.d_train_freq = min(self.max_d_freq, self.d_train_freq + 1)
            self.g_train_freq = max(self.min_g_freq, self.g_train_freq - 1)
            print(f"📈 生成器过强 (ratio={loss_ratio:.4f})，调整: G={self.g_train_freq}, D={self.d_train_freq}")

        # 平衡状态：保持2:1比例
        elif 0.5 <= loss_ratio <= 2.0:
            # 恢复到默认2:1比例
            if self.g_train_freq != 2:
                self.g_train_freq = 2
                self.d_train_freq = 1
                print(f"⚖️ GAN平衡 (ratio={loss_ratio:.4f})，恢复2:1比例")

        return self.g_train_freq, self.d_train_freq

    def adjust_learning_rates(self, optimizer_g, optimizer_d, epoch):
        """
        动态调整学习率（改进版 - 激进策略）

        核心改进：
        - 现在主要依靠学习率来平衡G/D（不是训练频率）
        - 更激进的调整幅度（±20-30%而非±10%）
        - 分级响应不同程度的失衡
        """
        if len(self.d_loss_history) < 10:
            for param_group in optimizer_g.param_groups:
                param_group['lr'] = self.g_lr_initial
            for param_group in optimizer_d.param_groups:
                param_group['lr'] = self.d_lr_initial

        else:
            recent_d, recent_g = self.get_recent_avg_loss()
            loss_ratio = recent_d / (recent_g + 1e-8)

            # ============ 激进的学习率调整（主要平衡手段）============
            # 条件1：判别器极度过强 - ratio < 0.05，大幅调整
            if loss_ratio < 0.05 or recent_d < 0.03:
                # 大幅降低D学习率（-40%），大幅提高G学习率（+30%）
                self.d_lr_current *= 0.6  # 降低40%
                self.g_lr_current = min(self.g_lr_initial * 2.0, self.g_lr_current * 1.3)  # 提高30%，上限2倍
                print(f"🚨 D极度过强 (ratio={loss_ratio:.4f})，大幅调整学习率: G_lr={self.g_lr_current:.2e}, D_lr={self.d_lr_current:.2e}")

            # 条件2：判别器严重过强 - ratio < 0.1，激进调整
            elif loss_ratio < 0.1 or recent_d < 0.05:
                # 显著降低D学习率（-30%），提高G学习率（+20%）
                self.d_lr_current *= 0.7  # 降低30%
                self.g_lr_current = min(self.g_lr_initial * 1.8, self.g_lr_current * 1.2)  # 提高20%
                print(f"⚠️ D严重过强 (ratio={loss_ratio:.4f})，激进调整学习率: G_lr={self.g_lr_current:.2e}, D_lr={self.d_lr_current:.2e}")

            # 条件3：判别器过强 - ratio < 0.3，常规调整
            elif loss_ratio < 0.3 or recent_d < 0.2:
                # 降低D学习率（-20%），提高G学习率（+10%）
                self.d_lr_current *= 0.8  # 降低20%
                self.g_lr_current = min(self.g_lr_initial * 1.5, self.g_lr_current * 1.1)  # 提高10%
                print(f"📉 D过强 (ratio={loss_ratio:.4f})，调整学习率: G_lr={self.g_lr_current:.2e}, D_lr={self.d_lr_current:.2e}")

            # 生成器过强 - ratio > 2.5
            elif loss_ratio > 2.5 or (recent_d > 0.65 and recent_g < 0.3):
                # 提高D学习率（+10%），降低G学习率（-10%）
                self.d_lr_current = min(self.d_lr_initial * 1.2, self.d_lr_current * 1.1)  # 上限1.2倍初始值
                self.g_lr_current *= 0.9
                print(f"📈 G过强 (ratio={loss_ratio:.4f})，调整学习率: G_lr={self.g_lr_current:.2e}, D_lr={self.d_lr_current:.2e}")

            # 平衡状态 - ratio在0.5-2.0之间，逐渐恢复到初始值
            elif 0.5 <= loss_ratio <= 2.0:
                # 缓慢恢复到初始学习率（每次恢复5%的差距）
                if self.g_lr_current < self.g_lr_initial:
                    self.g_lr_current = min(self.g_lr_initial, self.g_lr_current * 1.05)
                elif self.g_lr_current > self.g_lr_initial:
                    self.g_lr_current = max(self.g_lr_initial, self.g_lr_current * 0.95)

                if self.d_lr_current < self.d_lr_initial:
                    self.d_lr_current = min(self.d_lr_initial, self.d_lr_current * 1.05)
                elif self.d_lr_current > self.d_lr_initial:
                    self.d_lr_current = max(self.d_lr_initial, self.d_lr_current * 0.95)

            # 设置学习率边界：防止过度调整
            self.d_lr_current = max(self.d_lr_initial * 0.05, min(self.d_lr_initial * 1.5, self.d_lr_current))  # 5%-150%
            self.g_lr_current = max(self.g_lr_initial * 0.3, min(self.g_lr_initial * 2.0, self.g_lr_current))  # 30%-200%

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
        loss_ratio = recent_d / (recent_g + 1e-8)

        if is_real:
            # 基准分支的策略：区分 real 和 fake 标签
            base_smooth = self.real_label_smooth - max_smooth_offset * (1 - progress)
            smooth_val = base_smooth

            # ============ 增强的动态平滑（新增）============
            if self.dynamic_smooth:
                # 极端不平衡 - ratio < 0.05，使用最强平滑
                if loss_ratio < 0.05 or recent_d < 0.03:
                    smooth_val = 0.80  # 强平滑：1.0 → 0.80
                    noise_std = base_noise_std + 0.01  # 增加噪声
                # 严重不平衡 - ratio < 0.1
                elif loss_ratio < 0.1 or recent_d < 0.05:
                    smooth_val = max(0.85, smooth_val - 0.10)  # 0.90 → 0.85
                    noise_std = base_noise_std + 0.008
                # 原有逻辑 - D过强
                elif recent_d < self.d_strong_threshold:
                    smooth_val = max(0.90, smooth_val - 0.05)
                    noise_std = base_noise_std + 0.005
                else:
                    noise_std = base_noise_std
            else:
                noise_std = base_noise_std

            labels = torch.ones(batch_size, 4, device=device).fill_(smooth_val)
            labels = torch.clamp(labels + torch.normal(0, noise_std, labels.shape, device=device), 0.75, 1.0)  # 扩大范围
        else:
            # 基准分支：修复了fake标签使用正确的base_smooth
            base_smooth = self.fake_label_smooth + max_smooth_offset * (1 - progress)
            smooth_val = base_smooth

            # ============ 增强的动态平滑（新增）============
            if self.dynamic_smooth:
                # 极端不平衡 - 给fake标签更高值，迷惑判别器
                if loss_ratio < 0.05 or recent_d < 0.03:
                    smooth_val = 0.20  # 强平滑：0.0 → 0.20
                    noise_std = base_noise_std + 0.01
                # 严重不平衡
                elif loss_ratio < 0.1 or recent_d < 0.05:
                    smooth_val = min(0.15, smooth_val + 0.10)  # 0.10 → 0.15
                    noise_std = base_noise_std + 0.008
                # 原有逻辑
                elif recent_g < self.g_strong_threshold:
                    smooth_val = min(0.10, smooth_val + 0.05)
                    noise_std = base_noise_std + 0.005
                else:
                    noise_std = base_noise_std
            else:
                noise_std = base_noise_std

            labels = torch.zeros(batch_size, 4, device=device).fill_(smooth_val)
            labels = torch.clamp(labels + torch.normal(0, noise_std, labels.shape, device=device), 0.0, 0.25)  # 扩大范围

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

    # ============ Initialize the curriculum/progressive training strategy ============
    # 新设计的学习率（基于epoch1训练崩溃的分析）
    # 旧设计：g_lr=lr/2=0.0005, d_lr=lr=0.001（D是G的2倍！导致D崩溃）
    # 新设计：g_lr > d_lr（D学得快，需要降低D学习率）
    lr_G = 1e-4  # 0.0001 - 生成器学习率
    lr_D = 3e-5  # 0.00003 - 判别器学习率（是G的30%）
    dynamic_trainer = CurriculumGANTraining(g_lr=lr_G, d_lr=lr_D)

    print(f"初始学习率设置:")
    print(f"  Generator LR: {lr_G:.2e}")
    print(f"  Discriminator LR: {lr_D:.2e}")
    print(f"  LR Ratio (G/D): {lr_G/lr_D:.2f}:1")
    print(f"初始训练频率: G=2, D=1 (2:1比例，不浪费计算资源)")

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

    # 混合精度训练 - 初始化GradScaler (使用更保守的参数防止NaN)
    # init_scale: 初始loss scale (默认2^16, 降低到2^12更保守)
    # growth_interval: 连续成功步数后才增加scale (默认2000, 增加到3000更保守)
    scaler_G = GradScaler(init_scale=2.**12, growth_interval=3000) if cuda else None
    scaler_D = GradScaler(init_scale=2.**12, growth_interval=3000) if cuda else None
    print("Mixed precision training enabled with conservative GradScaler (init_scale=2^12)")

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

                    # 梯度裁剪防止梯度爆炸 (Gradient clipping to prevent explosion)
                    scaler_G.unscale_(optimizer_G)  # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)

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

                    # 梯度裁剪防止梯度爆炸 (Gradient clipping to prevent explosion)
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)

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

                        # 梯度裁剪防止梯度爆炸 (Gradient clipping to prevent explosion)
                        scaler_D.unscale_(optimizer_D)  # Unscale before clipping
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

                        scaler_D.step(optimizer_D)
                        scaler_D.update()
                    else:
                        fake_d, _ = discriminator(fake_motion_detached.detach())
                        real_d, _ = discriminator(real_motion)

                        real_loss = d_loss1(real_d, valid)
                        fake_loss = d_loss2(fake_d, fake)
                        D_loss = real_loss + lambda_d * fake_loss

                        D_loss.backward()

                        # 梯度裁剪防止梯度爆炸 (Gradient clipping to prevent explosion)
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

                        optimizer_D.step()

            else:
                # Use the last time loss value, if skip the D training
                # D_loss = torch.tensor(d_loss_list[-1] if d_loss_list else 1.0)
                D_loss = torch.tensor(d_loss_list[-1])
                print(f"跳过判别器训练 - 判别器过强")

            # ============ NaN检测和处理 (NaN Detection and Handling) ============
            d_loss_val = D_loss.item()
            g_loss_val = G_loss.item()

            # 检测NaN或Inf - 如果发现则跳过这个batch并发出警告
            if torch.isnan(D_loss) or torch.isinf(D_loss) or torch.isnan(G_loss) or torch.isinf(G_loss):
                print(f"\n{'='*80}")
                print(f"⚠️ WARNING: NaN/Inf detected at Epoch {epoch}, Batch {i+1}")
                print(f"  D_loss: {d_loss_val}, G_loss: {g_loss_val}")
                print(f"  Skipping this batch and resetting gradients...")
                print(f"{'='*80}\n")

                # 清零梯度，跳过这个batch
                optimizer_G.zero_grad()
                optimizer_D.zero_grad()

                # 不更新损失历史，使用上一次的值（如果有的话）
                if len(d_loss_list) > 0:
                    d_loss_val = d_loss_list[-1]
                    g_loss_val = g_loss_list[-1]
                else:
                    d_loss_val = 1.0  # 初始默认值
                    g_loss_val = 1.0

                # 使用替代值更新历史
                dynamic_trainer.update_loss_history(d_loss_val, g_loss_val)
                continue  # 跳过这个batch

            # Update the loss history (only if valid)
            dynamic_trainer.update_loss_history(d_loss_val, g_loss_val)

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

