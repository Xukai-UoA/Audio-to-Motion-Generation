#!/usr/bin/env python3
"""
测试脚本 - 验证GAN模型改进
测试内容：
1. 模型初始化
2. 身体-手部联合注意力机制
3. 课程学习策略
4. 混合精度训练
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_motion_model import SelfAttention_G, SelfAttention_D
from version5_model_train import CurriculumGANTraining

def test_model_initialization():
    """测试模型初始化"""
    print("=" * 80)
    print("测试 1: 模型初始化")
    print("=" * 80)

    try:
        # 初始化生成器
        generator = SelfAttention_G(time_steps=64, in_channels=256, out_channels=256, out_feats=104, p=0.2)
        print("✓ 生成器初始化成功")

        # 检查新增的联合注意力层
        assert hasattr(generator, 'body_hand_cross_attention'), "缺少 body_hand_cross_attention"
        assert hasattr(generator, 'hand_body_cross_attention'), "缺少 hand_body_cross_attention"
        print("✓ 身体-手部联合注意力层存在")

        # 初始化判别器
        discriminator = SelfAttention_D(out_channels=64)
        print("✓ 判别器初始化成功")

        return True
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        return False

def test_forward_pass():
    """测试前向传播"""
    print("\n" + "=" * 80)
    print("测试 2: 前向传播（包括联合注意力）")
    print("=" * 80)

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")

        # 初始化模型
        generator = SelfAttention_G().to(device)
        discriminator = SelfAttention_D().to(device)

        # 创建模拟输入
        batch_size = 4
        time_steps = 64
        audio_feats = 128
        pose_feats = 104

        audio = torch.randn(batch_size, time_steps, audio_feats).to(device)
        real_pose = torch.randn(batch_size, time_steps, pose_feats).to(device)

        # 生成器前向传播
        print(f"输入音频形状: {audio.shape}")
        fake_pose, internal_losses = generator(audio, real_pose=real_pose)
        print(f"✓ 生成器前向传播成功")
        print(f"  生成姿态形状: {fake_pose.shape}")
        print(f"  内部损失数量: {len(internal_losses)}")

        # 判别器前向传播
        motion = torch.diff(fake_pose, n=1, dim=1)
        d_out, d_losses = discriminator(motion)
        print(f"✓ 判别器前向传播成功")
        print(f"  判别器输出形状: {d_out.shape}")

        return True
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_curriculum_training():
    """测试课程学习策略"""
    print("\n" + "=" * 80)
    print("测试 3: 课程学习策略")
    print("=" * 80)

    try:
        trainer = CurriculumGANTraining(g_lr=5e-6, d_lr=10e-6)
        print("✓ CurriculumGANTraining 初始化成功")

        # 测试权重调度
        epochs_to_test = [0, 5, 10, 25, 50, 100]
        print("\n权重调度测试:")
        print(f"{'Epoch':<10} {'Detail Weight':<15} {'Physics Weight':<15} {'Use AMP':<10}")
        print("-" * 50)
        for epoch in epochs_to_test:
            detail_w = trainer.get_curriculum_weight(epoch, 'detail')
            physics_w = trainer.get_curriculum_weight(epoch, 'physics')
            use_amp = trainer.should_use_mixed_precision(epoch)
            print(f"{epoch:<10} {detail_w:<15.3f} {physics_w:<15.3f} {str(use_amp):<10}")

        # 测试损失应用
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss_dict = {
            'motion_reg_loss': torch.tensor(1.0, device=device),
            'gan_loss': torch.tensor(0.5, device=device),
            'smoothness_loss': torch.tensor(0.1, device=device),
            'jerk_loss': torch.tensor(0.05, device=device),
            'bone_loss': torch.tensor(0.2, device=device),
            'angle_loss': torch.tensor(0.3, device=device)
        }

        total_loss = trainer.apply_curriculum_to_loss(loss_dict, epoch=0)
        print(f"\n✓ 课程学习损失应用成功 (epoch 0): {total_loss.item():.4f}")

        total_loss = trainer.apply_curriculum_to_loss(loss_dict, epoch=50)
        print(f"✓ 课程学习损失应用成功 (epoch 50): {total_loss.item():.4f}")

        return True
    except Exception as e:
        print(f"✗ 课程学习测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mixed_precision():
    """测试混合精度训练"""
    print("\n" + "=" * 80)
    print("测试 4: 混合精度训练")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("⚠ CUDA不可用，跳过混合精度测试")
        return True

    try:
        device = torch.device("cuda:0")

        # 初始化模型和优化器
        generator = SelfAttention_G().to(device)
        optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
        scaler = GradScaler()

        # 创建模拟输入
        audio = torch.randn(2, 64, 128).to(device)
        real_pose = torch.randn(2, 64, 104).to(device)

        # 混合精度前向传播
        optimizer.zero_grad()
        with autocast():
            fake_pose, internal_losses = generator(audio, real_pose=real_pose)
            loss = torch.mean((fake_pose - real_pose) ** 2)
            for internal_loss in internal_losses:
                loss += internal_loss

        # 混合精度反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f"✓ 混合精度训练成功")
        print(f"  损失值: {loss.item():.4f}")

        return True
    except Exception as e:
        print(f"✗ 混合精度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_discriminator_balance():
    """测试判别器平衡策略"""
    print("\n" + "=" * 80)
    print("测试 5: 判别器平衡策略")
    print("=" * 80)

    try:
        trainer = CurriculumGANTraining()

        # 模拟判别器过强的情况
        for i in range(20):
            trainer.update_loss_history(d_loss=0.15, g_loss=0.85)

        should_train = trainer.should_train_discriminator()
        print(f"✓ 判别器过强检测: should_train={should_train}")
        print(f"  跳过计数: {trainer.d_skip_count}")

        # 测试频率调整
        g_freq, d_freq = trainer.adjust_training_frequency(epoch=20)
        print(f"✓ 训练频率调整: G_freq={g_freq}, D_freq={d_freq}")

        return True
    except Exception as e:
        print(f"✗ 判别器平衡测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("GAN模型改进验证测试")
    print("=" * 80 + "\n")

    results = []

    # 运行测试
    results.append(("模型初始化", test_model_initialization()))
    results.append(("前向传播", test_forward_pass()))
    results.append(("课程学习", test_curriculum_training()))
    results.append(("混合精度", test_mixed_precision()))
    results.append(("判别器平衡", test_discriminator_balance()))

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:<20}: {status}")

    print("-" * 80)
    print(f"总计: {passed}/{total} 测试通过")
    print("=" * 80 + "\n")

    if passed == total:
        print("🎉 所有测试通过！模型改进验证成功！")
        return 0
    else:
        print("⚠️  部分测试失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    exit(main())
