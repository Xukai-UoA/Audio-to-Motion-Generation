import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GATConv
from torch_geometric.data import Data, Batch  # 用于批量图处理

from model_layers import *
from pats.data_loading import Skeleton2D


class SelfAttention_G(nn.Module):
    '''
    input_shape:  (N, time, frequency)
    output_shape: (N, time, pose_feats)
    '''

    def __init__(self, time_steps=64, in_channels=256, out_channels=256, out_feats=104, p=0.2):
        super(SelfAttention_G, self).__init__()
        self.audio_encoder = AudioEncoder(output_feats=time_steps, p=p)
        self.unet = UNet1D(input_channels=in_channels, output_channels=out_channels, p=p)

        """Graph Convolutional Network(GCN)"""
        # Separate decoder: Share features and process body and hand parts separately
        self.body_feats = 20
        self.hand_feats = out_feats - self.body_feats

        # Full body joints (first 10 are body, 10:52 hand)
        self.num_body_joints = 10  # Adjust based on skeleton (e.g., core body joints)
        self.num_hand_joints = 42  # LHandRoot to RHandLittle4
        self.joint_feat_dim = 64  # Feature dim per joint (increased for complexity)

        # get parents and joint_names from skeleton.py
        self.skeleton = Skeleton2D()
        parents = self.skeleton.parents
        joint_names = self.skeleton.joint_names

        # Body edge index (simplified body graph)
        body_parents = parents[:self.num_body_joints]
        body_parents = [par if par < self.num_body_joints else -1 for par in body_parents]
        body_edge_index = []
        for i, par in enumerate(body_parents):
            if par != -1:
                body_edge_index.append([par, i])
                body_edge_index.append([i, par])
        self.body_edge_index = torch.tensor(body_edge_index, dtype=torch.long).t().contiguous()

        # Hand edge index (as original)
        hand_parents = parents[10:10 + self.num_hand_joints]
        hand_parents = [par - 10 if par >= 10 else -1 for par in hand_parents]
        hand_edge_index = []
        for i, par in enumerate(hand_parents):
            if par != -1:
                hand_edge_index.append([par, i])  # 父→子
                hand_edge_index.append([i, par])  # 子→父（无向）
        self.hand_edge_index = torch.tensor(hand_edge_index, dtype=torch.long).t().contiguous()

        # Define hand_triples for angle loss
        self.hand_triples = self._initialize_hand_triples()

        # Cache templates
        self.register_buffer('body_edge_index_template', self.body_edge_index)
        self.register_buffer('hand_edge_index_template', self.hand_edge_index)

        # Enhanced Body Decoder: Added ResBlocks, more Attention, and GCN for body joints
        self.body_decoder_pre = nn.Sequential(
            ResBlock(out_channels, type='1d', p=p),
            ConvNormRelu(out_channels, out_channels, type='1d', leaky=True, downsample=False, p=p),
            ChannelAttention(out_channels),
            SelfAttention(out_channels)
        )
        self.body_proj_in = nn.Linear(out_channels, self.num_body_joints * self.joint_feat_dim)
        # Multi-layer GCN with GAT for attention-based graph processing (increased to 5 layers)
        self.body_gcn1 = GATConv(self.joint_feat_dim, self.joint_feat_dim, heads=4, concat=False)
        self.body_gcn2 = GraphConv(self.joint_feat_dim, self.joint_feat_dim)
        self.body_gcn3 = GATConv(self.joint_feat_dim, self.joint_feat_dim, heads=4, concat=False)
        self.body_gcn4 = GraphConv(self.joint_feat_dim, self.joint_feat_dim)  # Additional layer
        self.body_gcn5 = GATConv(self.joint_feat_dim, self.joint_feat_dim, heads=4, concat=False)  # Additional layer
        self.body_layer_norms = nn.ModuleList([nn.LayerNorm(self.joint_feat_dim) for _ in range(5)])
        self.body_relu = nn.LeakyReLU(0.2)
        self.body_dropout = nn.Dropout(p=p)
        self.body_proj_out = nn.Linear(self.num_body_joints * self.joint_feat_dim, out_channels)
        self.body_norm = nn.LayerNorm(out_channels)
        self.body_decoder_post = nn.Sequential(
            ResBlock(out_channels, type='1d', p=p),
            ConvNormRelu(out_channels, out_channels, type='1d', leaky=True, downsample=False, p=p),
            SelfAttention(out_channels)
        )
        self.body_logits = nn.Conv1d(out_channels, self.body_feats, kernel_size=1, stride=1)

        # Enhanced Hand Decoder: More layers, upgraded to multi-GAT/GCN, added ResBlock
        self.hand_decoder_pre = nn.Sequential(
            ResBlock(out_channels, type='1d', p=p),  # Added ResBlock
            ConvNormRelu(out_channels, out_channels, type='1d', leaky=True, downsample=False, p=p),
            SelfAttention(out_channels),
            ChannelAttention(out_channels)
        )
        self.hand_proj_in = nn.Linear(out_channels, self.num_hand_joints * self.joint_feat_dim)
        # Multi-layer GCN/GAT for enhanced hand modeling (increased to 5 layers)
        self.hand_gcn1 = GATConv(self.joint_feat_dim, self.joint_feat_dim, heads=4, concat=False)
        self.hand_gcn2 = GraphConv(self.joint_feat_dim, self.joint_feat_dim)
        self.hand_gcn3 = GATConv(self.joint_feat_dim, self.joint_feat_dim, heads=4, concat=False)
        self.hand_gcn4 = GraphConv(self.joint_feat_dim, self.joint_feat_dim)  # Additional layer
        self.hand_gcn5 = GATConv(self.joint_feat_dim, self.joint_feat_dim, heads=4, concat=False)  # Additional layer
        self.hand_layer_norms = nn.ModuleList([nn.LayerNorm(self.joint_feat_dim) for _ in range(5)])
        self.hand_relu = nn.LeakyReLU(0.2)
        self.hand_dropout = nn.Dropout(p=p)
        self.hand_proj_out = nn.Linear(self.num_hand_joints * self.joint_feat_dim, out_channels)
        self.hand_norm = nn.LayerNorm(out_channels)
        self.hand_decoder_post = nn.Sequential(
            ResBlock(out_channels, type='1d', p=p),  # Added ResBlock
            ConvNormRelu(out_channels, out_channels, type='1d', leaky=True, downsample=False, p=p),
            SelfAttention(out_channels),
            ChannelAttention(out_channels)  # Added extra attention
        )
        self.hand_logits = nn.Conv1d(out_channels, self.hand_feats, kernel_size=1, stride=1)

        # Bone loss setup (as original, assuming joint_subset is defined elsewhere or here)
        self.parents = self.skeleton.parents
        self.joint_names = self.skeleton.joint_names
        # Example joint_subset; adjust as needed
        self.joint_subset = list(range(out_feats // 2))  # Assuming 2D poses, half for x/y

        # Initialize body triples for body angle constraints
        self.body_triples = self._initialize_body_triples()

    def _expand_edge_index(self, edge_index, num_nodes, batch_size):
        """
        Vectorized edge index expansion for efficient batch GCN processing.
        Avoids creating individual Data objects and using Batch.from_data_list().

        Args:
            edge_index: [2, num_edges] template edge index
            num_nodes: number of nodes per graph
            batch_size: number of graphs in batch

        Returns:
            Expanded edge index [2, batch_size * num_edges]
        """
        # Create offsets for each graph in batch
        offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
        offsets = offsets.view(-1, 1, 1)  # [batch_size, 1, 1]

        # Expand edge_index for all graphs
        edge_index_expanded = edge_index.unsqueeze(0) + offsets  # [batch_size, 2, num_edges]
        edge_index_expanded = edge_index_expanded.permute(1, 0, 2).reshape(2, -1)  # [2, batch_size * num_edges]

        return edge_index_expanded

    def forward(self, audio, real_pose=None):
        # Audio encoding
        audio_feats = self.audio_encoder(audio)  # [B, 256, time_steps]
        refined_feats = self.unet(audio_feats)  # [B, 256, time_steps]

        # Body decoding
        body_x = self.body_decoder_pre(refined_feats)  # [B, C, T]
        B, C, T = body_x.shape
        body_x = body_x.permute(0, 2, 1)  # [B, T, C] for linear proj
        body_x = self.body_proj_in(body_x)  # [B, T, num_body_joints * joint_feat_dim]
        body_x = body_x.view(B * T, self.num_body_joints, self.joint_feat_dim)  # [B*T, joints, feat]

        # Optimized batch body graphs - vectorized edge index expansion
        body_x_flat = body_x.view(-1, self.joint_feat_dim)  # [B*T*num_joints, feat]
        body_edge_expanded = self._expand_edge_index(
            self.body_edge_index_template, self.num_body_joints, B * T
        )

        # 5-layer GCN with residual connections and layer normalization
        body_residual = body_x_flat
        body_x_flat = self.body_gcn1(body_x_flat, body_edge_expanded)
        body_x_flat = body_x_flat.view(-1, self.num_body_joints, self.joint_feat_dim)
        body_x_flat = self.body_layer_norms[0](body_x_flat).view(-1, self.joint_feat_dim)
        body_x_flat = self.body_relu(body_x_flat) + body_residual  # Residual

        body_residual = body_x_flat
        body_x_flat = self.body_gcn2(body_x_flat, body_edge_expanded)
        body_x_flat = body_x_flat.view(-1, self.num_body_joints, self.joint_feat_dim)
        body_x_flat = self.body_layer_norms[1](body_x_flat).view(-1, self.joint_feat_dim)
        body_x_flat = self.body_relu(body_x_flat) + body_residual

        body_residual = body_x_flat
        body_x_flat = self.body_gcn3(body_x_flat, body_edge_expanded)
        body_x_flat = body_x_flat.view(-1, self.num_body_joints, self.joint_feat_dim)
        body_x_flat = self.body_layer_norms[2](body_x_flat).view(-1, self.joint_feat_dim)
        body_x_flat = self.body_relu(body_x_flat) + body_residual

        body_residual = body_x_flat
        body_x_flat = self.body_gcn4(body_x_flat, body_edge_expanded)
        body_x_flat = body_x_flat.view(-1, self.num_body_joints, self.joint_feat_dim)
        body_x_flat = self.body_layer_norms[3](body_x_flat).view(-1, self.joint_feat_dim)
        body_x_flat = self.body_relu(body_x_flat) + body_residual

        body_residual = body_x_flat
        body_x_flat = self.body_gcn5(body_x_flat, body_edge_expanded)
        body_x_flat = body_x_flat.view(-1, self.num_body_joints, self.joint_feat_dim)
        body_x_flat = self.body_layer_norms[4](body_x_flat).view(-1, self.joint_feat_dim)
        body_x_flat = self.body_relu(body_x_flat) + body_residual

        body_x_flat = self.body_dropout(body_x_flat)

        body_x = body_x_flat.view(B, T, self.num_body_joints * self.joint_feat_dim)  # [B, T, joints*feat]
        body_x = self.body_proj_out(body_x)  # [B, T, C]
        body_x = self.body_norm(body_x)
        body_x = body_x.permute(0, 2, 1)  # [B, C, T]
        body_x = self.body_decoder_post(body_x)
        body_out = self.body_logits(body_x)  # [B, body_feats, T]

        # Hand decoding (similar optimized structure)
        hand_x = self.hand_decoder_pre(refined_feats)  # [B, C, T]
        hand_x = hand_x.permute(0, 2, 1)  # [B, T, C]
        hand_x = self.hand_proj_in(hand_x)  # [B, T, num_hand_joints * joint_feat_dim]
        hand_x = hand_x.view(B * T, self.num_hand_joints, self.joint_feat_dim)  # [B*T, joints, feat]

        # Optimized batch hand graphs - vectorized edge index expansion
        hand_x_flat = hand_x.view(-1, self.joint_feat_dim)  # [B*T*num_joints, feat]
        hand_edge_expanded = self._expand_edge_index(
            self.hand_edge_index_template, self.num_hand_joints, B * T
        )

        # 5-layer GCN with residual connections and layer normalization
        hand_residual = hand_x_flat
        hand_x_flat = self.hand_gcn1(hand_x_flat, hand_edge_expanded)
        hand_x_flat = hand_x_flat.view(-1, self.num_hand_joints, self.joint_feat_dim)
        hand_x_flat = self.hand_layer_norms[0](hand_x_flat).view(-1, self.joint_feat_dim)
        hand_x_flat = self.hand_relu(hand_x_flat) + hand_residual  # Residual

        hand_residual = hand_x_flat
        hand_x_flat = self.hand_gcn2(hand_x_flat, hand_edge_expanded)
        hand_x_flat = hand_x_flat.view(-1, self.num_hand_joints, self.joint_feat_dim)
        hand_x_flat = self.hand_layer_norms[1](hand_x_flat).view(-1, self.joint_feat_dim)
        hand_x_flat = self.hand_relu(hand_x_flat) + hand_residual

        hand_residual = hand_x_flat
        hand_x_flat = self.hand_gcn3(hand_x_flat, hand_edge_expanded)
        hand_x_flat = hand_x_flat.view(-1, self.num_hand_joints, self.joint_feat_dim)
        hand_x_flat = self.hand_layer_norms[2](hand_x_flat).view(-1, self.joint_feat_dim)
        hand_x_flat = self.hand_relu(hand_x_flat) + hand_residual

        hand_residual = hand_x_flat
        hand_x_flat = self.hand_gcn4(hand_x_flat, hand_edge_expanded)
        hand_x_flat = hand_x_flat.view(-1, self.num_hand_joints, self.joint_feat_dim)
        hand_x_flat = self.hand_layer_norms[3](hand_x_flat).view(-1, self.joint_feat_dim)
        hand_x_flat = self.hand_relu(hand_x_flat) + hand_residual

        hand_residual = hand_x_flat
        hand_x_flat = self.hand_gcn5(hand_x_flat, hand_edge_expanded)
        hand_x_flat = hand_x_flat.view(-1, self.num_hand_joints, self.joint_feat_dim)
        hand_x_flat = self.hand_layer_norms[4](hand_x_flat).view(-1, self.joint_feat_dim)
        hand_x_flat = self.hand_relu(hand_x_flat) + hand_residual

        hand_x_flat = self.hand_dropout(hand_x_flat)

        hand_x = hand_x_flat.view(B, T, self.num_hand_joints * self.joint_feat_dim)  # [B, T, joints*feat]
        hand_x = self.hand_proj_out(hand_x)  # [B, T, C]
        hand_x = self.hand_norm(hand_x)
        hand_x = hand_x.permute(0, 2, 1)  # [B, C, T]
        hand_x = self.hand_decoder_post(hand_x)
        hand_out = self.hand_logits(hand_x)  # [B, hand_feats, T]

        # Combine body and hand outputs
        out = torch.cat([body_out, hand_out], dim=1)  # [B, out_feats, T]
        out = out.transpose(1, 2)  # [B, T, out_feats] for pose sequence

        # Internal losses (as original)
        internal_losses = []
        if real_pose is not None:
            bone_loss = self.compute_bone_length_loss(real_pose, out)
            internal_losses.append(bone_loss)

        # 使用综合角度损失（包括手部和身体关节）
        angle_loss = self.compute_comprehensive_angle_loss(out)
        internal_losses.append(angle_loss)

        return out, internal_losses

    def _initialize_hand_triples(self):
        """Initialize hand_triples for angle loss computation."""
        hand_triples = []
        for i in range(self.num_hand_joints):
            parent_idx = self.skeleton.parents[i + 10] - 10 if self.skeleton.parents[i + 10] >= 10 else -1
            if parent_idx != -1:
                # Find a child (next joint in hierarchy if exists)
                for j in range(i + 1, self.num_hand_joints):
                    if self.skeleton.parents[j + 10] - 10 == i:
                        hand_triples.append((parent_idx, i, j))
                        break
        return hand_triples

    def _initialize_body_triples(self):
        """Initialize body_triples for body angle loss computation."""
        body_triples = []
        for i in range(self.num_body_joints):
            parent_idx = self.skeleton.parents[i] if self.skeleton.parents[i] < self.num_body_joints else -1
            if parent_idx != -1:
                # Find children
                for j in range(i + 1, self.num_body_joints):
                    if self.skeleton.parents[j] == i:
                        body_triples.append((parent_idx, i, j))
                        break
        return body_triples


    def compute_bone_length_loss(self, real_pose, gen_pose):
        """
        计算骨长损失：L2损失于生成和真实姿势的骨长之间。
        - real_pose: [B, T, 104] 真实姿势（x,y坐标展平）
        - gen_pose: [B, T, 104] 生成姿势
        返回: 标量损失
        """
        B, T, _ = real_pose.shape
        num_joints = len(self.parents)  # 52关节
        assert real_pose.shape[-1] == num_joints * 2, "Pose dimension mismatch"

        # 重塑为 [B, T, num_joints, 2] (x,y坐标)
        real_pose = real_pose.view(B, T, num_joints, 2)
        gen_pose = gen_pose.view(B, T, num_joints, 2)

        # 只计算joint_subset中的关节（去除无关如鼻子）
        real_pose = real_pose[:, :, self.joint_subset, :]
        gen_pose = gen_pose[:, :, self.joint_subset, :]
        parents_subset = np.array(self.parents)[self.joint_subset]  # 调整父节点索引

        # 调整父节点索引
        parents_subset = np.array([np.where(self.joint_subset == p)[0][0] if p != -1 else -1 for p in parents_subset])

        # 计算骨长：每个关节到父关节的欧几里得距离
        def get_bone_lengths(pose, parents):
            bone_lengths = []
            for i in range(len(parents)):
                if parents[i] != -1:
                    bone_vec = pose[:, :, i, :] - pose[:, :, parents[i], :]  # [B, T, 2]
                    bone_len = torch.norm(bone_vec, dim=-1)  # [B, T]
                    bone_lengths.append(bone_len)
            bone_lengths = torch.stack(bone_lengths, dim=-1)  # [B, T, num_bones]
            return bone_lengths.mean(dim=1)  # 平均过时间，[B, num_bones]

        real_bone_lens = get_bone_lengths(real_pose, parents_subset)
        gen_bone_lens = get_bone_lengths(gen_pose, parents_subset)

        # L2损失（MSE）于骨长
        bone_loss = nn.MSELoss()(gen_bone_lens, real_bone_lens)

        return bone_loss


    def compute_hand_joint_angle_loss(self, gen_pose):
        """
        计算手部关节角度约束损失：只对生成姿势的手部关节角度施加范围约束（0-180度），并惩罚反向弯曲（负角度）。
        - gen_pose: [B, T, 104] 生成姿势
        返回: 标量损失
        """
        B, T, _ = gen_pose.shape
        num_joints = len(self.parents)  # 52关节
        assert gen_pose.shape[-1] == num_joints * 2, "Pose dimension mismatch"

        # 重塑为 [B, T, num_joints, 2] (x,y坐标)
        gen_pose = gen_pose.view(B, T, num_joints, 2)

        # 只取手部关节 (10:52)
        gen_hand_pose = gen_pose[:, :, 10:52, :]  # [B, T, 42, 2]

        # 计算带符号角度：对于每个三元组 (p, j, c)，计算在 j 处的带符号角度
        def get_joint_angles(hand_pose):
            angles = []
            for p, j, c in self.hand_triples:
                # vec_pj: from p to j (parent to joint)
                vec_pj = hand_pose[:, :, j, :] - hand_pose[:, :, p, :]  # [B, T, 2]
                # vec_jc: from j to c (joint to child)
                vec_jc = hand_pose[:, :, c, :] - hand_pose[:, :, j, :]  # [B, T, 2]
                dot_product = torch.sum(vec_pj * vec_jc, dim=-1)  # [B, T]
                cross = vec_pj[:, :, 0] * vec_jc[:, :, 1] - vec_pj[:, :, 1] * vec_jc[:, :, 0]  # [B, T]，带符号叉积
                angle = torch.atan2(cross, dot_product)  # [B, T]，角度在[-pi, pi]
                angles.append(angle)
            angles = torch.stack(angles, dim=-1)  # [B, T, num_triples]
            return angles

        gen_angles = get_joint_angles(gen_hand_pose)  # [B, T, num_triples]

        # 定义角度范围：0-180度（0 - pi radians），惩罚负角度（反向弯曲）
        min_angle = 0.0
        max_angle = torch.pi

        # 计算范围约束损失：使用ReLU惩罚超出范围的角度
        lower_penalty = torch.relu(min_angle - gen_angles)  # 惩罚 <0 的角度（反向弯曲）
        upper_penalty = torch.relu(gen_angles - max_angle)  # 惩罚 >180 的角度（通常为0）
        angle_loss = torch.mean(lower_penalty + upper_penalty)

        return angle_loss

    def compute_body_joint_angle_loss(self, gen_pose):
        """
        计算身体关节角度约束损失，确保符合人体物理约束。
        - gen_pose: [B, T, 104] 生成姿势
        返回: 标量损失
        """
        B, T, _ = gen_pose.shape
        num_joints = len(self.parents)  # 52关节

        if gen_pose.shape[-1] != num_joints * 2:
            return torch.tensor(0.0, device=gen_pose.device)

        # 重塑为 [B, T, num_joints, 2] (x,y坐标)
        gen_pose = gen_pose.view(B, T, num_joints, 2)

        # 只取身体关节 (0:10)
        gen_body_pose = gen_pose[:, :, :10, :]  # [B, T, 10, 2]

        # 如果没有body_triples，返回0损失
        if not self.body_triples:
            return torch.tensor(0.0, device=gen_pose.device)

        def get_body_angles(body_pose):
            angles = []
            for p, j, c in self.body_triples:
                # vec_pj: from p to j (parent to joint)
                vec_pj = body_pose[:, :, j, :] - body_pose[:, :, p, :]  # [B, T, 2]
                # vec_jc: from j to c (joint to child)
                vec_jc = body_pose[:, :, c, :] - body_pose[:, :, j, :]  # [B, T, 2]
                dot_product = torch.sum(vec_pj * vec_jc, dim=-1)  # [B, T]
                cross = vec_pj[:, :, 0] * vec_jc[:, :, 1] - vec_pj[:, :, 1] * vec_jc[:, :, 0]  # [B, T]
                angle = torch.atan2(cross, dot_product)  # [B, T]
                angles.append(angle)
            if angles:
                angles = torch.stack(angles, dim=-1)  # [B, T, num_triples]
            else:
                return None
            return angles

        gen_angles = get_body_angles(gen_body_pose)

        if gen_angles is None:
            return torch.tensor(0.0, device=gen_pose.device)

        # 身体关节角度约束：合理范围为 -pi/2 到 pi（允许更大范围的运动）
        min_angle = -torch.pi / 2
        max_angle = torch.pi

        # 计算范围约束损失
        lower_penalty = torch.relu(min_angle - gen_angles)
        upper_penalty = torch.relu(gen_angles - max_angle)
        angle_loss = torch.mean(lower_penalty + upper_penalty)

        return angle_loss

    def compute_comprehensive_angle_loss(self, gen_pose):
        """
        综合角度损失：结合手部和身体关节角度约束。
        - gen_pose: [B, T, 104] 生成姿势
        返回: 标量损失
        """
        hand_angle_loss = self.compute_hand_joint_angle_loss(gen_pose)
        body_angle_loss = self.compute_body_joint_angle_loss(gen_pose)

        # 手部角度更重要，因为手指关节约束更严格
        total_angle_loss = 0.7 * hand_angle_loss + 0.3 * body_angle_loss

        return total_angle_loss


class SelfAttention_D(nn.Module):
    def __init__(self, in_channels=104, out_channels=64, n_downsampling=2, p=0.3, groups=1, aux_classes=10, **kwargs):
        super(SelfAttention_D, self).__init__()
        # Reduced n_downsampling to 2 to prevent excessive time dimension reduction
        self.n_downsampling = n_downsampling
        self.groups = groups
        self.p = p

        # Used for Graph Convolutional Network in graph-based branches
        self.skeleton = Skeleton2D()
        parents = self.skeleton.parents
        self.num_body_joints = 10  # Core body joints
        self.num_hand_joints = 42  # Hand joints
        self.joint_feat_dim = 64  # Feature dim per joint

        # Body edge index
        body_parents = parents[:self.num_body_joints]
        body_parents = [par if par < self.num_body_joints else -1 for par in body_parents]
        body_edge_index = []
        for i, par in enumerate(body_parents):
            if par != -1:
                body_edge_index.append([par, i])
                body_edge_index.append([i, par])
        self.body_edge_index = torch.tensor(body_edge_index, dtype=torch.long).t().contiguous()

        # Hand edge index
        hand_parents = parents[10:10 + self.num_hand_joints]
        hand_parents = [par - 10 if par >= 10 else -1 for par in hand_parents]
        hand_edge_index = []
        for i, par in enumerate(hand_parents):
            if par != -1:
                hand_edge_index.append([par, i])
                hand_edge_index.append([i, par])
        self.hand_edge_index = torch.tensor(hand_edge_index, dtype=torch.long).t().contiguous()

        # Register buffers
        self.register_buffer('body_edge_index_template', self.body_edge_index)
        self.register_buffer('hand_edge_index_template', self.hand_edge_index)

        # Enhanced conv1: Simplified - removed unnecessary attention and ResBlock
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels * groups, out_channels * groups, kernel_size=4, stride=2, padding=1, groups=groups),
            nn.BatchNorm1d(out_channels * groups),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=p),
            nn.Conv1d(out_channels * groups, out_channels * groups, kernel_size=4, stride=1, padding=1, groups=groups),
            nn.BatchNorm1d(out_channels * groups),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=p)
        )

        # Enhanced conv2: Simplified with fewer layers and no attention (reduce discriminator strength)
        self.conv2 = nn.ModuleList()
        current_channels = out_channels * groups
        for n in range(1, n_downsampling + 1):
            ch_mul = min(2 ** n, 16)
            self.conv2.append(nn.Sequential(
                nn.Conv1d(current_channels, current_channels * ch_mul, kernel_size=4, stride=2, padding=1,
                          groups=groups),
                nn.BatchNorm1d(current_channels * ch_mul),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=p),
                nn.Conv1d(current_channels * ch_mul, current_channels * ch_mul, kernel_size=4, stride=1, padding=1,
                          groups=groups),
                nn.BatchNorm1d(current_channels * ch_mul),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=p)
            ))
            current_channels = current_channels * ch_mul

        # Enhanced conv3: Simplified - only one attention at the end
        self.conv3 = nn.Sequential(
            nn.Conv1d(current_channels, current_channels * 2, kernel_size=4, stride=1, padding=1, groups=groups),
            nn.BatchNorm1d(current_channels * 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=p),

            nn.Conv1d(current_channels * 2, current_channels * 4, kernel_size=4, stride=1, padding=1, groups=groups),
            nn.BatchNorm1d(current_channels * 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=p),
            SelfAttention(current_channels * 4),  # 只在最后保留一个attention

            nn.Conv1d(current_channels * 4, current_channels * 4, kernel_size=3, stride=1, padding=1, groups=groups),
            nn.BatchNorm1d(current_channels * 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=p)
        )

        # Graph branches for body and hand (dual discriminators)
        self.body_proj = nn.Linear(current_channels * 4 // 2, self.num_body_joints * self.joint_feat_dim)
        self.hand_proj = nn.Linear(current_channels * 4 // 2, self.num_hand_joints * self.joint_feat_dim)

        self.body_gat = GATConv(self.joint_feat_dim, self.joint_feat_dim, heads=4, concat=False)
        self.hand_gat = GATConv(self.joint_feat_dim, self.joint_feat_dim, heads=4, concat=False)

        self.body_graph_out = nn.Linear(self.num_body_joints * self.joint_feat_dim, current_channels * 2)
        self.hand_graph_out = nn.Linear(self.num_hand_joints * self.joint_feat_dim, current_channels * 2)

        # Fusion for multi-modal (if audio provided)
        self.audio_fusion = nn.Conv1d(256, current_channels * 4, kernel_size=1)  # Assuming audio feats dim 256

        # Logits for discrimination
        out_shape = 1 if 'out_shape' not in kwargs else kwargs['out_shape']
        self.logits = nn.Conv1d(current_channels * groups * 4 * 2, out_shape * groups, kernel_size=3, stride=1,
                                padding=1, groups=groups)  # Adjusted kernel to 3

        # Auxiliary task: Classifier for gesture types (e.g., 10 classes)
        self.aux_classifier = nn.Sequential(
            nn.Linear(current_channels * 4, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p),
            nn.Linear(512, aux_classes)
        )
        self.aux_loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, audio=None, aux_labels=None):
        # Input shape check and padding
        x = x.transpose(-1, -2)  # (N, pose_feats, time)
        if x.size(2) < 4:  # Ensure minimum time dimension
            x = nn.functional.pad(x, (0, 4 - x.size(2) % 4))  # Pad to nearest multiple of 4

        x = self.conv1(x)
        for layer in self.conv2:
            x = layer(x)

        x = self.conv3(x)  # (N, high_channels, small_time)


        # Split for dual graph branches (body/hand)
        B, C, T = x.shape
        x_body = x[:, :C // 2, :]  # Split channels for body
        x_hand = x[:, C // 2:, :]  # For hand

        # Project and process body graph
        x_body = x_body.mean(dim=2)  # Global avg pool for simplicity
        x_body = self.body_proj(x_body)
        x_body = x_body.view(B, self.num_body_joints, self.joint_feat_dim)
        body_data_list = [Data(x=x_body[i], edge_index=self.body_edge_index_template) for i in range(B)]
        body_batch = Batch.from_data_list(body_data_list)
        x_body = self.body_gat(body_batch.x, body_batch.edge_index)
        x_body = x_body.view(B, -1)
        x_body = self.body_graph_out(x_body)

        # Hand graph
        x_hand = x_hand.mean(dim=2)
        x_hand = self.hand_proj(x_hand)
        x_hand = x_hand.view(B, self.num_hand_joints, self.joint_feat_dim)
        hand_data_list = [Data(x=x_hand[i], edge_index=self.hand_edge_index_template) for i in range(B)]
        hand_batch = Batch.from_data_list(hand_data_list)
        x_hand = self.hand_gat(hand_batch.x, hand_batch.edge_index)
        x_hand = x_hand.view(B, -1)
        x_hand = self.hand_graph_out(x_hand)

        # Fuse body/hand graph outputs back
        x_graph = torch.cat([x_body, x_hand], dim=1)
        x_graph = x_graph.unsqueeze(2).repeat(1, 1, T)
        x = torch.cat([x, x_graph], dim=1)

        # Multi-modal fusion if audio provided
        if audio is not None:
            audio = self.audio_fusion(audio)
            if audio.shape[2] != T:
                audio = nn.functional.adaptive_avg_pool1d(audio, T)
            x = torch.cat([x, audio], dim=1)

        # Final logits
        x = self.logits(x)
        x = x.transpose(-1, -2).squeeze(dim=-1)

        # Auxiliary loss if labels provided
        internal_losses = []
        if aux_labels is not None:
            aux_feats = x.mean(dim=1) if x.dim() > 1 else x
            aux_out = self.aux_classifier(aux_feats)
            aux_loss = self.aux_loss_fn(aux_out, aux_labels)
            internal_losses.append(aux_loss)

        return x, internal_losses


