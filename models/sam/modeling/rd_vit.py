# from typing import Any, Optional, Tuple, Type
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from typing import Optional, Tuple, Type
#
# # class DownscaleAdapter(nn.Module):
# #     def __init__(self, in_channels: int, out_channels: int):
# #         super().__init__()
# #         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
# #         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
# #         self.batch_norm = nn.BatchNorm2d(out_channels)
# #         self.gelu = nn.GELU()
# #
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         ##BHWC
# #         x = x.permute(0, 3, 1, 2)
# #         # print("x.permute",x.shape)
# #         ###BCHW
# #         x = self.conv1(x)
# #         x = self.conv2(x)
# #         x = self.batch_norm(x)
# #         x = self.gelu(x)
# #         x = x.permute(0, 2, 3, 1)
# #         print('DownscaleAdapter之后x的形状', x.shape)
# #         return x
# #
# #
# class FeatureAdapter(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, mlp_ratio: float = 4.0):
#         super().__init__()
#
#         # Down linear projection (MLPdown)
#         self.mlp_down = nn.Linear(in_channels, int(in_channels * mlp_ratio))
#
#         # GELU activation function
#         self.gelu = nn.GELU()
#
#         # Up linear projection (MLPup)
#         self.mlp_up = nn.Linear(int(in_channels * mlp_ratio), out_channels)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # First apply the down-projection MLP and GELU activation
#         x = self.mlp_down(x)
#         x = self.gelu(x)
#
#         # Then apply the up-projection MLP
#         x = self.mlp_up(x)
#         # print('FeatureAdapter之后x的形状', x.shape)
#         return x
#
# class LayerNorm2d(nn.Module):
#     def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(num_channels))
#         self.bias = nn.Parameter(torch.zeros(num_channels))
#         self.eps = eps
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         u = x.mean(1, keepdim=True)
#         s = (x - u).pow(2).mean(1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.eps)
#         x = self.weight[:, None, None] * x + self.bias[:, None, None]
#
#         return x
#
# class MLPBlock(nn.Module):
#     def __init__(
#         self,
#         embedding_dim: int,
#         mlp_dim: int,
#         act: Type[nn.Module] = nn.GELU,
#     ) -> None:
#         super().__init__()
#         self.lin1 = nn.Linear(embedding_dim, mlp_dim)
#         self.lin2 = nn.Linear(mlp_dim, embedding_dim)
#         self.act = act()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.lin2(self.act(self.lin1(x)))
#
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple, List
#
#
# # 假设你已有的相关模块已经导入（例如 PatchEmbed, DownscaleAdapter, ParallelBlock, LayerNorm2d, etc.）
#
# class RDViTEncoder(nn.Module):
#     def __init__(
#             self,
#             img_size: int = 1024,
#             patch_size: int = 16,
#             in_chans: int = 3,
#             embed_dim: int = 768,
#             depth: int = 4,
#             num_heads: int = 8,
#             mlp_ratio: float = 4.0,
#             out_chans: int = 256,
#             qkv_bias: bool = True,
#             norm_layer: type = nn.LayerNorm,
#             act_layer: type = nn.GELU,
#             use_abs_pos: bool = True,
#             use_rel_pos: bool = False,
#             rel_pos_zero_init: bool = True,
#             window_size: int = 0,
#             global_attn_indexes: Tuple[int, ...] = (),
#     ) -> None:
#         super().__init__()
#         self.img_size = img_size
#
#         # Patch Embedding：输出形状 [B, H_patch, W_patch, embed_dim]
#         self.patch_embed = PatchEmbed(
#             kernel_size=(patch_size, patch_size),
#             stride=(patch_size, patch_size),
#             in_chans=in_chans,
#             embed_dim=embed_dim,
#         )
#
#         # Positional Embedding（以绝对位置为例）
#         self.pos_embed: nn.Parameter = None
#         if use_abs_pos:
#             # 假设 patch 数为 (img_size // patch_size) 维度
#             num_patches = img_size // patch_size
#             self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dim))
#             # 可选：初始化 pos_embed
#             nn.init.trunc_normal_(self.pos_embed, std=0.02)
#
#         # 构造多个并行 Transformer block
#         self.blocks = nn.ModuleList()
#         for i in range(depth):
#             block = RGB_CRIM_FusionBlock(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 norm_layer=norm_layer,
#                 act_layer=act_layer,
#                 use_rel_pos=use_rel_pos,
#                 rel_pos_zero_init=rel_pos_zero_init,
#                 window_size=window_size if i not in global_attn_indexes else 0,
#                 input_size=(img_size // patch_size, img_size // patch_size),
#             )
#             self.blocks.append(block)
#
#         # Neck 模块：将 token 特征转为最终输出图，输出形状 [B, out_chans, H_out, W_out]
#         self.neck = nn.Sequential(
#             nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
#             LayerNorm2d(out_chans),
#             nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
#             LayerNorm2d(out_chans),
#         )
#     def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
#         if y is None:
#             x = self.patch_embed(x)
#             y = torch.zeros_like(x)  # ⚠️ 创建假 y 分支，避免 None 报错
#         else:
#             x = self.patch_embed(x)
#             y = self.patch_embed(y)
#
#         # 3. 添加位置编码（下采样 pos_embed 一致，确保与 x/y 对齐）
#         if self.pos_embed is not None:
#             pos = self.pos_embed  # 调整为与 x/y 同样的空间尺寸
#             x = x + pos
#             y = y + pos
#
#         # 4. 依次通过每个 block，并收集高分辨率特征
#         high_res_features = []
#         for blk in self.blocks:
#             out1, out2 = blk(x, y)  # 假设 blk 接受 (x, y) 并输出形状仍然为 [B, H, W, embed_dim]
#             # 更新 x 用于连续处理（也可以选择不更新 y）
#             x = out1
#             y = out2
#             out = out1 + out2
#             # 保存当前的高分辨率特征（转换到 [B, embed_dim, H, W]）
#             high_res_features.append(out.permute(0, 3, 1, 2))
#
#
#         # 5. 最终 Neck 处理：将最后一个 block 的输出转换成最终的图像特征
#         final_out = self.neck(out.permute(0, 3, 1, 2))
#         # print("最终输出的尺寸", final_out.shape)
#
#         # 6. 返回二元组：最终特征图和最后两层（或根据需求选择）的高分辨率特征
#         return final_out, high_res_features[:]
#
#
#
# class Block(nn.Module):
#     """Transformer blocks with support of window attention and residual propagation blocks"""
#
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         mlp_ratio: float = 4.0,
#         qkv_bias: bool = True,
#         norm_layer: Type[nn.Module] = nn.LayerNorm,
#         act_layer: Type[nn.Module] = nn.GELU,
#         use_rel_pos: bool = False,
#         rel_pos_zero_init: bool = True,
#         window_size: int = 0,
#         input_size: Optional[Tuple[int, int]] = None,
#     ) -> None:
#
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             use_rel_pos=use_rel_pos,
#             rel_pos_zero_init=rel_pos_zero_init,
#             input_size=input_size if window_size == 0 else (window_size, window_size),
#         )
#
#         self.norm2 = norm_layer(dim)
#         self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
#         self.window_size = window_size
#         self.FeatureAdapter = FeatureAdapter(in_channels=dim, out_channels=dim)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         shortcut = x
#         x = self.norm1(x)
#         # Window partition
#         if self.window_size > 0:
#             H, W = x.shape[1], x.shape[2]
#             x, pad_hw = window_partition(x, self.window_size)
#
#         x = self.attn(x)
#
#         # Reverse window partition
#         if self.window_size > 0:
#             x = window_unpartition(x, self.window_size, pad_hw, (H, W))
#
#         x = shortcut + x
#         x_F = self.FeatureAdapter(x)
#         x = x + self.mlp(self.norm2(x)) + x_F
#         return x
#
# class NewTransformerBlock(nn.Module):
#     """Transformer block supporting both Window Attention and Global Attention.
#        Only Global Attention uses FeatureAdapter (no module registered if local window attention)."""
#
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         mlp_ratio: float = 4.0,
#         qkv_bias: bool = True,
#         norm_layer: Type[nn.Module] = nn.LayerNorm,
#         act_layer: Type[nn.Module] = nn.GELU,
#         use_rel_pos: bool = False,
#         rel_pos_zero_init: bool = True,
#         window_size: int = 0,
#         input_size: Optional[Tuple[int, int]] = None,
#     ) -> None:
#         super().__init__()
#
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             use_rel_pos=use_rel_pos,
#             rel_pos_zero_init=rel_pos_zero_init,
#             input_size=input_size if window_size == 0 else (window_size, window_size),
#         )
#
#         self.norm2 = norm_layer(dim)
#         self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
#
#         self.window_size = window_size
#
#         if self.window_size == 0:
#             self.feature_adapter = FeatureAdapter(in_channels=dim, out_channels=dim)
#         else:
#             self.feature_adapter = None  # ⚡ 注意是 None，不实例化！
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         shortcut = x
#         x = self.norm1(x)
#
#         # Window attention
#         if self.window_size > 0:
#             H, W = x.shape[1], x.shape[2]
#             x, pad_hw = window_partition(x, self.window_size)
#             x = self.attn(x)
#             x = window_unpartition(x, self.window_size, pad_hw, (H, W))
#         else:
#             # Global attention
#             B, H, W, C = x.shape
#             x = x.view(B, H * W, C)
#             x = self.attn(x)
#             x = x.view(B, H, W, C)
#
#         x = shortcut + x
#
#         shortcut2 = x
#         x = self.norm2(x)
#         x = self.mlp(x)
#
#         # Only global attention branch adds feature adapter
#         if self.feature_adapter is not None:
#             x = shortcut2 + x + self.feature_adapter(shortcut2)
#         else:
#             x = shortcut2 + x
#
#         return x
#
#
#
#
# class RGBBranch(nn.Module):
#     """RGB 分支，由两个 Window Attention + 一个 Global Attention Block组成"""
#
#     def __init__(
#         self,
#         dim: int,
#         window_size: int = 7,
#         num_heads: int = 8,
#         mlp_ratio: float = 4.0,
#         qkv_bias: bool = True,
#         norm_layer: Type[nn.Module] = nn.LayerNorm,
#         act_layer: Type[nn.Module] = nn.GELU,
#         use_rel_pos: bool = False,
#         rel_pos_zero_init: bool = True,
#         input_size: Optional[Tuple[int, int]] = None,
#     ):
#         super().__init__()
#
#         self.block1 = NewTransformerBlock(
#             dim=dim,
#             num_heads=num_heads,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             norm_layer=norm_layer,
#             act_layer=act_layer,
#             use_rel_pos=use_rel_pos,
#             rel_pos_zero_init=rel_pos_zero_init,
#             window_size=window_size,
#             input_size=input_size,
#         )
#
#         self.block2 = NewTransformerBlock(
#             dim=dim,
#             num_heads=num_heads,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             norm_layer=norm_layer,
#             act_layer=act_layer,
#             use_rel_pos=use_rel_pos,
#             rel_pos_zero_init=rel_pos_zero_init,
#             window_size=window_size,
#             input_size=input_size,
#         )
#
#         self.block3 = NewTransformerBlock(
#             dim=dim,
#             num_heads=num_heads,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             norm_layer=norm_layer,
#             act_layer=act_layer,
#             use_rel_pos=use_rel_pos,
#             rel_pos_zero_init=rel_pos_zero_init,
#             window_size=0,  # Global Attention
#             input_size=input_size,
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         return x
#
#
# class CRIMBranch(nn.Module):
#     """CRIM 分支，由两个 Window Attention + 一个 Global Attention Block组成"""
#
#     def __init__(
#         self,
#         dim: int,
#         window_size: int = 7,
#         num_heads: int = 8,
#         mlp_ratio: float = 4.0,
#         qkv_bias: bool = True,
#         norm_layer: Type[nn.Module] = nn.LayerNorm,
#         act_layer: Type[nn.Module] = nn.GELU,
#         use_rel_pos: bool = False,
#         rel_pos_zero_init: bool = True,
#         input_size: Optional[Tuple[int, int]] = None,
#     ):
#         super().__init__()
#
#         self.block1 = NewTransformerBlock(
#             dim=dim,
#             num_heads=num_heads,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             norm_layer=norm_layer,
#             act_layer=act_layer,
#             use_rel_pos=use_rel_pos,
#             rel_pos_zero_init=rel_pos_zero_init,
#             window_size=window_size,
#             input_size=input_size,
#         )
#
#         self.block2 = NewTransformerBlock(
#             dim=dim,
#             num_heads=num_heads,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             norm_layer=norm_layer,
#             act_layer=act_layer,
#             use_rel_pos=use_rel_pos,
#             rel_pos_zero_init=rel_pos_zero_init,
#             window_size=window_size,
#             input_size=input_size,
#         )
#
#         self.block3 = NewTransformerBlock(
#             dim=dim,
#             num_heads=num_heads,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             norm_layer=norm_layer,
#             act_layer=act_layer,
#             use_rel_pos=use_rel_pos,
#             rel_pos_zero_init=rel_pos_zero_init,
#             window_size=0,  # Global Attention
#             input_size=input_size,
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         return x
#
# class CrossBranchAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=True):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = (dim // num_heads) ** -0.5
#
#         self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim)
#
#     def forward(self, q_input, kv_input):
#         """
#         Args:
#             q_input: (B, N, C) - 来自CRIM分支（Query）
#             kv_input: (B, N, C) - 来自RGB分支（Key和Value）
#         Returns:
#             输出更新后的 Query (B, N, C)
#         """
#         B, N, C = q_input.shape
#
#         # 做 Q, K, V 投影
#         q = self.q_proj(q_input).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, heads, N, head_dim)
#         kv = self.kv_proj(kv_input).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]
#
#         # Cross Attention
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         out = (attn @ v).transpose(1, 2).reshape(B, N, C)
#
#         out = self.proj(out)
#         return out
#
# class RGB_CRIM_FusionBlock(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         window_size: int = 7,
#         num_heads: int = 8,
#         mlp_ratio: float = 4.0,
#         qkv_bias: bool = True,
#         norm_layer: Type[nn.Module] = nn.LayerNorm,
#         act_layer: Type[nn.Module] = nn.GELU,
#         use_rel_pos: bool = False,
#         rel_pos_zero_init: bool = True,
#         input_size: Optional[Tuple[int, int]] = None,
#     ):
#         super().__init__()
#
#         # RGB分支：局部+局部+全局
#         self.rgb_window1 = NewTransformerBlock(
#             dim=dim,
#             num_heads=num_heads,
#             window_size=window_size,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             norm_layer=norm_layer,
#             act_layer=act_layer,
#             use_rel_pos=use_rel_pos,
#             rel_pos_zero_init=rel_pos_zero_init,
#             input_size=input_size
#         )
#         self.rgb_window2 = NewTransformerBlock(
#             dim=dim,
#             num_heads=num_heads,
#             window_size=window_size,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             norm_layer=norm_layer,
#             act_layer=act_layer,
#             use_rel_pos=use_rel_pos,
#             rel_pos_zero_init=rel_pos_zero_init,
#             input_size=input_size
#         )
#         self.rgb_global = NewTransformerBlock(
#             dim=dim,
#             num_heads=num_heads,
#             window_size=0,  # global attention
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             norm_layer=norm_layer,
#             act_layer=act_layer,
#             use_rel_pos=use_rel_pos,
#             rel_pos_zero_init=rel_pos_zero_init,
#             input_size=input_size
#         )
#
#         # CRIM分支（修正）
#         self.crim_window1 = NewTransformerBlock(
#             dim=dim,
#             num_heads=num_heads,
#             window_size=window_size,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             norm_layer=norm_layer,
#             act_layer=act_layer,
#             use_rel_pos=use_rel_pos,
#             rel_pos_zero_init=rel_pos_zero_init,
#             input_size=input_size
#         )
#         self.crim_window2 = NewTransformerBlock(
#             dim=dim,
#             num_heads=num_heads,
#             window_size=window_size,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             norm_layer=norm_layer,
#             act_layer=act_layer,
#             use_rel_pos=use_rel_pos,
#             rel_pos_zero_init=rel_pos_zero_init,
#             input_size=input_size
#         )
#         self.crim_global = NewTransformerBlock(
#             dim=dim,
#             num_heads=num_heads,
#             window_size=0,  # global attention
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             norm_layer=norm_layer,
#             act_layer=act_layer,
#             use_rel_pos=use_rel_pos,
#             rel_pos_zero_init=rel_pos_zero_init,
#             input_size=input_size
#         )
#
#         # Cross-branch Attention
#         self.cross_branch_attention = CrossBranchAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
#
#         # 对齐SAM结构：FusionBlock内的 norm2 + mlp
#         self.norm2 = norm_layer(dim)
#         self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
#
#     def forward(self, rgb, crim):
#         """
#         Args:
#             rgb: (B, H, W, C)
#             crim: (B, H, W, C)
#         Returns:
#             updated rgb and crim features
#         """
#
#         # RGB分支
#         rgb = self.rgb_window1(rgb)
#         rgb = self.rgb_window2(rgb)
#         rgb = self.rgb_global(rgb)
#
#         # CRIM分支
#         crim = self.crim_window1(crim)
#         crim = self.crim_window2(crim)
#
#         # CrossBranch Attention
#         B, H, W, C = crim.shape
#         crim_flat = crim.view(B, H * W, C)
#         rgb_flat = rgb.view(B, H * W, C)
#         q_updated = self.cross_branch_attention(crim_flat, rgb_flat)
#
#         # Global attention + FeatureAdapter（新的对称结构）
#         x = q_updated.view(B, H, W, C)
#         crim = self.crim_global(x)
#
#         return rgb, crim
#
# # class ParallelBlock(nn.Module):
# #     """Parallel Transformer blocks with CBA fusion module."""
# #
# #     def __init__(
# #             self,
# #             dim: int,
# #             num_heads: int,
# #             mlp_ratio: float = 4.0,
# #             qkv_bias: bool = True,
# #             norm_layer: Type[nn.Module] = nn.LayerNorm,
# #             act_layer: Type[nn.Module] = nn.GELU,
# #             use_rel_pos: bool = False,
# #             rel_pos_zero_init: bool = True,
# #             window_size: int = 0,
# #             input_size: Optional[Tuple[int, int]] = None
# #     ) -> None:
# #         super().__init__()
# #
# #         self.block1 = Block(
# #             dim, num_heads, mlp_ratio, qkv_bias, norm_layer,
# #             act_layer, use_rel_pos, rel_pos_zero_init, window_size, input_size
# #         )
# #
# #         self.block2 = Block(
# #             dim, num_heads, mlp_ratio, qkv_bias, norm_layer,
# #             act_layer, use_rel_pos, rel_pos_zero_init, window_size, input_size
# #         )
# #
# #         self.norm1_branch1 = norm_layer(dim)
# #         self.norm1_branch2 = norm_layer(dim)
# #
# #         # self.otfm = OTFM( 768,768,768 )
# #
# #     def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
# #         norm_x1 = self.norm1_branch1(x1)
# #         norm_x2 = self.norm1_branch2(x2)
# #
# #         # 调整形状适配 Conv2d
# #         norm_x1 = norm_x1.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
# #         norm_x2 = norm_x2.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
# #
# #         # OTFM 融合
# #         fused_output = self.otfm(norm_x1, norm_x2)
# #
# #         # 还原形状
# #         fused_output = fused_output.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
# #
# #         out1 = self.block1(x1)
# #         print('RGB分支的输出尺寸', out1.shape)
# #
# #         # 直接传递 fused_output 给 self.block2.attn
# #         out2 = self.block2.attn(fused_output)
# #         out2 = self.block2.norm2(out2)
# #         out2 = self.block2.mlp(out2) + self.block2.attn(fused_output)
# #         # out2 = self.block2.attn(fused_output)
# #         print("DEM分支的输出尺寸", out2.shape)
# #
# #         print("经过ParallelBlock之后的尺寸",out1.shape,out2.shape)
# #         return out1,out2
#
#
# class Attention(nn.Module):
#     """Multi-head Attention block with relative position embeddings."""
#
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 8,
#         qkv_bias: bool = True,
#         use_rel_pos: bool = False,
#         rel_pos_zero_init: bool = True,
#         input_size: Optional[Tuple[int, int]] = None,
#     ) -> None:
#         """
#         Args:
#             dim (int): Number of input channels.
#             num_heads (int): Number of attention heads.
#             qkv_bias (bool):  If True, add a learnable bias to query, key, value.
#             rel_pos (bool): If True, add relative positional embeddings to the attention map.
#             rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
#             input_size (tuple(int, int) or None): Input resolution for calculating the relative
#                 positional parameter size.
#         """
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim**-0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim)
#
#         self.use_rel_pos = use_rel_pos
#         if self.use_rel_pos:
#             assert (
#                 input_size is not None
#             ), "Input size must be provided if using relative positional encoding."
#             # initialize relative positional embeddings
#             self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
#             self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() == 4:
#             # 输入是 [B, H, W, C]
#             B, H, W, _ = x.shape
#             qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#             q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
#         elif x.dim() == 3:
#             # 输入是 [B, N, C]，比如 global attention
#             B, N, _ = x.shape
#             qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#             q, k, v = qkv.reshape(3, B * self.num_heads, N, -1).unbind(0)
#         else:
#             raise ValueError(f"Unsupported input shape for Attention: {x.shape}")
#
#         attn = (q * self.scale) @ k.transpose(-2, -1)
#         attn = attn.softmax(dim=-1)
#         x = (attn @ v).view(B, self.num_heads, -1, q.shape[-1]).transpose(1, 2).reshape(B, -1,
#                                                                                         self.num_heads * q.shape[-1])
#
#         x = self.proj(x)
#
#         if x.dim() == 3:
#             return x
#         else:
#             return x.view(B, H, W, -1)  # 如果是4D的情况，reshape回来
#
#     # def forward(self, x: torch.Tensor) -> torch.Tensor:
#     #     B, H, W, _ = x.shape
#     #     # qkv with shape (3, B, nHead, H * W, C)
#     #     qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#     #     # q, k, v with shape (B * nHead, H * W, C)
#     #     q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
#     #
#     #     attn = (q * self.scale) @ k.transpose(-2, -1)
#     #
#     #     if self.use_rel_pos:
#     #         attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
#     #
#     #     attn = attn.softmax(dim=-1)
#     #     x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
#     #     x = self.proj(x)
#     #
#     #     return x
#
#
# def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
#     """
#     Partition into non-overlapping windows with padding if needed.
#     Args:
#         x (tensor): input tokens with [B, H, W, C].
#         window_size (int): window size.
#
#     Returns:
#         windows: windows after partition with [B * num_windows, window_size, window_size, C].
#         (Hp, Wp): padded height and width before partition
#     """
#     B, H, W, C = x.shape
#
#     pad_h = (window_size - H % window_size) % window_size
#     pad_w = (window_size - W % window_size) % window_size
#     if pad_h > 0 or pad_w > 0:
#         x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
#     Hp, Wp = H + pad_h, W + pad_w
#
#     x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
#     return windows, (Hp, Wp)
#
#
# def window_unpartition(
#     windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
# ) -> torch.Tensor:
#     """
#     Window unpartition into original sequences and removing padding.
#     Args:
#         windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
#         window_size (int): window size.
#         pad_hw (Tuple): padded height and width (Hp, Wp).
#         hw (Tuple): original height and width (H, W) before padding.
#
#     Returns:
#         x: unpartitioned sequences with [B, H, W, C].
#     """
#     Hp, Wp = pad_hw
#     H, W = hw
#     B = windows.shape[0] // (Hp * Wp // window_size // window_size)
#     x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
#
#     if Hp > H or Wp > W:
#         x = x[:, :H, :W, :].contiguous()
#     return x
#
#
# def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
#     """
#     Get relative positional embeddings according to the relative positions of
#         query and key sizes.
#     Args:
#         q_size (int): size of query q.
#         k_size (int): size of key k.
#         rel_pos (Tensor): relative position embeddings (L, C).
#
#     Returns:
#         Extracted positional embeddings according to relative positions.
#     """
#     max_rel_dist = int(2 * max(q_size, k_size) - 1)
#     # Interpolate rel pos if needed.
#     if rel_pos.shape[0] != max_rel_dist:
#         # Interpolate rel pos.
#         rel_pos_resized = F.interpolate(
#             rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
#             size=max_rel_dist,
#             mode="linear",
#         )
#         rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
#     else:
#         rel_pos_resized = rel_pos
#
#     # Scale the coords with short length if shapes for q and k are different.
#     q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
#     k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
#     relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
#
#     return rel_pos_resized[relative_coords.long()]
#
#
# def add_decomposed_rel_pos(
#     attn: torch.Tensor,
#     q: torch.Tensor,
#     rel_pos_h: torch.Tensor,
#     rel_pos_w: torch.Tensor,
#     q_size: Tuple[int, int],
#     k_size: Tuple[int, int],
# ) -> torch.Tensor:
#     """
#     Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
#     https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
#     Args:
#         attn (Tensor): attention map.
#         q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
#         rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
#         rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
#         q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
#         k_size (Tuple): spatial sequence size of key k with (k_h, k_w).
#
#     Returns:
#         attn (Tensor): attention map with added relative positional embeddings.
#     """
#     q_h, q_w = q_size
#     k_h, k_w = k_size
#     Rh = get_rel_pos(q_h, k_h, rel_pos_h)
#     Rw = get_rel_pos(q_w, k_w, rel_pos_w)
#
#     B, _, dim = q.shape
#     r_q = q.reshape(B, q_h, q_w, dim)
#     rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
#     rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
#
#     attn = (
#         attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
#     ).view(B, q_h * q_w, k_h * k_w)
#
#     return attn
#
#
# class PatchEmbed(nn.Module):
#     """
#     Image to Patch Embedding.
#     """
#
#     def __init__(
#         self,
#         kernel_size: Tuple[int, int] = (16, 16),
#         stride: Tuple[int, int] = (16, 16),
#         padding: Tuple[int, int] = (0, 0),
#         in_chans: int = 3,
#         embed_dim: int = 768,
#     ) -> None:
#         """
#         Args:
#             kernel_size (Tuple): kernel size of the projection layer.
#             stride (Tuple): stride of the projection layer.
#             padding (Tuple): padding size of the projection layer.
#             in_chans (int): Number of input image channels.
#             embed_dim (int): Patch embedding dimension.
#         """
#         super().__init__()
#
#         self.proj = nn.Conv2d(
#             in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.proj(x)
#         # B C H W -> B H W C
#         x = x.permute(0, 2, 3, 1)
#         return x
#
#
# # class _OTFM(nn.Module):
# #     """Coupled attention fusion module"""
# #     def __init__(self, channels):
# #         super(_OTFM, self).__init__()
# #         self.conv_value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
# #         self.conv_query = nn.Conv2d(channels, channels, kernel_size=1)
# #         self.conv_key = nn.Conv2d(channels, channels, kernel_size=1)
# #
# #         self.softmax = nn.Softmax(dim=2)
# #         self.gamma = nn.Parameter(torch.zeros(1))
# #
# #     def forward(self, x, y):
# #         value = self.conv_value(y)
# #         value = value.view(value.size(0), value.size(1), -1)
# #
# #         query = self.conv_query(x)
# #         key = self.conv_key(y)
# #         query = query.view(query.size(0), query.size(1), -1)
# #         key = key.view(key.size(0), key.size(1), -1)
# #
# #         key_mean = key.mean(2).unsqueeze(2)
# #         query_mean = query.mean(2).unsqueeze(2)
# #         key -= key_mean
# #         query -= query_mean
# #
# #         sim_map = torch.bmm(query.transpose(1, 2), key)
# #         sim_map = self.softmax(sim_map)
# #         out_sim = torch.bmm(sim_map, value.transpose(1, 2))
# #         out_sim = out_sim.transpose(1, 2)
# #         out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
# #         out_sim = self.gamma * out_sim
# #
# #         return out_sim
# #
# #
# # class OTFM(nn.Module):
# #     """Coupled attention fusion module"""
# #     def __init__(self, channels_rgb, channels_dem, channels_fuse):
# #         super(OTFM, self).__init__()
# #         self.conv_rgb = nn.Conv2d(channels_rgb, channels_fuse, kernel_size=1)
# #         self.conv_dem = nn.Conv2d(channels_dem, channels_fuse, kernel_size=1)
# #
# #         self._OTFM1 = _OTFM(channels=channels_fuse)
# #         self._OTFM2 = _OTFM(channels=channels_fuse)
# #
# #         self.fuse_conv = nn.Conv2d(2 * channels_fuse, channels_fuse, kernel_size=1)
# #
# #     def forward(self, x, y):
# #         """x:rgb, y:dem"""
# #         x = self.conv_rgb(x)
# #         y = self.conv_dem(y)
# #         o2t = x + self._OTFM1(x, y).contiguous()
# #         t2o = y + self._OTFM2(y, x).contiguous()
# #         fuse = self.fuse_conv(torch.cat((o2t, t2o), dim=1))
# #
# #         return fuse
#
#
# def load_sam_weights_to_rgbcrim_encoder_final_superfixed(encoder, sam_checkpoint_path):
#     import torch
#
#     print(f"🔵 Loading SAM checkpoint from: {sam_checkpoint_path}")
#     checkpoint = torch.load(sam_checkpoint_path, map_location="cpu")
#
#     if "state_dict" in checkpoint:
#         sam_state = checkpoint["state_dict"]
#     else:
#         sam_state = checkpoint
#
#     sam_state = {k.replace("image_encoder.", ""): v for k, v in sam_state.items() if k.startswith("image_encoder.")}
#
#     model_state = encoder.state_dict()
#     new_state = {}
#     matched_keys = []
#
#     # Step 1: PatchEmbed 迁移
#     # PatchEmbed
#     # PatchEmbed: 自动适配不同 in_chans（1通道/4通道）
#     if "patch_embed.proj.weight" in sam_state and "patch_embed.proj.weight" in model_state:
#         pretrained_weight = sam_state["patch_embed.proj.weight"]  # [768, 3, 16, 16]
#         current_weight = model_state["patch_embed.proj.weight"]  # [768, in_chans, 16, 16]
#         in_chans = current_weight.shape[1]
#
#         if in_chans == 3:
#             new_state["patch_embed.proj.weight"] = pretrained_weight.clone()
#         elif in_chans == 1:
#             # DEM: 平均RGB权重或使用第一个通道
#             new_state["patch_embed.proj.weight"] = pretrained_weight.mean(dim=1, keepdim=True)
#         elif in_chans == 4:
#             # RGB+DEM：前三通道复制，DEM通道初始化为 0
#             new_weight = torch.zeros_like(current_weight)
#             new_weight[:, :3] = pretrained_weight
#             new_state["patch_embed.proj.weight"] = new_weight
#         else:
#             raise ValueError(f"Unsupported input channel count: {in_chans}")
#
#         matched_keys.append("patch_embed.proj.weight")
#         print("🧪 patch_embed.proj.weight shape:", encoder.patch_embed.proj.weight.shape)
#
#     # PatchEmbed.bias 直接复制（通常 shape 无通道依赖）
#     if "patch_embed.proj.bias" in sam_state and "patch_embed.proj.bias" in model_state:
#         new_state["patch_embed.proj.bias"] = sam_state["patch_embed.proj.bias"].clone()
#         matched_keys.append("patch_embed.proj.bias")
#
#     # PosEmbed
#     if "pos_embed" in sam_state and "pos_embed" in model_state:
#         new_state["pos_embed"] = sam_state["pos_embed"].clone()
#         matched_keys.append("pos_embed")
#     #     print("✅ PosEmbed: pos_embed")
#     # else:
#     #     print("❌ PosEmbed Missed: pos_embed")
#
#     # Step 3: Transformer Blocks映射
#     # print("\n================= Transformer Blocks 映射 =================")
#     mapping_list = [
#         (0, 0, "rgb_window1"), (1, 0, "rgb_window2"), (2, 0, "rgb_global"),
#         (3, 1, "rgb_window1"), (4, 1, "rgb_window2"), (5, 1, "rgb_global"),
#         (6, 2, "rgb_window1"), (7, 2, "rgb_window2"), (8, 2, "rgb_global"),
#         (9, 3, "rgb_window1"), (10, 3, "rgb_window2"), (11, 3, "rgb_global"),
#     ]
#     for sam_block_idx, fusion_block_idx, branch in mapping_list:
#         rgb_prefix = f"blocks.{fusion_block_idx}.{branch}"
#         for param_type in ["weight", "bias"]:
#             for norm in ["norm1", "norm2"]:
#                 sam_key = f"blocks.{sam_block_idx}.{norm}.{param_type}"
#                 model_key = f"{rgb_prefix}.{norm}.{param_type}"
#                 if sam_key in sam_state and model_key in model_state:
#                     new_state[model_key] = sam_state[sam_key].clone()
#                     matched_keys.append(model_key)
#                     # print(f"✅ Norm: {model_key}")
#         for part in ["qkv", "proj"]:
#             for param_type in ["weight", "bias"]:
#                 sam_key = f"blocks.{sam_block_idx}.attn.{part}.{param_type}"
#                 model_key = f"{rgb_prefix}.attn.{part}.{param_type}"
#                 if sam_key in sam_state and model_key in model_state:
#                     new_state[model_key] = sam_state[sam_key].clone()
#                     matched_keys.append(model_key)
#                     # print(f"✅ Attention: {model_key}")
#         for mlp_part in ["lin1", "lin2"]:
#             for param_type in ["weight", "bias"]:
#                 sam_key = f"blocks.{sam_block_idx}.mlp.{mlp_part}.{param_type}"
#                 model_key = f"{rgb_prefix}.mlp.{mlp_part}.{param_type}"
#                 if sam_key in sam_state and model_key in model_state:
#                     new_state[model_key] = sam_state[sam_key].clone()
#                     matched_keys.append(model_key)
#                     # print(f"✅ MLP: {model_key}")
#
#     # Step 4: RGB ➔ CRIM 拷贝
#     # print("\n================= RGB ➔ CRIM 拷贝 =================")
#     for fusion_idx in range(4):
#         for rgb_branch, crim_branch in [("rgb_window1", "crim_window1"), ("rgb_window2", "crim_window2"), ("rgb_global", "crim_global")]:
#             for sub_key in [
#                 "norm1.weight", "norm1.bias",
#                 "attn.qkv.weight", "attn.qkv.bias",
#                 "attn.proj.weight", "attn.proj.bias",
#                 "norm2.weight", "norm2.bias",
#                 "mlp.lin1.weight", "mlp.lin1.bias",
#                 "mlp.lin2.weight", "mlp.lin2.bias",
#             ]:
#                 rgb_key = f"blocks.{fusion_idx}.{rgb_branch}.{sub_key}"
#                 crim_key = f"blocks.{fusion_idx}.{crim_branch}.{sub_key}"
#                 if rgb_key in new_state and crim_key in model_state:
#                     new_state[crim_key] = new_state[rgb_key].clone()
#                     matched_keys.append(crim_key)
#                     # print(f"✅ Copy: {rgb_key} ➔ {crim_key}")
#
#     # Step 5: crim_global_mha
#     # print("\n================= crim_global_mha 映射 =================")
#     for fusion_block_idx in range(4):
#         sam_block_idx_for_global = fusion_block_idx * 3 + 2
#         prefix = f"blocks.{fusion_block_idx}.crim_global_mha"
#         for part in ["qkv", "proj"]:
#             for param_type in ["weight", "bias"]:
#                 sam_key = f"blocks.{sam_block_idx_for_global}.attn.{part}.{param_type}"
#                 model_key = f"{prefix}.{part}.{param_type}"
#                 if sam_key in sam_state and model_key in model_state:
#                     new_state[model_key] = sam_state[sam_key].clone()
#                     matched_keys.append(model_key)
#                     # print(f"✅ crim_global_mha: {model_key}")
#
#     # Step 6: FusionBlock norm2 + mlp
#     # print("\n================= FusionBlock Norm2 + MLP 映射 =================")
#     for fusion_block_idx in range(4):
#         sam_block_idx_for_global = fusion_block_idx * 3 + 2
#         fusion_prefix = f"blocks.{fusion_block_idx}"
#         for param_type in ["weight", "bias"]:
#             sam_key = f"blocks.{sam_block_idx_for_global}.norm2.{param_type}"
#             model_key = f"{fusion_prefix}.norm2.{param_type}"
#             if sam_key in sam_state and model_key in model_state:
#                 new_state[model_key] = sam_state[sam_key].clone()
#                 matched_keys.append(model_key)
#                 # print(f"✅ FusionBlock norm2: {model_key}")
#         for mlp_part in ["lin1", "lin2"]:
#             for param_type in ["weight", "bias"]:
#                 sam_key = f"blocks.{sam_block_idx_for_global}.mlp.{mlp_part}.{param_type}"
#                 model_key = f"{fusion_prefix}.mlp.{mlp_part}.{param_type}"
#                 if sam_key in sam_state and model_key in model_state:
#                     new_state[model_key] = sam_state[sam_key].clone()
#                     matched_keys.append(model_key)
#                     # print(f"✅ FusionBlock MLP: {model_key}")
#
#     # Step 7: Neck
#     # print("\n================= Neck 映射 =================")
#     for idx in range(4):
#         for param_type in ["weight", "bias"]:
#             sam_key = f"neck.{idx}.{param_type}"
#             model_key = f"neck.{idx}.{param_type}"
#             if sam_key in sam_state and model_key in model_state:
#                 new_state[model_key] = sam_state[sam_key].clone()
#                 matched_keys.append(model_key)
#                 # print(f"✅ Neck: {model_key}")
#
#     # Step 8: 加载权重
#     encoder.load_state_dict(new_state, strict=False)
#
#     # Step 9: 打印未匹配
#     print("\n================= 总结 =================")
#     total_keys = list(model_state.keys())
#     unmatched_keys = [k for k in total_keys if k not in matched_keys]
#
#     # 统计未迁移参数的数量和总参数量（单位：百万）
#     total_unmatched_params = 0
#     for k in unmatched_keys:
#         param = model_state[k]
#         total_unmatched_params += param.numel()
#         print(f"❌ {k} - shape: {list(param.shape)} - params: {param.numel() / 1e6:.6f} M")
#
#     total_params = sum(p.numel() for p in model_state.values())
#     matched_params = total_params - total_unmatched_params
#     print(f"\n✅ 成功迁移参数量: {matched_params / 1e6:.2f} M / {total_params / 1e6:.2f} M")
#     print(f"⚠️ 未迁移参数量: {total_unmatched_params / 1e6:.2f} M，共计 {len(unmatched_keys)} 个参数")
#     print("\n🚀 权重迁移完成，除Adapter和CrossAttention外，全部初始化完毕！\n")
#
#     return encoder
#
#
#
#
#
#
#
#
#
# def test_load_sam_to_rdvit(
#     sam_checkpoint_path=r"D:\pycharm\samtwo\sam2rad\weights\sam_vit_b_01ec64.pth",
#     img_size=1024,
#     patch_size=16,
#     in_chans=3,
#     embed_dim=768,
#     depth=4,
#     num_heads=8,
#     out_chans=256,
#     freeze=False
# ):
#     """
#     测试: 迁移SAM权重到RDViTEncoder并完成一次推理。
#     """
#
#     print("🔵 开始测试RDViTEncoder...")
#
#     # 1. 创建随机输入
#     B, H, W, C = 1, img_size, img_size, in_chans
#     rgb_input = torch.randn(B, C, H, W)
#     crim_input = torch.randn(B, C, H, W)
#
#     # 2. 初始化RDViTEncoder
#     encoder = RDViTEncoder(
#         img_size=img_size,
#         patch_size=patch_size,
#         in_chans=in_chans,
#         embed_dim=embed_dim,
#         depth=depth,
#         num_heads=num_heads,
#         mlp_ratio=4.0,
#         out_chans=out_chans,
#         qkv_bias=True,
#         norm_layer=torch.nn.LayerNorm,
#         act_layer=torch.nn.GELU,
#         use_abs_pos=True,
#         use_rel_pos=False,
#         rel_pos_zero_init=True,
#         window_size=7,
#         global_attn_indexes=(),  # 第3个block用global
#     )
#     for name in encoder.state_dict().keys():
#         print(name)
#     print(f"🛠️  RDViTEncoder实例化完成 (参数量: {sum(p.numel() for p in encoder.parameters())/1e6:.2f} M)")
#
#     # 3. 加载SAM权重
#     # 1. 打印sam里面实际有哪些key
#
#     encoder = load_sam_weights_to_rgbcrim_encoder_final_superfixed(encoder, sam_checkpoint_path)
#
#     # 4. Forward推理
#     encoder.eval()
#     with torch.no_grad():
#         final_out, high_res_features = encoder(rgb_input, crim_input)
#
#     print(f"✅ 推理成功！final_out.shape = {final_out.shape}")
#     for i, feat in enumerate(high_res_features):
#         print(f"✅ high_res_features[{i}].shape = {feat.shape}")
#
#     print("🏁 测试流程完成！")
#     return encoder
#
#
# sam_ckpt_path = r"D:\pycharm\samtwo\sam2rad\weights\sam_vit_b_01ec64.pth"  # SAM权重路径
# encoder = test_load_sam_to_rdvit(sam_ckpt_path, freeze=True)
#
from typing import Any, Optional, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type


# import torch
# import torch.nn as nn
#
# class PositionAdapter(nn.Module):
#     """
#     Refines position embeddings without changing spatial size or channels.
#     Input:  [B, H, W, C]  (e.g. [1, 64, 64, 768])
#     Output: [B, H, W, C]  (same shape)
#     """
#     def __init__(self, channels: int = 768):
#         super().__init__()
#         self.refine = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.GELU()
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Input is [B, H, W, C] → convert to [B, C, H, W]
#         x_bchw = x.permute(0, 3, 1, 2)
#
#         out = self.refine(x_bchw)   # 保持形状不变
#         out = out.permute(0, 2, 3, 1)  # 转回 BHWC
#
#         return out


# import torch
# import torch.nn as nn
#
# class DownscaleAdapter(nn.Module):
#     """
#     DownscaleAdapter with residual connection.
#     Input shape:  [B, H, W, 768]
#     Output shape: [B, H, W, 768] (unchanged)
#     """
#     def __init__(self, channels: int = 768):
#         super().__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
#         self.norm = nn.BatchNorm2d(channels)
#         self.act = nn.GELU()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # BHWC → BCHW
#         x_bchw = x.permute(0, 3, 1, 2)  # [B, C, H, W]
#         identity = x_bchw               # 保存残差
#
#         out = self.conv1(x_bchw)
#         out = self.conv2(out)
#         out = self.norm(out)
#         out = self.act(out)
#
#         out = out + identity            # 残差连接
#         out = out.permute(0, 2, 3, 1)   # BCHW → BHWC
#
#         print('DownscaleAdapter之后x的形状', out.shape)
#         return out


# import torch.nn as nn
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """
        Spatial Attention Module
        输入: [B, C, H, W]
        输出: [B, C, H, W] （带空间注意力权重）
        """
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = (kernel_size - 1) // 2

        # 用于计算空间注意力 (对通道求均值和最大值，然后卷积融合)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入 x: [B, C, H, W]

        # 通道维度上求平均池化和最大池化 -> 得到两个空间特征图
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # 拼接
        attn = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]

        # 卷积 + sigmoid 得到空间注意力权重
        attn = self.conv(attn)  # [B, 1, H, W]
        attn = self.sigmoid(attn)  # 权重范围 (0,1)

        # 将注意力应用到原特征
        return x * attn
class FeatureAdapter(nn.Module):
    """
    Multi-scale Domain Transfer Adapter.
    Supports input shapes: [B, N, C], [B, H, W, C], [B, C, H, W].
    Output matches input shape.
    """
    def __init__(self, in_channels, reduction=4, norm_layer=nn.BatchNorm2d, dropout=0.1):
        super().__init__()
        mid_channels = max(8, in_channels // reduction)

        # Down projection
        self.down = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout2d(dropout)  # After GELU1

        # Multi-scale convolutions
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(mid_channels, mid_channels, kernel_size=7, padding=3)

        self.norm = norm_layer(mid_channels)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout2d(dropout)  # After GELU2

        # Up projection
        self.up = nn.Conv2d(mid_channels, in_channels, kernel_size=1)

        # Spatial attention
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        orig_shape = x.shape
        input_type = None

        # Handle input
        if x.dim() == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            input_type = 'bnc'
        elif x.dim() == 4 and x.shape[1] != self.down.in_channels:
            x = x.permute(0, 3, 1, 2)
            input_type = 'bhwc'
        else:
            input_type = 'bchw'

        # Multi-scale processing
        out = self.down(x)
        out = self.gelu(out)
        out = self.dropout1(out)

        out1 = self.conv1(out)
        out3 = self.conv3(out)
        out5 = self.conv5(out)
        out7 = self.conv7(out)

        out = out1 + out3 + out5 + out7
        out = self.norm(out)
        out = self.gelu2(out)
        out = self.dropout2(out)
        out = self.up(out)
        out = self.sa(out)

        # Restore original shape
        if input_type == 'bnc':
            B, N, C = orig_shape
            out = out.reshape(B, C, N).permute(0, 2, 1)
        elif input_type == 'bhwc':
            B, H, W, C = orig_shape
            out = out.permute(0, 2, 3, 1)

        return out



class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.act(x)
        x = self.dropout(x)     # 🔽 dropout after activation
        x = self.lin2(x)
        x = self.dropout(x)     # 🔽 dropout after second linear
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


# 假设你已有的相关模块已经导入（例如 PatchEmbed, DownscaleAdapter, ParallelBlock, LayerNorm2d, etc.）

class RDViTEncoder(nn.Module):
    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 4,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: type = nn.LayerNorm,
            act_layer: type = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.img_size = img_size

        # Patch Embedding：输出形状 [B, H_patch, W_patch, embed_dim]
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Positional Embedding（以绝对位置为例）
        self.pos_embed: nn.Parameter = None
        if use_abs_pos:
            # 假设 patch 数为 (img_size // patch_size) 维度
            num_patches = img_size // patch_size
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dim))
            # 可选：初始化 pos_embed
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 构造多个并行 Transformer block
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = RGB_CRIM_FusionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                dropout=dropout,
            )
            self.blocks.append(block)

        # Neck 模块：将 token 特征转为最终输出图，输出形状 [B, out_chans, H_out, W_out]
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if y is None:
            x = self.patch_embed(x)
            y = torch.zeros_like(x)  # ⚠️ 创建假 y 分支，避免 None 报错
        else:
            x = self.pos_embed
            y = self.pos_embed

        # 3. 添加位置编码（下采样 pos_embed 一致，确保与 x/y 对齐）
        if self.pos_embed is not None:
            pos = self.pos_embed  # 调整为与 x/y 同样的空间尺寸
            x = x + pos
            y = y + pos

        # 4. 依次通过每个 block，并收集高分辨率特征
        high_res_features = []
        for blk in self.blocks:
            out1, out2 = blk(x, y)  # 假设 blk 接受 (x, y) 并输出形状仍然为 [B, H, W, embed_dim]
            # 更新 x 用于连续处理（也可以选择不更新 y）
            x = out1
            y = out2
            out = out1 + out2
            # 保存当前的高分辨率特征（转换到 [B, embed_dim, H, W]）
            high_res_features.append(out.permute(0, 3, 1, 2))


        # 5. 最终 Neck 处理：将最后一个 block 的输出转换成最终的图像特征
        final_out = self.neck(out.permute(0, 3, 1, 2))
        # print("最终输出的尺寸", final_out.shape)

        # 6. 返回二元组：最终特征图和最后两层（或根据需求选择）的高分辨率特征
        return final_out, high_res_features[:]


class NewTransformerBlock(nn.Module):
    """Transformer block supporting both Window Attention and Global Attention.
       Only Global Attention uses FeatureAdapter (no module registered if local window attention)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer,dropout=dropout)

        self.window_size = window_size

        if self.window_size == 0:
            self.feature_adapter = FeatureAdapter(in_channels=dim,dropout=dropout)
        else:
            self.feature_adapter = None  # ⚡ 注意是 None，不实例化！

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        # Window attention
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            x = self.attn(x)
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        else:
            # Global attention
            B, H, W, C = x.shape
            x = x.view(B, H * W, C)
            x = self.attn(x)
            x = x.view(B, H, W, C)

        x = shortcut + x

        shortcut2 = x
        x = self.norm2(x)
        x = self.mlp(x)

        # Only global attention branch adds feature adapter
        if self.feature_adapter is not None:
            x = shortcut2 + x + self.feature_adapter(shortcut2)
        else:
            x = shortcut2 + x
        # x = shortcut2 + x

        return x

# class CrossBranchAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=True):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = (dim // num_heads) ** -0.5
#
#         self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim)
#
#     def forward(self, q_input, kv_input):
#         """
#         Args:
#             q_input: (B, N, C) - 来自CRIM分支（Query）
#             kv_input: (B, N, C) - 来自RGB分支（Key和Value）
#         Returns:
#             输出更新后的 Query (B, N, C)
#         """
#         B, N, C = q_input.shape
#
#         # 做 Q, K, V 投影
#         q = self.q_proj(q_input).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, heads, N, head_dim)
#         kv = self.kv_proj(kv_input).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]
#
#         # Cross Attention
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         out = (attn @ v).transpose(1, 2).reshape(B, N, C)
#
#         out = self.proj(out)
#         return out


import torch
import torch.nn as nn

class CrossBranchAttentionWithSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, H=None, W=None, sa_kernel=7):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.H = H  # 用于 SpatialAttention reshape
        self.W = W

        # QKV 投影
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Spatial Attention
        self.spatial_attn = SpatialAttention(kernel_size=sa_kernel)

    def forward(self, q_input, kv_input):
        """
        q_input: (B, N, C)
        kv_input: (B, N, C)
        """
        B, N, C = q_input.shape

        # QKV
        q = self.q_proj(q_input).reshape(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        kv = self.kv_proj(kv_input).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        k, v = kv[0], kv[1]

        # Cross Attention
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B, N, C)
        out = self.proj(out)

        # ---- Spatial Attention ----
        if self.H is not None and self.W is not None:
            x_2d = out.transpose(1,2).reshape(B, C, self.H, self.W)  # (B,C,H,W)
            x_2d = self.spatial_attn(x_2d)
            out = x_2d.flatten(2).transpose(1,2)  # (B,N,C)

        return out



# cba融合的模块
class RGB_CRIM_FusionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int = 7,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        # RGB分支：局部+局部+全局
        self.rgb_window1 = NewTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
            dropout=dropout
        )
        self.rgb_window2 = NewTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
            dropout=dropout
        )
        self.rgb_global = NewTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=0,  # global attention
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
            dropout=dropout
        )

        # CRIM分支（修正）
        self.crim_window1 = NewTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
            dropout=dropout
        )
        self.crim_window2 = NewTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
            dropout=dropout
        )
        self.crim_global = NewTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=0,  # global attention
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
            dropout=dropout
        )


        self.cross_branch_attention = CrossBranchAttentionWithSA(dim=dim, num_heads=8, qkv_bias=True, H=None, W=None, sa_kernel=7)


        # 对齐SAM结构：FusionBlock内的 norm2 + mlp
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer,dropout=dropout)

    def forward(self, rgb, crim):
        """
        Args:
            rgb: (B, H, W, C)
            crim: (B, H, W, C)
        Returns:
            updated rgb and crim features
        """

        # RGB分支
        rgb = self.rgb_window1(rgb)
        rgb = self.rgb_window2(rgb)
        # CRIM分支
        crim = self.crim_window1(crim)
        crim = self.crim_window2(crim)

        # CrossBranch Attention
        B, H, W, C = rgb.shape
        q = self.rgb_global.norm1(rgb).view(B, H * W, C)
        kv = self.crim_global(crim).view(B, H * W, C)

        fusion_feature = self.cross_branch_attention(q, kv)
        rgb_1 = fusion_feature.view(B, H, W, C)  # ✅ 必须还原回 [B, H, W, C]

        # Global attention + FeatureAdapter（新的对称结构）
        rgb_2 = self.rgb_global(rgb)
        return rgb_2, rgb_1


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # 输入是 [B, H, W, C]
            B, H, W, _ = x.shape
            qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        elif x.dim() == 3:
            # 输入是 [B, N, C]，比如 global attention
            B, N, _ = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * self.num_heads, N, -1).unbind(0)
        else:
            raise ValueError(f"Unsupported input shape for Attention: {x.shape}")

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, -1, q.shape[-1]).transpose(1, 2).reshape(B, -1,
                                                                                        self.num_heads * q.shape[-1])

        x = self.proj(x)

        if x.dim() == 3:
            return x
        else:
            return x.view(B, H, W, -1)  # 如果是4D的情况，reshape回来

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x




def load_sam_weights_to_rgbcrim_encoder_final_superfixed(encoder, sam_checkpoint_path):
    import torch

    print(f"🔵 Loading SAM checkpoint from: {sam_checkpoint_path}")
    checkpoint = torch.load(sam_checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        sam_state = checkpoint["state_dict"]
    else:
        sam_state = checkpoint

    sam_state = {k.replace("image_encoder.", ""): v for k, v in sam_state.items() if k.startswith("image_encoder.")}

    model_state = encoder.state_dict()
    new_state = {}
    matched_keys = []

    # Step 1: PatchEmbed 迁移
    # PatchEmbed
    # PatchEmbed: 自动适配不同 in_chans（1通道/4通道）
    if "patch_embed.proj.weight" in sam_state and "patch_embed.proj.weight" in model_state:
        pretrained_weight = sam_state["patch_embed.proj.weight"]  # [768, 3, 16, 16]
        current_weight = model_state["patch_embed.proj.weight"]  # [768, in_chans, 16, 16]
        in_chans = current_weight.shape[1]

        if in_chans == 3:
            new_state["patch_embed.proj.weight"] = pretrained_weight.clone()
        elif in_chans == 1:
            # DEM: 平均RGB权重或使用第一个通道
            new_state["patch_embed.proj.weight"] = pretrained_weight.mean(dim=1, keepdim=True)
        elif in_chans == 4:
            # RGB+DEM：前三通道复制，DEM通道初始化为 0
            new_weight = torch.zeros_like(current_weight)
            new_weight[:, :3] = pretrained_weight
            new_state["patch_embed.proj.weight"] = new_weight
        else:
            raise ValueError(f"Unsupported input channel count: {in_chans}")

        matched_keys.append("patch_embed.proj.weight")
        print("🧪 patch_embed.proj.weight shape:", encoder.patch_embed.proj.weight.shape)

    # PatchEmbed.bias 直接复制（通常 shape 无通道依赖）
    if "patch_embed.proj.bias" in sam_state and "patch_embed.proj.bias" in model_state:
        new_state["patch_embed.proj.bias"] = sam_state["patch_embed.proj.bias"].clone()
        matched_keys.append("patch_embed.proj.bias")

    # PosEmbed
    if "pos_embed" in sam_state and "pos_embed" in model_state:
        new_state["pos_embed"] = sam_state["pos_embed"].clone()
        matched_keys.append("pos_embed")
    #     print("✅ PosEmbed: pos_embed")
    # else:
    #     print("❌ PosEmbed Missed: pos_embed")

    # Step 3: Transformer Blocks映射
    # print("\n================= Transformer Blocks 映射 =================")
    mapping_list = [
        (0, 0, "rgb_window1"), (1, 0, "rgb_window2"), (2, 0, "rgb_global"),
        (3, 1, "rgb_window1"), (4, 1, "rgb_window2"), (5, 1, "rgb_global"),
        (6, 2, "rgb_window1"), (7, 2, "rgb_window2"), (8, 2, "rgb_global"),
        (9, 3, "rgb_window1"), (10, 3, "rgb_window2"), (11, 3, "rgb_global"),
    ]
    for sam_block_idx, fusion_block_idx, branch in mapping_list:
        rgb_prefix = f"blocks.{fusion_block_idx}.{branch}"
        for param_type in ["weight", "bias"]:
            for norm in ["norm1", "norm2"]:
                sam_key = f"blocks.{sam_block_idx}.{norm}.{param_type}"
                model_key = f"{rgb_prefix}.{norm}.{param_type}"
                if sam_key in sam_state and model_key in model_state:
                    new_state[model_key] = sam_state[sam_key].clone()
                    matched_keys.append(model_key)
                    # print(f"✅ Norm: {model_key}")
        for part in ["qkv", "proj"]:
            for param_type in ["weight", "bias"]:
                sam_key = f"blocks.{sam_block_idx}.attn.{part}.{param_type}"
                model_key = f"{rgb_prefix}.attn.{part}.{param_type}"
                if sam_key in sam_state and model_key in model_state:
                    new_state[model_key] = sam_state[sam_key].clone()
                    matched_keys.append(model_key)
                    # print(f"✅ Attention: {model_key}")
        for mlp_part in ["lin1", "lin2"]:
            for param_type in ["weight", "bias"]:
                sam_key = f"blocks.{sam_block_idx}.mlp.{mlp_part}.{param_type}"
                model_key = f"{rgb_prefix}.mlp.{mlp_part}.{param_type}"
                if sam_key in sam_state and model_key in model_state:
                    new_state[model_key] = sam_state[sam_key].clone()
                    matched_keys.append(model_key)
                    # print(f"✅ MLP: {model_key}")

    # Step 4: RGB ➔ CRIM 拷贝
    # print("\n================= RGB ➔ CRIM 拷贝 =================")
    for fusion_idx in range(4):
        for rgb_branch, crim_branch in [("rgb_window1", "crim_window1"), ("rgb_window2", "crim_window2"), ("rgb_global", "crim_global")]:
            for sub_key in [
                "norm1.weight", "norm1.bias",
                "attn.qkv.weight", "attn.qkv.bias",
                "attn.proj.weight", "attn.proj.bias",
                "norm2.weight", "norm2.bias",
                "mlp.lin1.weight", "mlp.lin1.bias",
                "mlp.lin2.weight", "mlp.lin2.bias",
            ]:
                rgb_key = f"blocks.{fusion_idx}.{rgb_branch}.{sub_key}"
                crim_key = f"blocks.{fusion_idx}.{crim_branch}.{sub_key}"
                if rgb_key in new_state and crim_key in model_state:
                    new_state[crim_key] = new_state[rgb_key].clone()
                    matched_keys.append(crim_key)
                    # print(f"✅ Copy: {rgb_key} ➔ {crim_key}")

    # Step 5: crim_global_mha
    # print("\n================= crim_global_mha 映射 =================")
    for fusion_block_idx in range(4):
        sam_block_idx_for_global = fusion_block_idx * 3 + 2
        prefix = f"blocks.{fusion_block_idx}.crim_global_mha"
        for part in ["qkv", "proj"]:
            for param_type in ["weight", "bias"]:
                sam_key = f"blocks.{sam_block_idx_for_global}.attn.{part}.{param_type}"
                model_key = f"{prefix}.{part}.{param_type}"
                if sam_key in sam_state and model_key in model_state:
                    new_state[model_key] = sam_state[sam_key].clone()
                    matched_keys.append(model_key)
                    # print(f"✅ crim_global_mha: {model_key}")

    # Step 6: FusionBlock norm2 + mlp
    # print("\n================= FusionBlock Norm2 + MLP 映射 =================")
    for fusion_block_idx in range(4):
        sam_block_idx_for_global = fusion_block_idx * 3 + 2
        fusion_prefix = f"blocks.{fusion_block_idx}"
        for param_type in ["weight", "bias"]:
            sam_key = f"blocks.{sam_block_idx_for_global}.norm2.{param_type}"
            model_key = f"{fusion_prefix}.norm2.{param_type}"
            if sam_key in sam_state and model_key in model_state:
                new_state[model_key] = sam_state[sam_key].clone()
                matched_keys.append(model_key)
                # print(f"✅ FusionBlock norm2: {model_key}")
        for mlp_part in ["lin1", "lin2"]:
            for param_type in ["weight", "bias"]:
                sam_key = f"blocks.{sam_block_idx_for_global}.mlp.{mlp_part}.{param_type}"
                model_key = f"{fusion_prefix}.mlp.{mlp_part}.{param_type}"
                if sam_key in sam_state and model_key in model_state:
                    new_state[model_key] = sam_state[sam_key].clone()
                    matched_keys.append(model_key)
                    # print(f"✅ FusionBlock MLP: {model_key}")

    # Step 7: Neck
    # print("\n================= Neck 映射 =================")
    for idx in range(4):
        for param_type in ["weight", "bias"]:
            sam_key = f"neck.{idx}.{param_type}"
            model_key = f"neck.{idx}.{param_type}"
            if sam_key in sam_state and model_key in model_state:
                new_state[model_key] = sam_state[sam_key].clone()
                matched_keys.append(model_key)
                # print(f"✅ Neck: {model_key}")

    # Step 8: 加载权重
    encoder.load_state_dict(new_state, strict=False)

    # Step 9: 打印未匹配
    print("\n================= 总结 =================")
    total_keys = list(model_state.keys())
    unmatched_keys = [k for k in total_keys if k not in matched_keys]

    # 统计未迁移参数的数量和总参数量（单位：百万）
    total_unmatched_params = 0
    for k in unmatched_keys:
        param = model_state[k]
        total_unmatched_params += param.numel()
        print(f"❌ {k} - shape: {list(param.shape)} - params: {param.numel() / 1e6:.6f} M")

    total_params = sum(p.numel() for p in model_state.values())
    matched_params = total_params - total_unmatched_params
    print(f"\n✅ 成功迁移参数量: {matched_params / 1e6:.2f} M / {total_params / 1e6:.2f} M")
    print(f"⚠️ 未迁移参数量: {total_unmatched_params / 1e6:.2f} M，共计 {len(unmatched_keys)} 个参数")
    print("\n🚀 权重迁移完成，除Adapter和CrossAttention外，全部初始化完毕！\n")

    return encoder









def test_load_sam_to_rdvit(
    sam_checkpoint_path=r"D:\zhuhe\sam2rad\sam2rad\weights\sam_vit_b_01ec64.pth",
    img_size=1024,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    depth=4,
    num_heads=8,
    out_chans=256,
    freeze=False
):
    """
    测试: 迁移SAM权重到RDViTEncoder并完成一次推理。
    """

    print("🔵 开始测试RDViTEncoder...")

    # 1. 创建随机输入
    B, H, W, C = 2, img_size, img_size, in_chans
    rgb_input = torch.randn(B, C, H, W)
    crim_input = torch.randn(B, C, H, W)

    # 2. 初始化RDViTEncoder
    encoder = RDViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        out_chans=out_chans,
        qkv_bias=True,
        norm_layer=torch.nn.LayerNorm,
        act_layer=torch.nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=7,
        global_attn_indexes=(),  # 第3个block用global
    )
    for name in encoder.state_dict().keys():
        print(name)
    print(f"🛠️  RDViTEncoder实例化完成 (参数量: {sum(p.numel() for p in encoder.parameters())/1e6:.2f} M)")

    # 3. 加载SAM权重
    # 1. 打印sam里面实际有哪些key

    encoder = load_sam_weights_to_rgbcrim_encoder_final_superfixed(encoder, sam_checkpoint_path)

    # 4. Forward推理
    encoder.eval()
    with torch.no_grad():
        final_out, high_res_features = encoder(rgb_input,crim_input)

    print(f"✅ 推理成功！final_out.shape = {final_out.shape}")
    for i, feat in enumerate(high_res_features):
        print(f"✅ high_res_features[{i}].shape = {feat.shape}")

    print("🏁 测试流程完成！")
    return encoder


sam_ckpt_path = r"D:\zhuhe\sam2rad\sam2rad\weights\sam_vit_b_01ec64.pth"  # SAM权重路径
encoder = test_load_sam_to_rdvit(sam_ckpt_path, freeze=True)

