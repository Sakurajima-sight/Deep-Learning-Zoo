from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# 定义 Bottleneck 模块，基本残差单元，使用了 ResNet 的 Bottleneck 架构
class Bottleneck(nn.Module):
    expansion = 4  # 输出通道数会被扩展 4 倍

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # 第一层 1x1 卷积用于压缩通道数
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        # 第二层 3x3 卷积保持通道数
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        # 如果 stride > 1，使用 avgpool 替代卷积下采样（抗锯齿）
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        # 第三层 1x1 卷积扩展通道数
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        # 如果维度不一致或需要下采样，定义跳连分支
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),  # 使用平均池化代替 stride 卷积
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)  # 只有当 stride > 1 时才真正执行下采样
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu3(out)
        return out


# AttentionPool2d：使用注意力代替平均池化的模块（视觉Transformer常用）
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # 添加位置编码
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        # 三个线性变换用于 Q/K/V
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # 输出投影
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # 将输入展平为 (H*W, N, C)，再 permute 成 transformer 所需格式
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # 增加 [CLS] token
        x = x + self.positional_embedding[:, None, :].to(x.dtype)

        # 多头注意力机制，仅输出 [CLS] 的注意力结果
        x, _ = F.multi_head_attention_forward(
            # 仅使用 [CLS] token 作为唯一 query，从所有位置 (包括自身) 聚合全局信息，模拟全局池化效果
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)  # 输出 [CLS] token 的嵌入


# ModifiedResNet：一个修改版的 ResNet 模型，用于更强的视觉表征
class ModifiedResNet(nn.Module):
    """
    Modified ResNet 架构，包含以下不同之处：
    - 使用3层卷积替代传统 ResNet 的1个 stem 卷积（更细致特征提取）
    - 所有下采样都使用 avgpool + conv 实现（抗锯齿设计）
    - 最终使用 AttentionPool2d 替代全局平均池化（更强表达能力）
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # 三层stem卷积
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(2)  # 下采样，抗锯齿

        # 残差层构建
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # 输出特征维度
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem部分
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)  # 匹配输入类型（float16/float32）
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)  # 用注意力池化取代平均池化

        return x


class LayerNorm(nn.LayerNorm):
    """继承自 PyTorch 的 LayerNorm，增加对 fp16（半精度）输入的支持"""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype  # 记录原始数据类型
        ret = super().forward(x.type(torch.float32))  # 先将输入转换为 float32 进行归一化
        return ret.type(orig_type)  # 再转换回原始数据类型


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        # 使用 QuickGELU 激活函数，比标准 GELU 更快
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        # 多头自注意力模块
        self.ln_1 = LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)

        # LayerNorm + MLP
        self.ln_2 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),  # 全连接层，升维
            ("gelu", QuickGELU()),  # QuickGELU 激活函数
            ("c_proj", nn.Linear(d_model * 4, d_model))  # 全连接层，降维回原大小
        ]))

        # 注意力掩码（可选）
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # 如果有注意力掩码，则将其转换为与输入相同的 dtype 和设备
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # 执行自注意力计算（只取输出，不需要权重）
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # 残差连接 + 注意力
        x = x + self.attention(self.ln_1(x))
        # 残差连接 + MLP
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        # 构建多个 ResidualAttentionBlock 组成的 Transformer 层
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)  # 逐层传递输入


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        # 图像分块并进行线性投影，相当于 patch embedding
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # 初始化类标记向量和位置嵌入
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))  # 类别嵌入（分类用）
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))  # 位置编码

        self.ln_pre = LayerNorm(width)  # Transformer 输入前的 LayerNorm

        # Transformer 编码器
        self.transformer = Transformer(width, layers, heads)

        # 输出处理层：LayerNorm + 线性映射
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))  # 最后的输出投影矩阵

    def forward(self, x: torch.Tensor):
        # 将图像切分为 patch 并做线性投影
        x = self.conv1(x)  # 输出形状: [batch, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # 展平空间维度: [batch, width, grid**2]
        x = x.permute(0, 2, 1)  # 交换维度: [batch, grid**2, width]

        # 添加类别嵌入（class token）
        class_token = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([class_token, x], dim=1)  # 拼接类别标记: [batch, grid**2 + 1, width]

        # 加上位置编码
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)  # 归一化

        # 输入 transformer，注意 transformer 期望的输入是 [sequence_length, batch, embedding_dim]
        x = x.permute(1, 0, 2)  # [batch, seq_len, dim] -> [seq_len, batch, dim]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [seq_len, batch, dim] -> [batch, seq_len, dim]

        # 取第一个 token（类别标记）作为输出，并做最后处理
        x = self.ln_post(x[:, 0, :])  # 取出类别标记并归一化

        if self.proj is not None:
            x = x @ self.proj  # 投影到输出维度

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # 图像相关参数
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # 文本相关参数
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length  # 文本最大长度

        # 根据传入参数选择视觉模型结构（ResNet 或 ViT）
        if isinstance(vision_layers, (tuple, list)):
            # 如果是 ResNet，计算注意力头数量
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            # 如果是 Vision Transformer
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        # 初始化文本 transformer 编码器
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()  # 构建自回归注意力掩码
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)  # 词嵌入矩阵
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))  # 位置嵌入
        self.ln_final = LayerNorm(transformer_width)  # transformer 输出的最终 LayerNorm

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))  # 文本特征映射到共同空间
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # 特征对比的 logit 缩放参数

        self.initialize_parameters()  # 初始化参数

    def initialize_parameters(self):
        # 初始化词嵌入和位置嵌入
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        # 如果是 ResNet 并且使用了注意力池化，初始化注意力权重
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            # 将 ResNet 中残差块的第 3 层 BN 层的 gamma 参数初始化为 0
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        # 初始化 transformer 层的注意力和前馈网络权重
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # 初始化文本投影矩阵
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # 构建一个因果注意力掩码，只允许关注当前及之前的 token
        # PyTorch 使用加性 attention mask，这里填充为 -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # 上三角填 -inf，保持下三角为 0
        return mask

    @property
    def dtype(self):
        # 返回模型中卷积层使用的张量数据类型（通常为 float32 或 float16）
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        # 对图像进行编码，输出图像特征
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        # 文本输入：[batch_size, context_length]
        x = self.token_embedding(text).type(self.dtype)  # 获取 token 嵌入向量

        x = x + self.positional_embedding.type(self.dtype)  # 加上位置嵌入
        x = x.permute(1, 0, 2)  # NLD -> LND，适配 transformer 输入
        x = self.transformer(x)  # transformer 编码
        x = x.permute(1, 0, 2)  # LND -> NLD，恢复原顺序
        x = self.ln_final(x).type(self.dtype)  # 最后再归一化

        # 获取每个样本中 End-of-Text（EOT）标记对应的特征作为整体文本表示
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        # 编码图像和文本
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # 对图像和文本特征进行归一化
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # 计算图文之间的余弦相似度 logits（乘以 logit 缩放因子）
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # 返回图文相似度矩阵（图像对文本 / 文本对图像）
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """将模型中可转换的参数转换为 fp16 半精度"""

    def _convert_weights_to_fp16(l):
        # 转换卷积层和线性层的参数为半精度
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        # 转换多头注意力层的权重
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        # 转换自定义投影层参数
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    # 应用到整个模型
    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    # 判断视觉模块是 ViT（Vision Transformer）还是 ResNet
    vit = "visual.proj" in state_dict

    if vit:
        # 对于 ViT 结构，读取模型结构参数
        vision_width = state_dict["visual.conv1.weight"].shape[0]  # ViT 中 token embedding 的维度
        # ViT 的 transformer 层数，统计 attn 层出现的次数
        vision_layers = len([
            k for k in state_dict.keys()
            if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        ])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]  # patch 大小
        # 计算网格大小（位置信息），位置嵌入数量 - 1，然后开平方
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size  # 还原图像输入分辨率
    else:
        # 对于 ResNet 结构，解析层数结构（例如每个 block 中的残差单元数）
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")))
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)  # 每层包含的 block 数量，例如 (3, 4, 6, 3)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]  # 第一层卷积输出通道数
        # 计算输出特征图大小（从位置嵌入数量反推）
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None  # ResNet 不使用 patch
        # 验证位置嵌入数量是否等于输出特征点数量 + 1（加了 [CLS] token）
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32  # 每个 patch 是 32x32，大致还原原图分辨率

    # 提取文本编码器相关结构参数
    embed_dim = state_dict["text_projection"].shape[1]  # 图文共同的嵌入维度
    context_length = state_dict["positional_embedding"].shape[0]  # 最大文本长度
    vocab_size = state_dict["token_embedding.weight"].shape[0]  # 词表大小
    transformer_width = state_dict["ln_final.weight"].shape[0]  # transformer 的隐藏层维度
    transformer_heads = transformer_width // 64  # 注意力头数，每个 head 默认 64 维
    # 统计 transformer 的层数，通过 resblock 的编号推断
    transformer_layers = len(set(
        k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")
    ))

    # 构建 CLIP 模型实例
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    # 删除部分额外的辅助信息（不是模型权重的）
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # 将模型参数转换为 FP16（半精度）以提升推理性能
    convert_weights(model)
    # 加载权重
    model.load_state_dict(state_dict)
    return model.eval()  # 返回 eval 模式的模型
