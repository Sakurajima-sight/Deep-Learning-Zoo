import math
import torch
from torch import nn

# layers
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # 学习的缩放因子
        self.beta = nn.Parameter(torch.zeros(d_model))  # 学习的平移因子
        self.eps = eps  # 防止除以0的小常数

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # 计算输入的均值
        var = x.var(-1, unbiased=False, keepdim=True)  # 计算输入的方差
        # '-1' 表示最后一个维度（通常是特征维度）
        
        out = (x - mean) / torch.sqrt(var + self.eps)  # 对输入进行标准化
        out = self.gamma * out + self.beta  # 进行缩放和平移
        return out


class ScaleDotProductAttention(nn.Module):
    """
    计算缩放点积注意力（Scaled Dot-Product Attention）

    Query：给定的目标句子（我们关注的内容，通常是解码器部分）
    Key：每一个句子，与Query的关系（通常是编码器部分）
    Value：与Key相同的每一个句子（也是编码器部分）
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)  # 使用softmax计算注意力分布

    def forward(self, q, k, v, mask=None, e=1e-12):
        # 输入是一个4维张量
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. 将Query与Key的转置进行点积计算，得到相似度
        k_t = k.transpose(2, 3)  # 转置Key的最后两个维度
        score = (q @ k_t) / math.sqrt(d_tensor)  # 缩放点积（除以sqrt(d_tensor)）

        # 2. 如果有mask，则应用mask（通常用于填充无效位置）
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)  # 对mask位置的score赋一个非常小的值（-10000）

        # 3. 使用softmax将score转换为[0, 1]之间的权重
        score = self.softmax(score)

        # 4. 将计算得到的权重与Value相乘，得到最终的输出
        v = score @ v  # 对Value加权求和

        return v, score  # 返回加权后的Value和注意力权重


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)  # 第一层线性变换，将输入维度映射到隐藏层维度
        self.linear2 = nn.Linear(hidden, d_model)  # 第二层线性变换，将隐藏层维度映射回输入维度
        self.relu = nn.ReLU()  # 使用ReLU作为激活函数
        self.dropout = nn.Dropout(p=drop_prob)  # Dropout层，用于防止过拟合

    def forward(self, x):
        x = self.linear1(x)  # 通过第一层线性变换
        x = self.relu(x)  # 使用ReLU激活函数
        x = self.dropout(x)  # 使用Dropout层
        x = self.linear2(x)  # 通过第二层线性变换
        return x  # 返回输出
    

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head  # 头数，表示将Q、K、V分成多少个子空间
        self.attention = ScaleDotProductAttention()  # 使用缩放点积注意力
        self.w_q = nn.Linear(d_model, d_model)  # 线性变换，将输入映射到Q的维度
        self.w_k = nn.Linear(d_model, d_model)  # 线性变换，将输入映射到K的维度
        self.w_v = nn.Linear(d_model, d_model)  # 线性变换，将输入映射到V的维度
        self.w_concat = nn.Linear(d_model, d_model)  # 用于将多头注意力的输出拼接并映射回原始维度

    def forward(self, q, k, v, mask=None):
        # 1. 使用权重矩阵对Q、K、V进行线性变换
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. 按照头的数量拆分tensor
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. 进行缩放点积计算相似度
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. 拼接多头的输出，并通过线性变换
        out = self.concat(out)
        out = self.w_concat(out)

        return out  # 返回经过多头注意力机制处理的输出

    def split(self, tensor):
        """
        按照头数拆分tensor

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()  # 获取输入tensor的尺寸

        d_tensor = d_model // self.n_head  # 每个头的维度
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2) # 将tensor重新排列，类似于分组卷积（按头数拆分）

        return tensor

    def concat(self, tensor):
        """
        与self.split函数的逆操作，合并多个头的输出

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor  # 合并后的维度

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)  # 拼接各个头的输出
        return tensor



# embedding
class PositionalEncoding(nn.Module):
    """
    计算正弦位置编码
    """

    def __init__(self, d_model, max_len, device):
        """
        正弦位置编码类的构造函数

        :param d_model: 模型的维度
        :param max_len: 允许的最大序列长度
        :param device: 计算所使用的设备（如cpu或cuda）
        """
        super(PositionalEncoding, self).__init__()

        # 创建一个与输入相同大小的矩阵（用于与输入相加）
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # 位置编码不需要梯度（不参与训练）

        pos = torch.arange(0, max_len, device=device)  # 生成位置序列 0~max_len
        pos = pos.float().unsqueeze(dim=1)  # 变成二维，表示单词的位置

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # _2i 表示d_model中的偶数位置索引，例如[0, 2, 4, ..., d_model]

        # 计算正弦和余弦位置编码，用于提供位置信息
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))  # 偶数维度使用sin
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))  # 奇数维度使用cos

    def forward(self, x):
        # self.encoding 的shape是 [max_len, d_model]，例如 [512, 512]

        batch_size, seq_len = x.size()
        # x的shape是 [batch_size, seq_len]，例如 [128, 30]

        return self.encoding[:seq_len, :]
        # 返回与输入长度相同的位置信息 [seq_len, d_model]
        # 这将在后续与token embedding相加


class TokenEmbedding(nn.Embedding):
    """
    Token嵌入（词向量嵌入）模块，继承自nn.Embedding
    通过可学习的权重矩阵将词id转换为稠密表示（即embedding）
    """

    def __init__(self, vocab_size, d_model):
        """
        Token嵌入类的构造函数

        :param vocab_size: 词表大小
        :param d_model: 模型的维度（词向量的维度）
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
        # padding_idx=1 表示当输入为padding的索引时，embedding为全0


class TransformerEmbedding(nn.Module):
    """
    Transformer输入嵌入模块 = Token嵌入 + 位置编码（正弦函数）
    位置编码可以为网络提供位置信息
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        Transformer输入嵌入类的构造函数

        :param vocab_size: 词表大小
        :param d_model: 模型的维度
        :param max_len: 最大序列长度
        :param drop_prob: dropout概率
        :param device: 计算设备
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)  # Token Embedding
        self.pos_emb = PositionalEncoding(d_model, max_len, device)  # 位置编码
        self.drop_out = nn.Dropout(p=drop_prob)  # Dropout层

    def forward(self, x):
        tok_emb = self.tok_emb(x)  # 获取词嵌入
        pos_emb = self.pos_emb(x)  # 获取位置编码

        # 词嵌入 + 位置编码 → Dropout → 作为Transformer的输入
        return self.drop_out(tok_emb + pos_emb)



# blocks
class EncoderLayer(nn.Module):
    """
    编码器层（Encoder Layer）
    """

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        # 自注意力机制
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)  # 层归一化
        self.dropout1 = nn.Dropout(p=drop_prob)  # Dropout层，用于防止过拟合

        # 前馈神经网络（Positionwise Feedforward）
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)  # 层归一化
        self.dropout2 = nn.Dropout(p=drop_prob)  # Dropout层

    def forward(self, x, src_mask):
        """
        编码器前向传播
        :param x: 输入的源序列
        :param src_mask: 源序列的掩码
        :return: 编码器的输出
        """
        # 1. 计算自注意力（Self Attention）
        _x = x  # 保存输入的原始值，用于残差连接
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. 残差连接和层归一化
        x = self.dropout1(x)  # Dropout
        x = self.norm1(x + _x)  # 残差连接并进行层归一化

        # 3. 前馈神经网络
        _x = x  # 保存输入的原始值，用于残差连接
        x = self.ffn(x)

        # 4. 残差连接和层归一化
        x = self.dropout2(x)  # Dropout
        x = self.norm2(x + _x)  # 残差连接并进行层归一化
        return x
    

class DecoderLayer(nn.Module):
    """
    解码器层（Decoder Layer）
    """

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        # 自注意力机制
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)  # 层归一化
        self.dropout1 = nn.Dropout(p=drop_prob)  # Dropout层，用于防止过拟合

        # 编码器-解码器注意力
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)  # 层归一化
        self.dropout2 = nn.Dropout(p=drop_prob)  # Dropout层

        # 前馈神经网络（Positionwise Feedforward）
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)  # 层归一化
        self.dropout3 = nn.Dropout(p=drop_prob)  # Dropout层

    def forward(self, dec, enc, trg_mask, src_mask):
        """
        解码器前向传播
        :param dec: 解码器输入
        :param enc: 编码器输出
        :param trg_mask: 目标序列的掩码
        :param src_mask: 源序列的掩码
        :return: 解码器的输出
        """
        # 1. 计算自注意力（Self Attention）
        _x = dec  # 保存输入的原始值，用于残差连接
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        # 2. 残差连接和层归一化
        x = self.dropout1(x)  # Dropout
        x = self.norm1(x + _x)  # 残差连接并进行层归一化

        if enc is not None:
            # 3. 计算编码器-解码器注意力（Encoder-Decoder Attention）
            _x = x  # 保存输入的原始值，用于残差连接
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

            # 4. 残差连接和层归一化
            x = self.dropout2(x)  # Dropout
            x = self.norm2(x + _x)  # 残差连接并进行层归一化

        # 5. 前馈神经网络
        _x = x  # 保存输入的原始值，用于残差连接
        x = self.ffn(x)

        # 6. 残差连接和层归一化
        x = self.dropout3(x)  # Dropout
        x = self.norm3(x + _x)  # 残差连接并进行层归一化
        return x
    


# model
class Encoder(nn.Module):
    """
    编码器（Encoder）类
    """

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        """
        编码器初始化函数

        :param enc_voc_size: 编码器词汇表大小
        :param max_len: 最大序列长度
        :param d_model: 模型的维度
        :param ffn_hidden: 前馈神经网络的隐藏层大小
        :param n_head: 多头注意力的头数
        :param n_layers: 编码器层数
        :param drop_prob: Dropout的概率
        :param device: 设备类型（cpu或cuda）
        """
        super().__init__()

        # 词嵌入 + 位置编码
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        # 构建多层编码器
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        """
        编码器前向传播函数

        :param x: 输入序列（源语言）
        :param src_mask: 源序列的掩码
        :return: 编码器输出
        """
        # 通过嵌入层转换输入
        x = self.emb(x)

        # 通过多层编码器层
        for layer in self.layers:
            x = layer(x, src_mask)

        return x
    

class Decoder(nn.Module):
    """
    解码器（Decoder）类
    """

    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        """
        解码器初始化函数

        :param dec_voc_size: 解码器词汇表大小
        :param max_len: 最大序列长度
        :param d_model: 模型的维度
        :param ffn_hidden: 前馈神经网络的隐藏层大小
        :param n_head: 多头注意力的头数
        :param n_layers: 解码器层数
        :param drop_prob: Dropout的概率
        :param device: 设备类型（cpu或cuda）
        """
        super().__init__()

        # 词嵌入 + 位置编码
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        # 构建多层解码器
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        # 输出层：将解码器输出映射到词汇表大小
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        解码器前向传播函数

        :param trg: 目标序列（翻译或生成的文本）
        :param enc_src: 编码器的输出
        :param trg_mask: 目标序列的掩码
        :param src_mask: 源序列的掩码
        :return: 解码器的输出，经过线性变换后为词汇表大小的预测
        """
        # 通过嵌入层转换目标输入
        trg = self.emb(trg)

        # 通过多层解码器层
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # 经过线性层映射到词汇表大小
        output = self.linear(trg)
        return output


class Transformer(nn.Module):
    """
    Transformer模型类
    """

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        """
        Transformer初始化函数

        :param src_pad_idx: 源序列的填充标志（padding index）
        :param trg_pad_idx: 目标序列的填充标志（padding index）
        :param trg_sos_idx: 目标序列的起始标志（start of sequence index）
        :param enc_voc_size: 编码器词汇表大小
        :param dec_voc_size: 解码器词汇表大小
        :param d_model: 模型的维度
        :param n_head: 多头注意力的头数
        :param max_len: 最大序列长度
        :param ffn_hidden: 前馈神经网络的隐藏层大小
        :param n_layers: 编码器和解码器的层数
        :param drop_prob: Dropout的概率
        :param device: 设备类型（cpu或cuda）
        """
        super().__init__()

        self.src_pad_idx = src_pad_idx  # 源序列填充索引
        self.trg_pad_idx = trg_pad_idx  # 目标序列填充索引
        self.trg_sos_idx = trg_sos_idx  # 目标序列起始索引
        self.device = device  # 设备类型

        # 初始化编码器
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        # 初始化解码器
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        """
        Transformer的前向传播

        :param src: 源序列（输入文本）
        :param trg: 目标序列（翻译后的文本）
        :return: Transformer模型的输出
        """
        # 为源序列生成掩码
        src_mask = self.make_src_mask(src)
        # 为目标序列生成掩码
        trg_mask = self.make_trg_mask(trg)
        # 编码器处理源序列，得到编码器输出
        enc_src = self.encoder(src, src_mask)
        # 解码器处理目标序列和编码器输出，得到最终输出
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        """
        创建源序列的掩码（防止填充部分影响模型）

        :param src: 源序列
        :return: 源序列的掩码
        """
        # 生成一个布尔值的掩码，源序列的填充部分为False
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        """
        创建目标序列掩码，防止看到未来的信息，并对padding进行mask
        """
        device = trg.device  # 保证掩码和输入数据在同一个设备

        # Padding Mask (batch_size, 1, 1, trg_len)
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)

        # Sub Mask (trg_len, trg_len) -- causal mask
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()  # ✅ 直接创建在正确设备上！

        # Combine both masks
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

