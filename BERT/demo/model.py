import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        self.gelu = nn.GELU()  # 使用GELU作为激活函数
        self.dropout = nn.Dropout(p=drop_prob)  # Dropout层，用于防止过拟合

    def forward(self, x):
        x = self.linear1(x)  # 通过第一层线性变换
        x = self.gelu(x)  # 使用GELU激活函数
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
        x = q
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

class PositionalEmbedding(nn.Module):
    """
    位置编码模块（Positional Encoding）
    通过正弦和余弦函数为每个位置生成唯一的编码，提供位置信息。
    """

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # 提前计算好位置编码矩阵，避免每次都重新计算
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False  # 不需要梯度更新（固定的）

        # 生成位置索引（0 到 max_len - 1）
        position = torch.arange(0, max_len).float().unsqueeze(1)

        # 计算每个维度对应的分母（按照论文公式，频率按 log 均匀分布）
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置

        # 增加batch维度（为了和输入x在batch维度对齐）
        pe = pe.unsqueeze(0)

        # 将位置编码注册为缓冲区，不作为参数更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 根据输入序列长度返回对应位置的编码
        return self.pe[:, :x.size(1)]
    

class SegmentEmbedding(nn.Embedding):
    """
    句子片段编码（Segment Embedding）
    用于区分句子A和句子B（例如：问答任务中问题和答案的区分），通常是 0, 1, 2 三种情况
    """
    def __init__(self, num_segments=3, embed_size=512):
        # 3表示最多支持3个片段（一般只用到两个，0和1）
        super().__init__(num_segments, embed_size, padding_idx=0)


class TokenEmbedding(nn.Embedding):
    """
    词嵌入（Token Embedding）
    将单词ID映射为稠密向量
    """
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class BERTEmbedding(nn.Module):
    """
    BERT嵌入层（BERT Embedding）
    
    由以下三部分组成：
      1. TokenEmbedding ：将词转换为向量
      2. PositionalEmbedding ：为每个单词加入位置信息（让模型区分单词顺序）
      3. SegmentEmbedding ：为句子片段加入片段信息（区分句子A和句子B）

    这三者的加和作为BERT的最终输入，之后送入Transformer。
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        初始化BERTEmbedding
        :param vocab_size: 词表大小
        :param embed_size: 每个单词的向量维度（embedding size）
        :param dropout: dropout概率，防止过拟合
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)     # 词嵌入
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)         # 位置嵌入
        self.segment = SegmentEmbedding(num_segments=2, embed_size=self.token.embedding_dim)          # 片段嵌入
        self.dropout = nn.Dropout(p=dropout)                                          # dropout
        self.embed_size = embed_size
        self.norm = LayerNorm(self.embed_size)

    def forward(self, sequence, segment_label):
        """
        前向传播
        :param sequence: 输入的单词ID序列
        :param segment_label: 输入的片段ID序列（例如全0或全1，标识属于哪一个句子）
        :return: 融合了三种信息后的embedding（带位置信息和片段信息）
        """
        # 将词嵌入、位置嵌入和片段嵌入相加
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        x = self.norm(x)
        # 最后加上dropout
        return self.dropout(x)  

class TransformerBlock(nn.Module):
    """
    编码器层（Encoder Layer）
    """

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(TransformerBlock, self).__init__()
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

class BERT(nn.Module):
    """
    BERT 模型：双向编码器表示（Bidirectional Encoder Representations from Transformers）
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        初始化BERT

        :param vocab_size: 词表大小
        :param hidden: 隐藏层维度（一般为768）
        :param n_layers: Transformer块的数量（即深度）
        :param attn_heads: 多头注意力的头数
        :param dropout: dropout比例
        """
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # 前馈层的隐藏层大小，通常是hidden的4倍
        self.feed_forward_hidden = hidden * 4

        # BERT的嵌入层（Token + Positional + Segment Embedding之和）
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # 多层Transformer块（堆叠n_layers层）
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, hidden * 4, attn_heads, dropout) for _ in range(n_layers)]
        )

    def forward(self, x, segment_info):
        # 创建注意力mask，pad位置为False，其他为True（防止pad影响注意力计算）
        # mask维度: [batch_size, 1, seq_len, seq_len]
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # 输入x经过嵌入层（得到词向量 + 位置向量 + segment向量的和）
        x = self.embedding(x, segment_info)

        # 依次通过多个Transformer块
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        # 最终输出
        return x


class NextSentencePrediction(nn.Module):
    """
    下一句预测分类器（2分类：is_next 或 is_not_next）
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT输出的隐藏层维度
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)  # 线性层，输出两个类别
        self.softmax = nn.LogSoftmax(dim=-1)  # LogSoftmax用于计算对数概率

    def forward(self, x):
        # 取BERT输出的第一个token（[CLS]）做分类
        return self.softmax(self.linear(x[:, 0]))
    

class MaskedLanguageModel(nn.Module):
    """
    掩码语言模型（Masked Language Model）

    任务：预测被mask掉的token，属于多分类问题，类别数等于词表大小
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: BERT输出的隐藏层维度
        :param vocab_size: 词表大小
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)  # 线性层，输出词表大小的logits
        self.softmax = nn.LogSoftmax(dim=-1)  # LogSoftmax 

    def forward(self, x):
        # 直接输出每个token的分类结果（词表概率分布）
        return self.softmax(self.linear(x))
    

class BERTLM(nn.Module):
    """
    BERT语言模型（BERTLM）

    由两部分组成：
    - Next Sentence Prediction（下一句预测）
    - Masked Language Model（掩码语言模型）
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: 已训练的BERT模型
        :param vocab_size: 词表大小（用于Masked LM预测）
        """
        super().__init__()
        self.bert = bert

        # 下一句预测分类器
        self.next_sentence = NextSentencePrediction(self.bert.hidden)

        # 掩码语言模型分类器
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        # 经过BERT主模型
        x = self.bert(x, segment_label)

        # 返回下一句预测和掩码语言模型的输出
        return self.next_sentence(x), self.mask_lm(x)
    

class SQuADWeightMapper(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super().__init__()
        self.bert = BERT(vocab_size, hidden, n_layers, attn_heads, dropout)
        self.qa_outputs = nn.Linear(hidden, 2)
    
    def forward(self, x, segment_label):
        # 经过BERT主模型
        x = self.bert(x, segment_label)

        logits = self.qa_outputs(x)  # (batch_size, seq_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
