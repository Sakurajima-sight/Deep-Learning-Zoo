import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import ipdb
class Attention(nn.Module):
    """
    计算 '缩放点积注意力' (Scaled Dot Product Attention)
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        # 计算注意力分数 (query 与 key 的点积，然后除以 sqrt(d_k) 进行缩放)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        # 如果有mask，屏蔽掉不该关注的位置（如padding）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 将mask为0的位置填充一个极小的值，softmax后接近于0

        # 计算注意力权重 (对分数做softmax)
        p_attn = F.softmax(scores, dim=-1)

        # 如果提供了dropout，则对注意力权重进行dropout
        if dropout is not None:
            p_attn = dropout(p_attn)
        ipdb.set_trace()
        # 返回加权后的value（注意力结果）和注意力权重
        return torch.matmul(p_attn, value), p_attn
    

class MultiHeadedAttention(nn.Module):
    """
    多头注意力机制
    输入: 模型维度 (d_model) 和头数 (h)，将注意力机制拆分为多个头并行计算
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0  # 确保模型维度可以被头数整除

        # 每个头的维度（假设d_k == d_v）
        self.d_k = d_model // h
        self.h = h

        # 线性变换层，分别用于 query、key、value
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        
        # 最终的输出线性变换
        self.output_linear = nn.Linear(d_model, d_model)
        
        # 注意力计算模块（使用上面定义的Attention类）
        self.attention = Attention()

        # Dropout用于防止过拟合
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) 对 query、key 和 value 进行线性投影，并拆分成多头（batch_size, h, seq_len, d_k）
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) 计算注意力（每个头独立计算）
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        ipdb.set_trace()
        # 3) 将多头的结果拼接起来 (合并维度)，再通过输出线性层
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)  # 返回最终的输出


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
    

class GELU(nn.Module):
    """
    激活函数：GELU（高斯误差线性单元）
    
    来自论文的第 3.4 小节，BERT 选择了 GELU 而不是 ReLU 作为激活函数。
    GELU 比 ReLU 更平滑，在自然语言任务中表现更好。
    """

    def forward(self, x):
        # 计算GELU公式，返回GELU激活后的结果
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    

class LayerNorm(nn.Module):
    """
    层归一化（Layer Normalization）
    
    作用：对输入特征在最后一个维度上做归一化（保证均值为0，标准差为1），
    并引入可学习的缩放和偏移参数，提升训练稳定性。
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 可学习参数，a_2是缩放因子，b_2是偏移量
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps  # 避免除零的小常数

    def forward(self, x):
        # 计算均值和标准差（在最后一个维度上）
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # 标准化 + 缩放 + 偏移
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

class SublayerConnection(nn.Module):
    """
    子层连接（Sublayer Connection）
    
    结构: 残差连接（Residual） + 层归一化（LayerNorm）
    
    注意：与ResNet不同，这里是 "先归一化再残差连接"，以保持代码简单。
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)           # 层归一化
        self.dropout = nn.Dropout(dropout)    # dropout 防止过拟合

    def forward(self, x, sublayer):
        """
        前向传播
        x -> 归一化 -> 传入子层（如注意力或前馈）-> dropout -> 残差加和
        """
        return x + self.dropout(sublayer(self.norm(x)))
    

class PositionwiseFeedForward(nn.Module):
    """
    前馈全连接层（Position-wise Feed Forward Network, FFN）

    这个模块在Transformer中对每个位置的表示单独进行非线性变换：
    公式：FFN(x) = max(0, x * W1 + b1) * W2 + b2 （这里激活函数是GELU而不是ReLU）
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)   # 第一层全连接，将维度扩展到d_ff
        self.w_2 = nn.Linear(d_ff, d_model)   # 第二层全连接，将维度变回d_model
        self.dropout = nn.Dropout(dropout)    # dropout层
        self.activation = GELU()              # 使用GELU激活函数

    def forward(self, x):
        # 依次通过w_1 -> GELU激活 -> dropout -> w_2
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    

class TransformerBlock(nn.Module):
    """
    双向编码器（Transformer块）
    
    组成部分：
    - 多头自注意力机制（Multi-Head Attention）
    - 前馈全连接层（Feed Forward）
    - 子层连接（残差 + LayerNorm）
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        初始化 TransformerBlock

        :param hidden: Transformer隐藏层维度
        :param attn_heads: 多头注意力机制中的头数
        :param feed_forward_hidden: 前馈全连接网络的隐藏层维度，一般是hidden的4倍
        :param dropout: dropout比例
        """
        super().__init__()
        # 多头注意力机制
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)

        # 前馈全连接网络
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)

        # 两个子层连接（残差 + LayerNorm）
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)

        # 最后加一个dropout
        self.dropout = nn.Dropout(p=dropout)

    # 注意：本实现中的 Transformer Encoder 块集成 Masked 机制
    def forward(self, x, mask):
        # 第一步：自注意力 + 残差 + LayerNorm
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))

        # 第二步：前馈网络 + 残差 + LayerNorm
        x = self.output_sublayer(x, self.feed_forward)

        # 最后做一次dropout
        return self.dropout(x)
    

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
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)]
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
        import ipdb
        ipdb.set_trace()
        logits = self.qa_outputs(x)  # (batch_size, seq_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
