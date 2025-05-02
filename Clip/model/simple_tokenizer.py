import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


@lru_cache()
def default_bpe():
    # 返回默认的 BPE 词表路径（压缩的 .txt.gz 文件）
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    返回一个 utf-8 字节到 unicode 字符的映射表。
    可逆的 BPE 编码是基于 unicode 字符串进行的。
    这意味着如果你想避免 <UNK>，就需要在词表中包含尽量多的 unicode 字符。
    对于大规模数据集（例如 100 亿 token），通常需要大约 5000 个字符来覆盖。
    如果我们不使用映射表，词表中会被占用掉很多符号，甚至包含空格和控制符，会导致 BPE 出错。
    """
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))  # 返回 byte 到字符的映射


def get_pairs(word):
    """
    返回单词中所有相邻的符号对（bigram）。
    单词是由多个子词（symbol）组成的元组。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    # 处理常见的编码错误并反转 HTML 转义字符
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    # 将多个空白符替换成一个空格，并去掉前后空格
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # 加载并解析 BPE 合并规则（压缩的词表文件）
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]  # 去掉前1行header，并只取有效的合并规则
        merges = [tuple(merge.split()) for merge in merges]

        # 构建初始词表（单字节字符 + </w> 结束符版本 + 合并词 + 特殊符号）
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]  # 添加词尾标记
        for merge in merges:
            vocab.append(''.join(merge))  # 添加合并后的词
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])  # 添加起始与结束特殊标记

        self.encoder = dict(zip(vocab, range(len(vocab))))  # 构建词到 ID 的映射
        self.decoder = {v: k for k, v in self.encoder.items()}  # 构建 ID 到词的映射
        self.bpe_ranks = dict(zip(merges, range(len(merges))))  # BPE 合并规则的优先级表

        # 缓存已计算过的 token BPE 结果，提高性能
        self.cache = {
            '<|startoftext|>': '<|startoftext|>',
            '<|endoftext|>': '<|endoftext|>'
        }

        # 正则表达式用于分词，包括：特殊标记、英文缩写、字母、数字、其他符号
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE
        )

    def bpe(self, token):
        # 如果该 token 已在缓存中，直接返回
        if token in self.cache:
            return self.cache[token]

        # 将 token 转换为字符序列，最后一个字符加上 </w> 表示结尾
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)  # 获取初始的 bigram 对

        if not pairs:
            return token + '</w>'

        # 按照 BPE 规则不断合并优先级最高的 bigram
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)  # 合并 bigram
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word  # 存入缓存
        return word

    def encode(self, text):
        # 文本预处理（修复+反转HTML+空格清理+小写化）
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):  # 使用正则分词
            # 编码 utf-8 字节流为 unicode 字符
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # 将 BPE 分词后的每个子词映射到词表 ID
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        # 将 token ID 转换为字符串
        text = ''.join([self.decoder[token] for token in tokens])
        # 将 unicode 字符转回原始字节，并去掉 </w> 结束标志
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
