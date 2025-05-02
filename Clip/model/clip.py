import hashlib
import os
import urllib
import warnings
from packaging import version
from typing import Union, List

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from model import build_model
from simple_tokenizer import SimpleTokenizer as _Tokenizer

# 尝试导入插值模式，兼容 torchvision 的新旧版本
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC  # torchvision >= 0.9 的插值模式写法
except ImportError:
    BICUBIC = Image.BICUBIC  # torchvision < 0.9 的写法

# 检查 PyTorch 版本，建议使用 1.7.1 或更高版本
if version.parse(torch.__version__) < version.parse("1.7.1"):
    warnings.warn("建议使用 PyTorch 1.7.1 或更高版本")

# 导出接口函数
__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

# 模型名称与对应的下载链接（包含 SHA256 校验码）
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    """
    从给定 URL 下载模型文件到指定目录 root，并进行 SHA256 校验
    """
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]  # 从 URL 中提取 SHA256 校验码
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} 存在但不是普通文件")

    # 如果文件存在且 SHA256 校验通过，直接返回路径
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} 已存在，但 SHA256 校验不匹配，将重新下载")

    # 下载文件，并显示进度条
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    # 再次进行 SHA256 校验，确保文件完整
    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("模型已下载，但 SHA256 校验失败")

    return download_target


def _convert_image_to_rgb(image):
    """
    将输入图像转换为 RGB 模式（确保图像格式统一）
    """
    return image.convert("RGB")


def _transform(n_px):
    """
    定义图像预处理流程：
    - 调整大小
    - 中心裁剪
    - 转换为 RGB
    - 转为张量
    - 标准化（使用 CLIP 的图像均值和标准差）
    """
    return Compose([
        Resize(n_px, interpolation=BICUBIC),  # 缩放到目标尺寸
        CenterCrop(n_px),                    # 中心裁剪
        _convert_image_to_rgb,               # 转换为 RGB 模式
        ToTensor(),                          # 转为 PyTorch 张量
        Normalize((0.48145466, 0.4578275, 0.40821073),  # 使用 CLIP 模型预定义的均值
                  (0.26862954, 0.26130258, 0.27577711)) # 和标准差进行归一化
    ])


def available_models() -> List[str]:
    """
    返回当前支持的 CLIP 模型名称列表
    """
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """
    加载一个 CLIP 模型

    参数
    ----------
    name : str
        模型名称，可通过 `clip.available_models()` 查询，或提供包含 state_dict 的模型文件路径

    device : Union[str, torch.device]
        加载模型的设备（默认为 CUDA，如可用）

    jit : bool
        是否加载优化过的 JIT 模型。False（默认）时加载可修改的非 JIT 模型

    download_root: str
        下载模型文件的路径；默认使用 "~/.cache/clip"

    返回值
    -------
    model : torch.nn.Module
        加载的 CLIP 模型

    preprocess : Callable[[PIL.Image], torch.Tensor]
        图像预处理函数（将 PIL 图像转换为模型可接受的张量）
    """

    # 判断 name 是不是在官方提供的模型名称列表中
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name  # 如果是本地路径，直接使用
    else:
        raise RuntimeError(f"未找到模型 {name}；可用模型列表 = {available_models()}")

    # 尝试打开模型文件并加载
    with open(model_path, 'rb') as opened_file:
        try:
            # 尝试加载为 JIT 格式模型
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # 如果不是 JIT 模型，尝试以普通 state_dict 方式加载
            if jit:
                warnings.warn(f"文件 {model_path} 不是 JIT 格式，将以 state_dict 加载")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    # 如果不是 JIT 模型，构建非 JIT 模型结构
    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()  # 如果是 CPU，转为 float32
        return model, _transform(model.visual.input_resolution)

    # JIT 模型需要“打补丁”以适配当前设备
    # 下面构建一个 device_holder，表示一个张量在目标设备上的构建方式
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def _node_get(node: torch._C.Node, key: str):
        """
        获取 TorchScript 中节点的属性值（可能为不同类型）

        参考：https://github.com/pytorch/pytorch/pull/82628
        """
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    # 替换模型中所有“常量设备”字段为当前 device
    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)

    # 对整个模型及其子函数 encode_image、encode_text 应用 device 修补
    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # 如果使用的是 CPU，还需要替换 dtype 为 float32
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype 通常是 aten::to 的第二或第三个参数
                        if _node_get(inputs[i].node(), "value") == 5:  # 5 代表 float32
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()  # 模型转 float32，适配 CPU

    return model, _transform(model.input_resolution.item())  # 返回模型和预处理函数


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    返回输入字符串（或字符串列表）对应的分词结果（token 序列）

    参数
    ----------
    texts : Union[str, List[str]]
        输入的一段文本，或由多段文本组成的列表

    context_length : int
        上下文长度。CLIP 模型使用的默认长度为 77

    truncate : bool
        是否在文本过长时截断（截断后最后一个 token 会被设为 EOT）

    返回
    -------
    返回一个二维的 tensor，形状为 [文本数量, context_length]，包含分词后的 token 序列。
    如果使用的是旧版本的 PyTorch（<1.8.0），返回 LongTensor（因为旧版本要求 index 必须是 long 类型）。
    """

    # 如果是单个字符串，转换为包含一个元素的列表
    if isinstance(texts, str):
        texts = [texts]

    # 获取特殊 token 的 ID
    sot_token = _tokenizer.encoder["<|startoftext|>"]  # 开始标记
    eot_token = _tokenizer.encoder["<|endoftext|>"]    # 结束标记

    # 对每个文本进行编码，添加起始和结束标记
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    # 根据 PyTorch 版本选择使用 LongTensor 或 IntTensor
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    # 将每段文本的 token 写入结果张量
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                # 截断到 context_length 长度，最后一个 token 替换为 eot
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                # 若不允许截断且超长，抛出异常
                raise RuntimeError(f"输入文本过长：'{texts[i]}' 超出了最大上下文长度 {context_length}")
        # 将 token 序列填入对应行
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result