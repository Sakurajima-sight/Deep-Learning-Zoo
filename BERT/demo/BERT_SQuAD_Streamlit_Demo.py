import streamlit as st
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from model import BERT, SQuADWeightMapper

# -------------------------------
# 加载并转换BERT权重（就是你之前那套代码）
@st.cache_resource
def load_model():
    # 加载预训练BERT
    pretrained_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    pretrained_state_dict = pretrained_model.state_dict()

    # 自定义模型
    my_bert = SQuADWeightMapper(vocab_size=30522, hidden=1024, n_layers=24, attn_heads=16)
    my_bert_state_dict = my_bert.state_dict()

    # 加载Embedding
    my_bert_state_dict["bert.embedding.token.weight"].copy_(pretrained_state_dict["bert.embeddings.word_embeddings.weight"])
    my_bert_state_dict["bert.embedding.position.pe"][0].copy_(pretrained_state_dict["bert.embeddings.position_embeddings.weight"])
    my_bert_state_dict["bert.embedding.segment.weight"][:2].copy_(pretrained_state_dict["bert.embeddings.token_type_embeddings.weight"])
    my_bert_state_dict["bert.embedding.norm.gamma"].copy_(pretrained_state_dict["bert.embeddings.LayerNorm.weight"])
    my_bert_state_dict["bert.embedding.norm.beta"].copy_(pretrained_state_dict["bert.embeddings.LayerNorm.bias"])

    # 加载Transformer
    for layer_idx in range(24):
        prefix_hf = f"bert.encoder.layer.{layer_idx}."
        prefix_my = f"bert.transformer_blocks.{layer_idx}."

        my_bert_state_dict[prefix_my + "attention.w_q.weight"].copy_(pretrained_state_dict[prefix_hf + "attention.self.query.weight"])
        my_bert_state_dict[prefix_my + "attention.w_q.bias"].copy_(pretrained_state_dict[prefix_hf + "attention.self.query.bias"])
        my_bert_state_dict[prefix_my + "attention.w_k.weight"].copy_(pretrained_state_dict[prefix_hf + "attention.self.key.weight"])
        my_bert_state_dict[prefix_my + "attention.w_k.bias"].copy_(pretrained_state_dict[prefix_hf + "attention.self.key.bias"])
        my_bert_state_dict[prefix_my + "attention.w_v.weight"].copy_(pretrained_state_dict[prefix_hf + "attention.self.value.weight"])
        my_bert_state_dict[prefix_my + "attention.w_v.bias"].copy_(pretrained_state_dict[prefix_hf + "attention.self.value.bias"])
        my_bert_state_dict[prefix_my + "attention.w_concat.weight"].copy_(pretrained_state_dict[prefix_hf + "attention.output.dense.weight"])
        my_bert_state_dict[prefix_my + "attention.w_concat.bias"].copy_(pretrained_state_dict[prefix_hf + "attention.output.dense.bias"])
        my_bert_state_dict[prefix_my + "norm1.gamma"].copy_(pretrained_state_dict[prefix_hf + "attention.output.LayerNorm.weight"])
        my_bert_state_dict[prefix_my + "norm1.beta"].copy_(pretrained_state_dict[prefix_hf + "attention.output.LayerNorm.bias"])
        my_bert_state_dict[prefix_my + "ffn.linear1.weight"].copy_(pretrained_state_dict[prefix_hf + "intermediate.dense.weight"])
        my_bert_state_dict[prefix_my + "ffn.linear1.bias"].copy_(pretrained_state_dict[prefix_hf + "intermediate.dense.bias"])
        my_bert_state_dict[prefix_my + "ffn.linear2.weight"].copy_(pretrained_state_dict[prefix_hf + "output.dense.weight"])
        my_bert_state_dict[prefix_my + "ffn.linear2.bias"].copy_(pretrained_state_dict[prefix_hf + "output.dense.bias"])
        my_bert_state_dict[prefix_my + "norm2.gamma"].copy_(pretrained_state_dict[prefix_hf + "output.LayerNorm.weight"])
        my_bert_state_dict[prefix_my + "norm2.beta"].copy_(pretrained_state_dict[prefix_hf + "output.LayerNorm.bias"])

    # 加载QA Head
    my_bert_state_dict["qa_outputs.weight"].copy_(pretrained_state_dict["qa_outputs.weight"])
    my_bert_state_dict["qa_outputs.bias"].copy_(pretrained_state_dict["qa_outputs.bias"])

    my_bert.eval()
    return my_bert

# -------------------------------
# 加载Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# -------------------------------
# Streamlit界面
st.title("📚 基于自制BERT的SQuAD问答系统")

question = st.text_input("请输入你的问题", "Where do I live?")
# 默认文本
default_text = "My name is John and I live in New York."

# 创建一个文本框，自动根据行数来设置显示的高度
context = st.text_area("请输入上下文（Context）", default_text, height=300)

if st.button("获取答案"):
    if not question.strip() or not context.strip():
        st.warning("请输入问题和上下文")
    else:
        model = load_model()
        inputs = tokenizer(question, context, return_tensors="pt")
        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]

        with torch.no_grad():
            start_logits, end_logits = model(input_ids, token_type_ids)
            start = torch.argmax(start_logits)
            end = torch.argmax(end_logits) + 1
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start:end]))

        st.success(f"🎉 答案是：{answer}")
