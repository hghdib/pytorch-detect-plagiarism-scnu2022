import os
import random
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 加载BERT模型和分词器
model_path = "bert-base-uncased"  # 替换为适合的模型路径（如CodeBERT）
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 读取代码文件
def load_code_from_folder(folder_path, num_samples=3):
    code_files = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".java"):  # 只加载 .java 文件
                file_path = os.path.join(root, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    code_files.append(f.read())
    # 随机选择指定数量的代码
    return random.sample(code_files, min(len(code_files), num_samples))

# 获取代码嵌入向量
def get_code_embedding(code, tokenizer, model, device, max_length=512):
    inputs = tokenizer(
        code, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=max_length
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # 使用 [CLS] 向量
    return cls_embedding.squeeze().cpu().numpy()

# 文件路径
case_folder = "/content/drive/MyDrive/IR-Plag-Dataset/case-01"
original_path = os.path.join(case_folder, "original")
non_plagiarized_path = os.path.join(case_folder, "non-plagiarized")
plagiarized_path = os.path.join(case_folder, "plagiarized")

# 随机选择代码
original_codes = load_code_from_folder(original_path, num_samples=3)
non_plagiarized_codes = load_code_from_folder(non_plagiarized_path, num_samples=3)
plagiarized_codes = load_code_from_folder(plagiarized_path, num_samples=3)

# 合并所有代码
all_codes = original_codes + non_plagiarized_codes + plagiarized_codes

# 生成嵌入向量
embeddings = [get_code_embedding(code, tokenizer, model, device) for code in all_codes]

# 计算相似性矩阵
similarity_matrix = cosine_similarity(embeddings)

# 打印相似性矩阵
categories = ["Original"] * 3 + ["Non-Plagiarized"] * 3 + ["Plagiarized"] * 3
print("Similarity Matrix:")
print(similarity_matrix)

# 打印详细结果
print("\nDetailed Similarities:")
for i, category_i in enumerate(categories):
    for j, category_j in enumerate(categories):
        if i < j:
            print(f"Similarity between {category_i} Code-{i+1} and {category_j} Code-{j+1}: {similarity_matrix[i][j]:.4f}")
