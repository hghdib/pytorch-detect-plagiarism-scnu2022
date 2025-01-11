import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 递归加载文件夹中的所有代码文件
def load_code_files_recursive(folder_path):
    code_files = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            if file_path.endswith(".java"):  # 如果是 .java 文件，加载它
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
                    code_files.append((file_path, code))
    return code_files

# 递归加载所有 case-* 文件夹中的代码文件
def load_all_code_files(base_path):
    dataset = []
    for case_folder in os.listdir(base_path):  # 遍历所有 case-* 文件夹
        case_path = os.path.join(base_path, case_folder)
        if os.path.isdir(case_path):
            # 加载原始代码、非抄袭代码和抄袭代码
            original_code = load_code_files_recursive(os.path.join(case_path, "original"))
            non_plagiarized_codes = load_code_files_recursive(os.path.join(case_path, "non-plagiarized"))
            plagiarized_codes = load_code_files_recursive(os.path.join(case_path, "plagiarized"))

            # 处理数据并加入到 dataset
            for _, code in non_plagiarized_codes:
                dataset.append({"code1": original_code[0][1], "code2": code, "label": 0})  # 0: 不相似
            for _, code in plagiarized_codes:
                dataset.append({"code1": original_code[0][1], "code2": code, "label": 1})  # 1: 相似

    return dataset

# 加载所有数据
base_path = "/content/drive/MyDrive/IR-Plag-Dataset"
dataset = load_all_code_files(base_path)

print("Dataset Size:", len(dataset))  # 输出数据集大小

# 加载 T5 分词器
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# 自定义 PyTorch 数据集
class CodeSimilarityDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        inputs = self.tokenizer(
            "code1: " + ex["code1"] + " code2: " + ex["code2"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": inputs["input_ids"].squeeze(0),  # 去掉批量维度
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
        }
        return item

# 创建训练数据集
train_dataset = CodeSimilarityDataset(dataset, tokenizer)

# 自定义 T5 分类模型
class T5Classifier(nn.Module):
    def __init__(self, base_model_name, num_labels):
        super().__init__()
        self.encoder = T5ForConditionalGeneration.from_pretrained(base_model_name)
        self.classifier = nn.Linear(self.encoder.config.d_model, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        encoder_outputs = self.encoder.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = encoder_outputs.last_hidden_state[:, 0, :]  # [CLS] 向量
        logits = self.classifier(cls_token)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5Classifier("t5-small", num_labels=2)
model.to(device)

# 定义性能评估函数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_safetensors=False,
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
)

# 定义 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # 添加评估函数
)

# 开始训练
trainer.train()

# 输出最佳模型性能
metrics = trainer.evaluate()
print("Evaluation Metrics:", metrics)

