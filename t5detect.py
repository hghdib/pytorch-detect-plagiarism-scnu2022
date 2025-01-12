import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
import numpy as np

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
base_path = "/content/drive/MyDrive/Colab Notebooks/IR-Plag-Dataset"
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
    def __init__(self, base_model_name, num_labels, dropout_rate=0.3):
        super().__init__()
        self.encoder = T5ForConditionalGeneration.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(dropout_rate)  # 添加 Dropout 层
        self.classifier = nn.Linear(self.encoder.config.d_model, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        encoder_outputs = self.encoder.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = encoder_outputs.last_hidden_state[:, 0, :]  # [CLS] 向量
        cls_token = self.dropout(cls_token)  # 应用 Dropout
        logits = self.classifier(cls_token)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# 定义 K 折交叉验证
def k_fold_cross_validation(dataset, model, tokenizer, k=5, training_args=None):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    metrics_list = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n********** Fold {fold + 1}/{k} **********")

        # 根据索引划分数据集
        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]

        # 构造 PyTorch 数据集
        train_dataset = CodeSimilarityDataset(train_data, tokenizer)
        val_dataset = CodeSimilarityDataset(val_data, tokenizer)

        # 初始化模型
        model = T5Classifier("t5-small", num_labels=2).to(device)

        # 定义 Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # 训练模型
        trainer.train()

        # 评估模型
        metrics = trainer.evaluate()
        metrics_list.append(metrics)

        print(f"Metrics for Fold {fold + 1}: {metrics}")

    # 计算平均指标
    avg_metrics = {
        "accuracy": np.mean([m["eval_accuracy"] for m in metrics_list]),
        "precision": np.mean([m["eval_precision"] for m in metrics_list]),
        "recall": np.mean([m["eval_recall"] for m in metrics_list]),
        "f1": np.mean([m["eval_f1"] for m in metrics_list]),
    }

    print("\n********** Average Metrics Across All Folds **********")
    print(avg_metrics)
    return avg_metrics

# 定义训练参数，添加 weight_decay 实现 L2 正则化
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_safetensors=False,
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,  # 每折的训练轮数
    weight_decay=0.01,  # L2 正则化
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
)

# 执行 5 折交叉验证
avg_metrics = k_fold_cross_validation(dataset, model, tokenizer, k=5, training_args=training_args)

# 测试模型
def predict_similarity(code1, code2):
    # 确保模型已被移动到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.model.to(device)  # 确保 Trainer 内的模型也在设备上

    inputs = tokenizer(
        "code1: " + code1 + " code2: " + code2,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = trainer.model(**inputs)
        logits = outputs["logits"]
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
    
    print(f"Logits: {logits}")
    print(f"Probabilities: {probabilities}")
    return "Similar" if prediction == 1 else "Not Similar"


# 测试代码
test_code1 = """
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.LinkedHashMap;

public class WordFrequencyCounter {
    public static void main(String[] args) {
        String filePath = "sample.txt"; // 文件路径
        HashMap<String, Integer> wordCountMap = new HashMap<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;

            while ((line = reader.readLine()) != null) {
                // 按空格分割行中的单词
                String[] words = line.toLowerCase().replaceAll("[^a-zA-Z0-9 ]", "").split("\\s+");
                for (String word : words) {
                    if (!word.isEmpty()) {
                        wordCountMap.put(word, wordCountMap.getOrDefault(word, 0) + 1);
                    }
                }
            }
        } catch (IOException e) {
            System.out.println("Error reading the file: " + e.getMessage());
            return;
        }

        // 按频率降序排序
        Map<String, Integer> sortedMap = wordCountMap.entrySet()
            .stream()
            .sorted((e1, e2) -> e2.getValue().compareTo(e1.getValue()))
            .collect(Collectors.toMap(
                Map.Entry::getKey,
                Map.Entry::getValue,
                (e1, e2) -> e1,
                LinkedHashMap::new
            ));

        // 打印结果
        System.out.println("Word Frequencies:");
        for (Map.Entry<String, Integer> entry : sortedMap.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }
}
"""

test_code2 = """
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Map;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.Comparator;

public class FrequencyAnalyzer {
    public static void main(String[] args) {
        String pathToFile = "sample.txt"; // 指定文件路径
        Map<String, Integer> frequencyMap = new HashMap<>();

        try {
            BufferedReader br = new BufferedReader(new FileReader(pathToFile));
            String currentLine;

            while ((currentLine = br.readLine()) != null) {
                // 将行转换为小写，移除标点符号，并分割为单词数组
                String[] tokens = currentLine
                    .replaceAll("[^a-zA-Z0-9 ]", "")
                    .toLowerCase()
                    .split("\\s+");

                for (String token : tokens) {
                    if (token.length() > 0) {
                        frequencyMap.put(token, frequencyMap.getOrDefault(token, 0) + 1);
                    }
                }
            }
            br.close();
        } catch (IOException ioEx) {
            System.err.println("File could not be read: " + ioEx.getMessage());
            return;
        }

        // 对结果按频率进行排序
        TreeMap<String, Integer> sortedFrequencyMap = new TreeMap<>(
            Comparator.comparingInt(frequencyMap::get).reversed()
        );
        sortedFrequencyMap.putAll(frequencyMap);

        // 打印排序后的单词频率
        System.out.println("Word Frequency Analysis:");
        sortedFrequencyMap.forEach((word, count) -> System.out.println(word + ": " + count));
    }
}
"""

# 调用预测函数测试是否存在抄袭
print(predict_similarity(test_code1, test_code2))
