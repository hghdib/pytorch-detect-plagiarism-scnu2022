import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import ast

# 设置 AST 特征维度
ast_dim = 2  # 假设 AST 特征为 2 维：函数定义数量和类定义数量

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
    for case_folder in os.listdir(base_path):
        case_path = os.path.join(base_path, case_folder)
        if os.path.isdir(case_path):
            # 加载原始代码、非抄袭代码和抄袭代码
            original_code = load_code_files_recursive(os.path.join(case_path, "original"))
            non_plagiarized_codes = load_code_files_recursive(os.path.join(case_path, "non-plagiarized"))
            plagiarized_codes = load_code_files_recursive(os.path.join(case_path, "plagiarized"))

            # 处理数据并加入到 dataset
            for filename, code in non_plagiarized_codes:
                dataset.append({"code1": original_code[0][1], "code2": code, "label": 0})  # 0: 不相似
            for filename, code in plagiarized_codes:
                dataset.append({"code1": original_code[0][1], "code2": code, "label": 1})  # 1: 相似

    return dataset

# AST 特征提取函数
def extract_ast_features(code):
    try:
        tree = ast.parse(code)
        # 示例：提取函数定义的数量、类定义的数量等
        function_defs = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        class_defs = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        return function_defs, class_defs
    except:
        return 0, 0  # 如果无法解析为 AST，返回默认值

# 提前提取 AST 特征
def preprocess_ast_features(dataset, ast_dim):
    for item in dataset:
        code1_ast_features = extract_ast_features(item["code1"])
        code2_ast_features = extract_ast_features(item["code2"])
        item["ast_features_code1"] = torch.tensor(code1_ast_features, dtype=torch.float)
        item["ast_features_code2"] = torch.tensor(code2_ast_features, dtype=torch.float)
    return dataset


# 加载数据集并预处理
base_path = "/content/drive/MyDrive/IR-Plag-Dataset"
dataset = load_all_code_files(base_path)
dataset = preprocess_ast_features(dataset, ast_dim)

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

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
            "ast_features_code1": ex["ast_features_code1"],
            "ast_features_code2": ex["ast_features_code2"],
        }

# 创建训练数据集
train_dataset = CodeSimilarityDataset(dataset, tokenizer)

# 自定义 T5 分类模型
ast_dim = 2  # 假设每段代码提取 2 个维度的 AST 特征

# 自定义 T5 分类模型
class T5Classifier(nn.Module):
    def __init__(self, base_model_name, num_labels, ast_dim=0):
        super().__init__()
        self.encoder = T5ForConditionalGeneration.from_pretrained(base_model_name)
        self.ast_dim = ast_dim
        self.classifier = nn.Linear(self.encoder.config.d_model + 2 * self.ast_dim, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None, ast_features_code1=None, ast_features_code2=None):
        encoder_outputs = self.encoder.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = torch.mean(encoder_outputs.last_hidden_state, dim=1)

        # 合并编码器输出与 AST 特征
        if ast_features_code1 is not None and ast_features_code2 is not None:
            combined_features = torch.cat([pooled_output, ast_features_code1, ast_features_code2], dim=1)
        else:
            combined_features = pooled_output

        logits = self.classifier(combined_features)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


# 自定义 Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        自定义计算损失函数。
        """
        # 从输入中提取标签和 AST 特征
        labels = inputs.pop("labels")
        ast_features_code1 = inputs.pop("ast_features_code1")
        ast_features_code2 = inputs.pop("ast_features_code2")

        # 计算模型输出
        outputs = model(
            **inputs,
            labels=labels,
            ast_features_code1=ast_features_code1,
            ast_features_code2=ast_features_code2,
        )

        # 提取损失
        loss = outputs["loss"]

        # 根据需要返回输出
        return (loss, outputs) if return_outputs else loss


# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5Classifier("t5-small", num_labels=2, ast_dim=ast_dim)
model.to(device)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_safetensors=False,
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
)

# 定义 Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()

# 测试模型
def predict_similarity(code1, code2):
    inputs = tokenizer(
        "code1: " + code1 + " code2: " + code2,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    ast_features_code1 = torch.tensor(extract_ast_features(code1), dtype=torch.float).unsqueeze(0).to(device)
    ast_features_code2 = torch.tensor(extract_ast_features(code2), dtype=torch.float).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = trainer.model(
            **inputs,
            ast_features_code1=ast_features_code1,
            ast_features_code2=ast_features_code2
        )
        logits = outputs["logits"]
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()

    return "Similar" if prediction == 1 else "Not Similar"


# 替换为实际的代码内容
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

print(predict_similarity(test_code1, test_code2))
