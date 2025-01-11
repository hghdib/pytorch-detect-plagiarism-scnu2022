import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

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
            for filename, code in non_plagiarized_codes:
                dataset.append({"code1": original_code[0][1], "code2": code, "label": 0})  # 0: 不相似
            for filename, code in plagiarized_codes:
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
# 检查可用的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5Classifier("t5-small", num_labels=2)
model.to(device)  # 将模型移动到设备



# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_safetensors=False,  # 禁用 safetensors
    learning_rate=1e-3,
    per_device_train_batch_size=32,
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
)

# 开始训练
trainer.train()

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
        String filePath = "sample.txt"; // 替换为实际文件路径
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
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Map;
import java.util.Scanner;
import java.util.TreeMap;
import java.util.Comparator;
import java.util.List;
import java.util.ArrayList;

public class WordFrequencyAnalyzer {
    public static void main(String[] args) {
        String filePath = "sample.txt"; // 替换为实际文件路径
        TreeMap<String, Integer> wordFrequency = new TreeMap<>();

        try (Scanner scanner = new Scanner(new File(filePath))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                // 分割单词并清理标点符号
                String[] words = line.toLowerCase().replaceAll("[^a-zA-Z0-9 ]", "").split("\\s+");
                for (String word : words) {
                    if (!word.isEmpty()) {
                        wordFrequency.put(word, wordFrequency.getOrDefault(word, 0) + 1);
                    }
                }
            }
        } catch (FileNotFoundException e) {
            System.out.println("File not found: " + e.getMessage());
            return;
        }

        // 按频率排序
        List<Map.Entry<String, Integer>> sortedEntries = new ArrayList<>(wordFrequency.entrySet());
        sortedEntries.sort(new Comparator<Map.Entry<String, Integer>>() {
            @Override
            public int compare(Map.Entry<String, Integer> e1, Map.Entry<String, Integer> e2) {
                return e2.getValue().compareTo(e1.getValue()); // 降序排序
            }
        });

        // 打印排序结果
        System.out.println("Word Frequencies (sorted):");
        for (Map.Entry<String, Integer> entry : sortedEntries) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }
}

"""

# 调用预测函数测试相似性
print(predict_similarity(test_code1, test_code2))

