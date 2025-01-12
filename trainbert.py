import os
import random
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# 加载BERT模型和分词器
model_path = "bert-base-uncased"  # 替换为合适的模型路径
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 加载代码文件
def load_random_code(folder_path):
    code_files = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".java"):  # 只加载 .java 文件
                file_path = os.path.join(root, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    code_files.append(f.read())
    if code_files:
        return random.choice(code_files)
    return None

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
original_code = load_random_code(original_path)
plagiarized_code = load_random_code(plagiarized_path)

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


if original_code and plagiarized_code:
    # 生成嵌入向量
    original_embedding = get_code_embedding(original_code, tokenizer, model, device)
    plagiarized_embedding = get_code_embedding(plagiarized_code, tokenizer, model, device)

    # 计算相似度
    similarity = cosine_similarity([original_embedding], [plagiarized_embedding])[0][0]

    # 打印结果
    print("随机提取测试:")
    print(f"Semantic Similarity: {similarity:.4f}")
else:
    print("Could not load enough code files. Please check the dataset structure.")


if test_code1 and test_code2:
    # 生成嵌入向量
    test_code1_embedding = get_code_embedding(test_code1, tokenizer, model, device)
    test_code2_embedding = get_code_embedding(test_code2, tokenizer, model, device)

    # 计算相似度
    similarity = cosine_similarity([test_code1_embedding], [test_code2_embedding])[0][0]

    # 打印结果
    print("定向代码测试:")
    print("Original Code:")
    print(test_code1[:500] + "...\n")  # 打印部分代码
    print("Plagiarized Code:")
    print(test_code2[:500] + "...\n")  # 打印部分代码
    print(f"Semantic Similarity: {similarity:.4f}")
else:
    print("Could not load enough code files. Please check the dataset structure.")


