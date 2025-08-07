# class_1 主要内容
## 一、Pipelines 概述
- **定位**：基于 Transformers 库封装的预训练模型推理接口
- **特点**：
  - 抽象底层复杂代码
  - 提供 12 类任务的统一 API
  - 支持多模态任务（音频/图像/文本）
  - 降低 AI 推理使用门槛
 
## 二、任务分类与 API
### 1. 音频处理
| 任务类型               | 功能描述                  | API 调用方式                     |
|------------------------|--------------------------|---------------------------------|
| 音频分类               | 分配音频标签             | `pipeline("audio-classification")` |
| 自动语音识别           | 语音转文本               | `pipeline("automatic-speech-recognition")` |
 
### 2. 计算机视觉
| 任务类型               | 功能描述                  | API 调用方式                     |
|------------------------|--------------------------|---------------------------------|
| 图像分类               | 分配图像标签             | `pipeline("image-classification")` |
| 目标检测               | 检测物体及边界框         | `pipeline("object-detection")` |
| 图像分割               | 像素级分类               | `pipeline("image-segmentation")` |
 
### 3. 自然语言处理
| 任务类型               | 功能描述                  | API 调用方式                     |
|------------------------|--------------------------|---------------------------------|
| 文本分类               | 情感/内容分类            | `pipeline("sentiment-analysis")` |
| 命名实体识别           | 识别文本中的实体         | `pipeline("ner")` |
| 问答系统               | 提取式问答               | `pipeline("question-answering")` |
| 文本摘要               | 生成浓缩文本             | `pipeline("summarization")` |
| 机器翻译               | 多语言互译               | `pipeline("translation")` |
 
### 4. 多模态任务
| 任务类型               | 功能描述                  | API 调用方式                     |
|------------------------|--------------------------|---------------------------------|
| 文档问答               | 基于文档的问答           | `pipeline("document-question-answering")` |
| 视觉问答               | 基于图像的问答           | `pipeline("vqa")` |
 
## 三、Pipeline API 使用示例
### 1. 基础使用模式
```python
from transformers import pipeline
 
# 文本分类示例
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
# 输出: [{'label': 'POSITIVE', 'score': 0.9998}]
```

# class_2 主要内容
# 1.Hugging Face Transformers 微调训练入门

## 一、核心流程概览
1. **数据准备**：YelpReviewFull 数据集加载与探索
2. **文本预处理**：BERT Tokenizer 分词与填充
3. **模型配置**：BERT 分类模型加载与超参数设置
4. **训练监控**：准确率评估指标与训练过程跟踪
5. **模型保存**：训练权重与训练状态的持久化存储
 
## 二、数据集详解
### 1. 数据集特性
- **任务类型**：情感分类（1-5星评分）
- **数据规模**：
  - 训练集：650,000 条
  - 测试集：50,000 条
- **数据结构**：
  ```python
  {
      'label': 0-4 (对应1-5星),
      'text': 转义后的英文评论
  }
  ```

# 2.Hugging Face Transformers 微调问答模型
 
## 一、核心流程概览
1. **数据准备**  
   - 使用 SQuAD(v1.1/v2.0) 数据集，包含 `context`/`question`/`answers` 字段
   - v2.0 额外包含无法回答的问题，需模型判断是否可回答
 
2. **关键预处理**  
   - **长文本分割**：`max_length=384` + `doc_stride=128` 滑动窗口
   - **答案定位**：通过 `offset_mapping` 映射 token 位置到原始文本
   - **特殊标记**：不可回答时使用 `[CLS]` 标记索引
 
3. **模型训练**
 ```python
  model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
  trainer = Trainer(
       model,
       args=TrainingArguments(learning_rate=2e-5, batch_size=64, epochs=3),
       train_dataset=tokenized_datasets["train"],
       eval_dataset=tokenized_datasets["validation"]
   )
  ```
## PEFT库LoRA实战 - OpenAI Whisper-large-v2 语音识别微调
 
## 核心亮点
1. **LoRA微调技术**：在Whisper-large-v2模型上实现ASR任务的高效微调
2. **int8量化训练**：降低显存消耗同时保持精度（训练显存减少约75%）
3. **端到端流程**：涵盖数据准备、模型配置、训练、保存和推理全流程
 
---
 
## 1. 全局参数设置
```python
model_name_or_path = "openai/whisper-large-v2"
model_dir = "models/whisper-large-v2-asr-int8"
language = "Chinese (China)"
language_abbr = "zh-CN"
task = "transcribe"
dataset_name = "mozilla-foundation/common_voice_11_0"
batch_size = 64
```
## 2. 数据准备
数据集加载
```python
from datasets import load_dataset, DatasetDict
 
common_voice = DatasetDict()
common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train")
common_voice["validation"] = load_dataset(dataset_name, language_abbr, split="validation")
```
关键预处理
字段清理：移除accent, age, client_id等无关字段
音频降采样：48kHz → 16kHz（Whisper预训练采样率）
数据抽样（演示用）：
训练集：640个样本
验证集：320个样本

数据预处理函数
```pythonn
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch
```

## 3. 模型准备

int8量化加载
```python
from transformers import AutoModelForSpeechSeq2Seq
 
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name_or_path,
    load_in_8bit=True,
    device_map="auto"
)
```

LoRA配置
```python
from peft import LoraConfig
 
config = LoraConfig(
    r=4,                  # 秩参数
    lora_alpha=64,        # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 适配层
    lora_dropout=0.05,
    bias="none"
)
```
模型初始化
```python
from peft import prepare_model_for_int8_training, get_peft_model
 
model = prepare_model_for_int8_training(model)
peft_model = get_peft_model(model, config)
```
训练参数
```python
from transformers import Seq2SeqTrainingArguments
 
training_args = Seq2SeqTrainingArguments(
    output_dir=model_dir,
    per_device_train_batch_size=batch_size,
    learning_rate=1e-3,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    per_device_eval_batch_size=batch_size,
    logging_steps=10
)
训练结果
Epoch 1/1
Training Loss: 1.502400
Validation Loss: 1.081281

```
## 5. 模型保存与推理
模型保存
```python
trainer.save_model(model_dir)
推理示例
python
from transformers import pipeline
 
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=peft_model,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor
)
 
result = asr_pipeline("test_audio.wav")
print(result["text"])  # 输出识别文本
```













