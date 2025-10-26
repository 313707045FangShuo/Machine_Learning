---
title: 'README: 使用 LoRA 進行模型微調'

---

# README: 使用 LoRA 進行模型微調

## 專案名稱
使用 LoRA (Low-Rank Adaptation) 進行 LLaMA 模型微調，確保模型可金隼回答中文選擇題。

## 專案目標
本專案旨在使用 LoRA 技術對大型語言模型 (LLaMA) 進行微調，以解決特定的文本生成問題，例如問答或其他指令完成任務。本 README 將詳細介紹 LoRA 微調的原理、實現步驟及相關的程式碼架構。

---

## LoRA
LoRA（Low-Rank Adaptation）是一種用於模型參數高效微調的技術。它的主要特點是：
1. **降低計算需求與記憶體占用**：僅微調低秩矩陣，無需改動原模型參數。
2. **加速微調流程**：只需訓練少量參數，減少時間與硬體資源需求。
3. **可拔插式設計**：微調的參數可輕鬆加載或卸載，無需影響原模型。

---

## 環境使用

1. Python 版本 3.10.13
2. 需要安裝以下主要套件：
   - `transformers`
   - `peft`
   - `torch`
   - `datasets`
   - `bitsandbytes`

可使用以下指令安裝：
```
!pip install pandas openpyxl
!pip install bitsandbytes==0.38.1
!pip install torch transformers==4.31.0 peft==0.4.0 sentencepiece bitsandbytes accelerate
!pip install huggingface-hub==0.27.0
```

---

## 微調步驟

### 1. 準備數據集
首先，需要將資料準備成標註好的數據集，先將原始xlsx檔製作成.json檔，再製作成包含訓練與驗證數據。數據格式如下：
```json
[
    {
        "input_text": "問題描述或輸入內容",
        "target_text": "模型應輸出的答案"
    },
    ...
]
```

### 2. 加載基礎模型
選擇預訓練的 LLaMA 模型作為基礎模型，本諄案使用的基礎模型是[Hugging Face 模型："lianghsun/Llama-3.2-Taiwan-3B-Instruct"](https://huggingface.co/lianghsun/Llama-3.2-Taiwan-3B-Instruct) (此模型需要向作者申請使用權)。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型名稱或本地路徑
model_name_or_path = "lianghsun/Llama-3.2-Taiwan-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto"
)
```

### 3. 配置 LoRA
設置 LoRA 微調參數。
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    inference_mode=False,  # 禁用推理模式（啟用訓練模式）
    lora_alpha=16,  # LoRA 的 alpha 超參數
    lora_dropout=0.05,  # LoRA dropout 超參數
    r=32,  # LoRA 等級
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj"
    ],  # LoRA 應用的目標模組
    task_type="CAUSAL_LM"  # 任務類型
)

# 添加 LoRA 配置到模型
lora_model = get_peft_model(base_model, lora_config)
```

### 4. 數據處理與訓練
#### 數據處理
將數據處理成模型的輸入格式。
```python
def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    labels = tokenizer(
        examples["target_text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )["input_ids"]
    labels = [-100 if token == tokenizer.pad_token_id else token for token in labels]
    model_inputs["labels"] = labels
    return model_inputs

# Tokenize 數據集
train_dataset = raw_train_dataset.map(tokenize_function, batched=True)
valid_dataset = raw_valid_dataset.map(tokenize_function, batched=True)
```

#### 訓練參數
設置訓練超參數並啟動訓練。
```python
# 配置訓練參數
training_args = TrainingArguments(
    output_dir="./lora_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=1e-4,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    fp16=True,
    bf16=False,  
    report_to="none",
)

# 使用 Trainer 進行微調
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tr_tokenized_dataset,
    eval_dataset=vd_tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 開始訓練
trainer.train()
```

### 5. 保存模型
保存微調後的 LoRA 模型參數。
```python
lora_model.save_pretrained("./lora_finetuned_model")
tokenizer.save_pretrained("./lora_finetuned_model")
```

下載基礎模型。
```
# 加載 LoRA 微調模型
lora_model = PeftModel.from_pretrained(base_model, "./lora_finetuned_model_6")

# 合併並卸載 LoRA 層
full_model = lora_model.merge_and_unload()

# 保存完整模型
full_model.save_pretrained("./lora_finetuned_model_6")
tokenizer.save_pretrained("./lora_finetuned_model_6")
```

---

## 預測與測試
使用微調後的模型進行文本生成。
```python
finetuned_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=token)
model_path = "./lora_finetuned_model"
finetuned_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")

# 創建 pipeline
qa_pipeline = pipeline(
    "text-generation",
    model=finetuned_model,
    tokenizer=finetuned_tokenizer,
)
```

---

## 文件結構
```
├── AI.xlsx                      # 訓練資料
├── AI1000.xlsx                  # 預測資料
├── AI.json                      # 訓練資料產生數據集
├── AI1000.json                  # 預測資料產生數據集
├── lora_finetuned_model         # 微調後的 LoRA 模型
├── LLM.ipynb                    # 相關程式碼
└── README.md                    # 專案說明文件
```

