from unsloth import FastLanguageModel
from datasets import load_dataset
import torch

model_name = "unsloth/llama-3-8b-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=4096,
    dtype=torch.float16,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
)

dataset = load_dataset("json", data_files="train.jsonl")["train"]

def format_prompt(example):
    return {
        "text": f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    }

dataset = dataset.map(format_prompt)

from transformers import TrainingArguments, Trainer

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir="./food_recipe_lora",
        save_strategy="epoch",
        report_to="none",
    ),
)

trainer.train()
