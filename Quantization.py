from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
import torch
import os
from datasets import load_dataset
import peft
from peft import LoraConfig
import transformers
from trl import SFTTrainer

working_dir = "./"
output_directory = os.path.join(working_dir, "lora")


dataset = "openai/gsm8k"
data = load_dataset(dataset, "main")

model_name = "meta-llama/Llama-3.2-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input = tokenizer(
    "Natalia sold clips to 48 of her friends in April, and then she sold half as \
many clips in May. How many clips did Natalia sell altogether in April and May?",
    return_tensors="pt",
).to("cuda")

response = quantized_model.generate(**input, max_new_tokens=100)
print(tokenizer.batch_decode(response, skip_special_tokens=True))


tokenizer.pad_token = tokenizer.eos_token
data = data.map(
    lambda samples: tokenizer(
        samples["question"],
        samples["answer"],
        truncation=True,
        padding="max_length",
        max_length=100,
    ),
    batched=True,
)
train_sample = data["train"].select(range(400))
print(train_sample)
print(train_sample[:1])

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)


training_args = TrainingArguments(
    output_dir=output_directory,
    auto_find_batch_size=True,
    learning_rate=3e-4,
    num_train_epochs=5,
)

trainer = SFTTrainer(
    model=quantized_model,
    args=training_args,
    train_dataset=train_sample,
    peft_config=lora_config,
    tokenizer=tokenizer,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

model_path = os.path.join(output_directory, f"lora_model")
trainer.model.save_pretrained(model_path)

loaded_model = AutoPeftModelForCausalLM.from_pretrained(
    model_path, quantization_config=bnb_config, device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input = tokenizer(
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    return_tensors="pt",
).to("cuda")

response = loaded_model.generate(**input, max_new_tokens=100)
print(tokenizer.batch_decode(response, skip_special_tokens=True))
