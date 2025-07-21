from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch
from datasets import Dataset

# 1. Dataset Creation
dataset_text = [
    "The quick brown fox jumps over the lazy dog.",
    "A cat is sitting on a mat.",
    "The sun is shining brightly.",
    "Birds are singing in the trees.",
    "The river flows gently.",
]

dataset = Dataset.from_dict({"text": dataset_text})

# 2. Model and Tokenizer Loading
model_name = "gpt2"  # Or a smaller GPT-2 variant like "gpt2-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. LoRA Configuration
lora_config = LoraConfig(
    r=8,  # Rank of the LoRA matrices
    lora_alpha=32,  # Scaling factor for the LoRA matrices
    lora_dropout=0.05,  # Dropout probability for LoRA layers
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Verify that only LoRA parameters are trainable

# 4. Training
training_args = TrainingArguments(
    output_dir="./lora_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()

# 5. Evaluation (Simple Generation)
def generate_text(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Generate text before fine-tuning
print("Before Fine-tuning:")
print(generate_text("The quick brown fox", model, tokenizer))

# Save the fine-tuned model
model.save_pretrained("./lora_fine_tuned")
tokenizer.save_pretrained("./lora_fine_tuned")

# Load the fine-tuned model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./lora_fine_tuned")
tokenizer = AutoTokenizer.from_pretrained("./lora_fine_tuned")

# Generate text after fine-tuning
print("\nAfter Fine-tuning:")
print(generate_text("The quick brown fox", model, tokenizer))