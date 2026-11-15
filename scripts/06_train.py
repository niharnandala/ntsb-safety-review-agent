# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "agenttune @ git+https://github.com/Lexsi-Labs/AgentTune.git",
# ]
# ///
"""
06 · Train Tiny Aya on the worked runs from 05 (LoRA + supervised fine-tuning).

Runs as a Hugging Face Job on one A100. Steps:
1. download the worked runs (private dataset niharnandala/ntsb-review-trajectories)
2. turn every run into tokens + a "labels" list: only Tiny Aya's own turns (the tool calls) are learned,
   the system prompt, the task and the tool results are only read (label -100)
3. LoRA with AgentTune's settings (r=16, alpha=32, dropout 0.05, target q/k/v/o projections)
4. TRL's SFTTrainer (the trainer AgentTune wraps), 2 passes over the data
5. a quick check: does the trained model start a case with get_recommendation?
6. upload the LoRA adapter + the training log to niharnandala/tiny-aya-ntsb-reviewer-lora (private for now)

AgentTune's own create_distill_trainer writes tool calls as <tool_call>...</tool_call>; our runtime (notebook 04)
reads JSON lists, so we tokenize our own chats in exactly the 04 format and give them to the same trainer.
"""
import json
import os
import time

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from agenttune.agentic.distill_trainer import _DEFAULT_LORA_TARGETS

MODEL_ID = "CohereLabs/tiny-aya-global"
DATASET_ID = "niharnandala/ntsb-review-trajectories"
ADAPTER_REPO = "niharnandala/tiny-aya-ntsb-reviewer-lora"
OUT = "/tmp/tiny-aya-ntsb-lora"
MAX_LENGTH = 8192
TOKEN = os.environ["HF_TOKEN"]

started = time.time()
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))

# ---------------------------------------------------------------- 1. the worked runs
data = load_dataset(DATASET_ID, token=TOKEN)
train_rows = data["train"]
val_rows = data["validation"]
print(len(train_rows), "training runs |", len(val_rows), "check runs")

tok = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


# ---------------------------------------------------------------- 2. tokens + labels
def chat_ids(messages, add_generation_prompt):
    rendered = tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, return_dict=True)
    return list(rendered["input_ids"])


def to_tokens(messages):
    full = chat_ids(messages, False)
    labels = []
    for i in range(len(full)):
        labels.append(-100)

    learned = 0
    for k in range(len(messages)):
        if messages[k]["role"] != "assistant":
            continue
        # everything before this reply, plus the "now the assistant speaks" marker
        before = chat_ids(messages[:k], True)
        # everything up to and including this reply (with its end-of-turn token)
        upto = chat_ids(messages[:k + 1], False)
        if full[:len(before)] != before or full[:len(upto)] != upto:
            return None, None          # the template doesn't render step by step the same way: skip this run
        for position in range(len(before), len(upto)):
            labels[position] = full[position]
            learned = learned + 1

    if learned == 0:
        return None, None
    return full, labels


def build(rows, name):
    out = []
    skipped = 0
    for row in rows:
        ids, labels = to_tokens(row["messages"])
        if ids is None:
            skipped = skipped + 1
            continue
        out.append({"input_ids": ids, "labels": labels})
    print(f"{name}: {len(out)} runs tokenized, {skipped} skipped")
    if skipped > 0.05 * len(rows):
        raise ValueError("too many runs skipped: the chat template does not render turn by turn the same way")
    return Dataset.from_list(out)


train_dataset = build(train_rows, "train")
val_dataset = build(val_rows, "validation")

# show what is learned in the first run: only the tool calls should appear
first = train_dataset[0]
learned_ids = []
for token_id, label in zip(first["input_ids"], first["labels"]):
    if label != -100:
        learned_ids.append(token_id)
print("\n--- learned part of run 1 (should be only the JSON tool calls) ---")
print(tok.decode(learned_ids)[:1500])
total_tokens = 0
learned_tokens = 0
for row in train_dataset:
    total_tokens = total_tokens + len(row["input_ids"])
    for label in row["labels"]:
        if label != -100:
            learned_tokens = learned_tokens + 1
print(f"\n{total_tokens:,} tokens in total, {learned_tokens:,} of them learned ({100 * learned_tokens / total_tokens:.0f}%)")

# ---------------------------------------------------------------- 3. model + LoRA (AgentTune's settings)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, token=TOKEN)

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=_DEFAULT_LORA_TARGETS,
)

config = SFTConfig(
    output_dir=OUT,
    num_train_epochs=2,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    max_length=MAX_LENGTH,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=40,
    save_strategy="no",
    report_to="none",
)

# ---------------------------------------------------------------- 4. train
trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tok,
    peft_config=lora,
)
trainer.model.print_trainable_parameters()
trainer.train()
final_eval = trainer.evaluate()
print("final check-set loss:", final_eval)

# ---------------------------------------------------------------- 5. quick check
first_val = val_rows[0]["messages"]
prompt = tok.apply_chat_template(first_val[:2], add_generation_prompt=True, return_tensors="pt", return_dict=True)
prompt = prompt.to(trainer.model.device)
trainer.model.eval()
with torch.no_grad():
    out = trainer.model.generate(**prompt, max_new_tokens=60, do_sample=False)
first_reply = tok.decode(out[0][prompt["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n--- trained model, first reply on a check case ---")
print(first_reply)

# ---------------------------------------------------------------- 6. save + upload
trainer.model.save_pretrained(OUT)
tok.save_pretrained(OUT)

summary = {
    "base_model": MODEL_ID,
    "dataset": DATASET_ID,
    "training_runs": len(train_dataset),
    "check_runs": len(val_dataset),
    "tokens_total": total_tokens,
    "tokens_learned": learned_tokens,
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": list(_DEFAULT_LORA_TARGETS)},
    "epochs": 2,
    "learning_rate": 1e-4,
    "effective_batch": 8,
    "final_eval": final_eval,
    "first_reply_check": first_reply,
    "minutes": round((time.time() - started) / 60, 1),
    "gpu": torch.cuda.get_device_name(0),
}
with open(f"{OUT}/training_summary.json", "w") as f:
    json.dump(summary, f, indent=1)
with open(f"{OUT}/training_log.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=1)

api = HfApi(token=TOKEN)
api.create_repo(ADAPTER_REPO, private=True, exist_ok=True)
api.upload_folder(folder_path=OUT, repo_id=ADAPTER_REPO,
                  allow_patterns=["adapter_*", "*.json", "tokenizer*", "special_tokens*", "*.jinja"])
print("\nuploaded →", f"https://huggingface.co/{ADAPTER_REPO}")
print("done in", summary["minutes"], "minutes")
