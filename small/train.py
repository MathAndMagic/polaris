from accelerate import Accelerator
accelerator = Accelerator()

import datetime
import json
import numpy as np
import os
import random
import torch
import wandb
import warnings

from dataclasses import dataclass, field
from datasets import load_dataset, Dataset
from datetime import timezone 
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, HfArgumentParser, DataCollatorForLanguageModeling
from trl import SFTTrainer
from typing import List, Optional, Union, Any, Dict, Tuple

os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"]     = "false"

@dataclass
class ScriptArguments:
    checkpoint: Optional[str] = field(default="Open-Orca/Mistral-7B-OpenOrca", metadata={"help": "Checkpoint to resume training from"})
    dataset: Optional[str] = field(default="StarfleetAI/function-calling", metadata={"help": "Dataset to train on"})
    output_dir: Optional[str] = field(default="out", metadata={"help": "Output directory"})
    seed: Optional[int] = field(default=23420, metadata={"help": "Rng seed"})
    learning_rate: Optional[float] = field(default=5e-6, metadata={"help": "Learning rate"})
    num_epochs: Optional[float] = field(default=2, metadata={"help": "Number of epochs"})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push to hub"})
    wandb_project: Optional[str] = field(default="polaris-small", metadata={"help": "Wandb project name"})

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Union[str, List[int]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        self.instruction_token_ids = self.tokenizer.encode(instruction_template, add_special_tokens=False)

        self.response_template = response_template
        self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            response_token_ids_idxs = []
            human_token_ids_idxs = []

            for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                # find the indexes of the start of a response.
                if (
                    self.response_token_ids
                    == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                ):
                    response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

            if len(response_token_ids_idxs) == 0:
                warnings.warn(
                    f"Could not find response key `{self.response_template}` in the "
                    f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                    f"This instance will be ignored in loss calculation. "
                    f"Note, if this happens often, consider increasing the `max_seq_length`."
                )
                batch["labels"][i, :] = self.ignore_index

            if len(response_token_ids_idxs) > 0:
                human_token_ids_idxs.append(0)

            # For each response, find a closest token <|im_start|> after it
            for start_idx in response_token_ids_idxs:
                for (idx, tid) in enumerate(batch["labels"][i][start_idx:]):
                    if tid == 32001: # <|im_start|>
                        human_token_ids_idxs.append(start_idx + idx)
                        break

            for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                # Make pytorch loss function ignore all non response tokens
                if idx != 0:
                    batch["labels"][i, start:end] = self.ignore_index
                else:
                    batch["labels"][i, :end] = self.ignore_index

            # If only non-assistant tokens are left, ignore them
            if len(human_token_ids_idxs) > len(response_token_ids_idxs):
                batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        return batch

def from_to_name(from_name):
    if from_name == "human":
        return "user"
    elif from_name == "gpt":
        return "assistant"
    elif from_name == "system":
        return "system"
    elif from_name == "function_response":
        return "fn"
    else:
        raise ValueError(f"Unknown message type {from_name}")

def apply_chat_template(row):
    messages = row["conversations"]
    strings = []

    if messages[0]["from"] != "system":
        warnings.warn("System message is not set, adding default system message")

        text = '## Configuration\n\nFunctions: disabled\n\n---\n\nYou are a helpful assistant.'
        messages.insert(0, {
            "from": "system",
            "value": text
        })

    for message in messages:
        strings.append(f"<|im_start|>{from_to_name(message['from'])}\n{message['value']}<|im_end|>")
    
    row["conversations"] = "".join(strings)

    return row

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    torch.manual_seed(script_args.seed)

    os.environ["WANDB_PROJECT"] = script_args.wandb_project

    model = AutoModelForCausalLM.from_pretrained(
        script_args.checkpoint,
        device_map = {"": accelerator.local_process_index},
        torch_dtype = torch.bfloat16,
        trust_remote_code = True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.checkpoint,
        use_fast = False
    )
    tokenizer.add_special_tokens({
        "pad_token": tokenizer.unk_token,
        "eos_token": "<|im_end|>",
        "additional_special_tokens": [
            "<|fn_start|>",
            "<|fn_end|>"
        ]
    })
    tokenizer.clean_up_tokenization_spaces = True
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    tokenizer.padding_side = 'left'
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    print(model)

    for name, param in model.named_parameters():
        param.requires_grad = True

    dataset = load_dataset(script_args.dataset, split='train').map(apply_chat_template)
    ds = dataset.train_test_split(test_size=0.05, seed=23420)
    train_ds = ds["train"]
    test_ds = ds["test"]

    # Filter out the longest conversations
    def filter_dataset(rows):
        return Dataset.from_list(sorted(rows, key=lambda x: len(x["conversations"]))[:-len(rows)//10])

    train_ds = filter_dataset(train_ds)
    test_ds = filter_dataset(test_ds)
    print(train_ds[0])
    print(test_ds[0])

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    print(
        "Number of trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    print(
        "Trainable parameter types:",
        set(param.dtype for param in model.parameters() if param.requires_grad)
    )

    trainer = SFTTrainer(
        model = model,
        train_dataset = train_ds.shuffle(seed=script_args.seed),
        eval_dataset = test_ds,
        dataset_text_field = "conversations",
        dataset_num_proc = 4,
        tokenizer = tokenizer,
        packing = False,
        max_seq_length = 4096,
        data_collator = DataCollatorForCompletionOnlyLM(
            instruction_template = "<|im_end|>",
            response_template = "<|im_start|>assistant\n",
            tokenizer = tokenizer
        ),
        args = TrainingArguments(
            report_to = "wandb",
            output_dir = script_args.output_dir,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 64,
            optim = "paged_adamw_32bit",
            learning_rate = script_args.learning_rate,
            lr_scheduler_type = "cosine",
            save_strategy = "no",
            logging_steps = 1,
            num_train_epochs = script_args.num_epochs,
            neftune_noise_alpha = 5,
            bf16 = True
        ),
    )
    trainer.train()

    tokenizer.add_eos_token = False

    print("Saving model")

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(script_args.output_dir)

    if script_args.push_to_hub:
        print("Pushing to hub")
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
        trainer.push_to_hub("StarfleetAI/polaris-small", f"revision-{timestamp}")

if __name__ == "__main__":
    main()
