# StarfleetAI Polaris Small LLM

> [!IMPORTANT]  
> Even though the model is not yet meant for production usage, we still encourage you to give it a try and tell us what you think.
>
> Any feedback or suggestions are welcomed!

Based on `Mistral-7b-OpenOrca`.

- [Model on HuggingFace](https://huggingface.co/StarfleetAI/polaris-small)
- [Training runs on W&B](https://wandb.ai/starfleetai/polaris-small)

## About the Family

Polaris is a family of large language models, developed by StarfleetAI with the aim of being used for autonomous AI agent scenarios. Its target characteristics include the ability to call functions, a low hallucination rate, multimodality, and exceptional coding abilities.

## Features

- [x] Functions calling
  - [ ] Needs more training on examples where function call should be performed in the middle/end parts of the conversation, rather than in the beginning
- [ ] Needs to eliminate some hallucinations (it, for example, sometimes regressing news articles instead of calling the `get_rss_feed` function, and stuff like that)
- [ ] Correct typography (currently: `Answer:123` instead of `Answer: 123`)
- [ ] Current date / time / timezone conversational abilities
- [ ] Needs some DPO

## Recipe

1. Take [Mistral-7B-OpenOrca](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca)
2. Train [Code-290k](https://huggingface.co/datasets/ajibawa-2023/Code-290k-ShareGPT) over it (1 epoch seems to be enough)
3. Train [function-calling](https://huggingface.co/datasets/StarfleetAI/function-calling) over it (2-3 epochs)

Or just run

```bash
./train.sh
```

## Using Trainer

```bash
accelerate launch --config_file ./fsdp_config.yaml train.py --num_epochs 3
```

### Useful Options

- `--num_epochs` - number of epochs to train
- `--checkpoint` - source model, can be either a HuggingFace model or a local path
- `--dataset` - dataset to train on 
- `--output_dir` - where to save the resulting weights

## Prompt Format

We wanted our prompt to:

1. Have a configurable `system` prompt part, which is meant to be changed by the end user.
2. Describe the functions available to the model (in a specific, static place in the prompt, making it easier for the model to identify them).
3. Be capable of carrying additional configuration options in the future, such as the current date/time or the user's name.

In order to fullfill these needs, we designed the prompt format, which seems to tick all the boxes.

### With Functions

```
## Configuration

Functions: enabled

## Available Functions

{ ... }
{ ... }

---

You are a helpful assistant.
```

### Without Functions

```
## Configuration

Functions: disabled

---

You are a helpful assistant.
```

## Function Calling

In order for the model to call a function, we have introduced two new tokens: `<|fn_start|>` and `<|fn_end|>`. It's safe to assume that if the model decides to call a function, there will be no other response from it except for the function call between these special tokens. If this is not true for some of your cases, please feel free to contact us with examples.

For now, the model is only capable of calling one function at a time.

### Example Function Call

```
<|fn_start|>{"name": "generate_password", "arguments": {"length": 42}}<|fn_end|>
```

## Function Call Response

The model expects the function call response to be provided from a role `fn` right after the function call request:

```
<|im_start|>fn
{"result": "87cc47fbc865a290d7c7de4be3c893175c51a566b3"}<|im_end|>
```

There are no specific requirements on the response format. Feel free to respond with anything you want.

## Example Conversation

> [!NOTE]  
> Newlines after the `<|im_end|>` are included here only for ease of reading. In the actual chat template, we don't use newlines in this position.

```
<|im_start|>system
## Configuration

Functions: enabled

## Available Functions

{"name": "generate_password", "description": "Generate a random password", "parameters": {"type": "object", "properties": {"length": {"type": "integer", "description": "The length of the password"}}, "required": ["length"]}}

---

You are a helpful assistant.<|im_end|>
<|im_start|>user
Generate a password, 42 characters long<|im_end|>
<|im_start|>assistant
<|fn_start|>{"name": "generate_password", "arguments": {"length": 42}}<|fn_end|><|im_end|>
<|im_start|>fn
{"result": "87cc47fbc865a290d7c7de4be3c893175c51a566b3"}<|im_end|>
<|im_start|>assistant
Here is your random password: 87cc47fbc865a290d7c7de4be3c893175c51a566b3. Please make sure to save it in a secure place.<|im_end|>
```
