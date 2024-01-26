# StarfleetAI Polaris Small

Based on `Mistral-7b-OpenOrca`.

- [Model on HuggingFace](https://huggingface.co/StarfleetAI/polaris-small)
- [Training runs on W&B](https://wandb.ai/starfleetai/polaris-small)

## Features

- [x] Functions calling
  - [ ] Need more training on examples where function call should be performed in the middle/end parts of the conversation, rather than in the beginning
- [ ] Need some DPO
- [ ] Need to eliminate some hallucinations (it, for example regressing news instead of calling the `get_rss_feed` function, and stuff like that)
- [ ] Correct typography (currently: `Answer:123` instead of `Answer: 123`)
- [ ] Current date / time / timezone conversational abilities

## Recipe

1. Take [Mistral-7B-OpenOrca](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca)
2. Train [Code-290k](https://huggingface.co/datasets/ajibawa-2023/Code-290k-ShareGPT) over it (1 epoch seems to be enough)
3. Train [function-calling](https://huggingface.co/datasets/StarfleetAI/function-calling) over it (2-3 epochs)

## Training

```bash
accelerate launch --config_file ./fsdp_config.yaml train.py --num_epochs 3
```

### Useful options

- `--num_epochs` - number of epochs to train
- `--checkpoint` - source model, can be either a HuggingFace model or a local path
- `--dataset` - dataset to train on 
- `--output_dir` - where to save the resulting weights
