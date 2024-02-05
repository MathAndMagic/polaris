# Copyright 2024 StarfleetAI
# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset, concatenate_datasets, Dataset
import random

def orca_messages(example):
    messages = []

    messages.append({
        'from': 'human',
        'value': example['question']
    })
    messages.append({
        'from': 'gpt',
        'value': example['response']
    })

    return messages

# Comply with OpenAI format
def from_to_name(from_name):
    if from_name == "human":
        return "user"
    elif from_name == "gpt":
        return "assistant"
    elif from_name == "system":
        return "system"
    elif from_name == "function_response":
        return "tool"
    else:
        raise ValueError(f"Unknown message type {from_name}")

def apply_chat_template(messages):
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
    
    # NOTE: not using newlines here on purpose (differs from the original ChatML format), since:
    #       - there is no point of having them at all
    #       - it saves one token per message
    #       - it makes the labels masking code in the collator cleaner and more obvious
    return "".join(strings)

def make_conversations(examples, oo_ds, code_ds):
    expanded = []

    for example in examples:
        system = example['conversations'][0]
        expanded.append(apply_chat_template(example['conversations']))

        for i in range(1, 5):
            conversation = []
            conversation.append(system)

            for j in range(i):
                if random.random() < 0.5:
                    conversation += orca_messages(random.choice(oo_ds))
                else:
                    conversation += random.choice(code_ds)['conversations']

            conversation += example['conversations'][1:]
            expanded.append(apply_chat_template(conversation))

    return expanded

def main():
    languages = [
        'python',
        'go',
        'rust',
        'ruby'
    ]
    oo_dataset = load_dataset('Open-Orca/OpenOrca', split='train')
    code_dataset = load_dataset('StarfleetAI/Code-290k-ShareGPT-MarkedLanguage', split='train').filter(lambda example: example['language'] in languages)
    fn_dataset = load_dataset('StarfleetAI/function-calling', split='train')

    enabled_called = fn_dataset.filter(lambda example: example['functions_enabled'] and example['function_called'])
    enabled_not_called = fn_dataset.filter(lambda example: example['functions_enabled'] and not example['function_called'])
    disabled = fn_dataset.filter(lambda example: not example['functions_enabled'])

    enabled_called = make_conversations(enabled_called, oo_dataset, code_dataset)
    enabled_not_called = make_conversations(concatenate_datasets([enabled_not_called, enabled_not_called]), oo_dataset, code_dataset)
    disabled = make_conversations(concatenate_datasets([disabled, disabled]), oo_dataset, code_dataset)

    dataset = Dataset.from_dict({
        'conversations': enabled_called + enabled_not_called + disabled
    }).shuffle()

    print(f'Final dataset size: {len(dataset)}')

    print(dataset[:2])

    dataset.save_to_disk('polaris-dataset')

if __name__ == '__main__':
    main()
