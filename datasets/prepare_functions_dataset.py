# Copyright 2024 StarfleetAI
# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset, Dataset
import json
import re

def cleanup(example):
    messages = example['conversations']

    system_message = messages[0]

    assert system_message['from'] == 'system'

    # Clean up the system message
    text = system_message['value']

    lines = []

    # Check if text has a function call
    if '<functioncall>' in text:
        # Remove the functioncall example (`<functioncall>(.*?)</functioncall>`)
        text = re.sub(r'<functioncall>(.*?)</functioncall>', '', text)

        # Find the first `{` and the last `}`, parse the JSON in between.
        left_bracket = text.find('{')
        right_bracket = text.rfind('}')

        assert left_bracket != -1 and right_bracket != -1

        funcs = text[left_bracket:right_bracket+1].split('\n\n')
        funcs = [json.loads(func) for func in funcs]

        # Construct new text
        lines.append("## Configuration\n\n")

        lines.append("Functions: enabled\n\n")

        lines.append("## Available Functions\n\n")

        funcs = [f"{json.dumps(func)}\n" for func in funcs]
        lines.append(''.join(funcs) + '\n')

        lines.append("---\n\n")

        lines.append("You are a helpful assistant.")
    else:
        lines.append("## Configuration\n\n")

        lines.append("Functions: disabled\n\n")

        lines.append("---\n\n")

        lines.append("You are a helpful assistant.")
    
    text = ''.join(lines)


    messages[0]['value'] = text

    for (i, message) in enumerate(messages[1:]):
        text = message['value'].replace('<functioncall>', '<|fn_start|>').replace('</functioncall>', '<|fn_end|>').replace('<|fn_start|> ', '<|fn_start|>').replace(' <|fn_end|>', '<|fn_end|>')
        
        # Check if text has a function call
        if '<|fn_start|>' in text:
            # Find the first `{` and the last `}`, parse the JSON in between.
            left_bracket = text.find('{')
            right_bracket = text.rfind('}')

            assert left_bracket != -1 and right_bracket != -1

            # Find the arguments
            left_apostrophe = text.find("'", left_bracket)
            right_apostrophe = text.rfind("'")

            args_text = text[left_apostrophe+1:right_apostrophe]
            # Unescape the apostrophes
            args_text = args_text.replace("\\'", "'")
            arguments = json.loads(args_text)

            text = text.replace(text[left_apostrophe:right_apostrophe+1], '""')
            right_bracket = text.rfind('}')
            func = json.loads(text[left_bracket:right_bracket+1])
            func['arguments'] = arguments

            # Replace the old function call with the new one
            text = text.replace(text[left_bracket:right_bracket+1], json.dumps(func))

        messages[i+1]['value'] = text

    example['conversations'] = messages

    return example

def main():
    dataset = load_dataset("hypervariance/function-calling-sharegpt", split='train').map(cleanup)
    
    print(dataset[:5])
    
    dataset.push_to_hub('StarfleetAI/function-calling')

if __name__ == '__main__':
    main()
