# Copyright 2024 StarfleetAI
# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset, Dataset
import json
import re

def cleanup(example):
    language = None

    # General cleanup
    for (i, message) in enumerate(example['conversations']):
        text = message['value']

        # Remove the \r
        text = text.replace('\r', '')

        example['conversations'][i]['value'] = text

    # Naive language detection by parsing the first code block
    for message in example['conversations']:
        text = message['value']

        left_apostrophe = text.find('```')
        if left_apostrophe == -1:
            continue
        closest_newline = text.find('\n', left_apostrophe)
        if closest_newline == -1:
            continue

        language = text[left_apostrophe+3:closest_newline]

    # If we found something, clean it up
    if language is not None:
        language = language.lower()

        if language == 'ruby-1.9.2' or language == 'rb':
            language = 'ruby'
        elif language == 'ts':
            language = 'typescript'
        elif language == 'js':
            language = 'javascript'
        elif language == 'golang':
            language = 'go'
        elif language == 'c#':
            language = 'csharp'
        elif language == 'f#':
            language = 'fsharp'
        elif language == 'c++' or language == 'cplusplus':
            language = 'cpp'
        elif language == 'shell':
            language = 'bash'
        elif language == 'plain':
            language = 'plaintext'
        elif language == 'python:' or language == '`python3' or language == 'python3' or language == 'py' or language == '{python}':
            language = 'python'
        elif language == 'assembly' or language == 'asm' or language == 'assembler':
            language = 'assembly'
        elif language == 'coffee':
            language = 'coffeescript'
        elif language == 'pseudo-code' or language == 'pseudo':
            language = 'pseudocode'
        elif language == 'yml':
            language = 'yaml'
        elif language == 'proto' or language == 'proto3':
            language = 'protobuf'
        elif language == 'objectivec' or language == 'objc':
            language = 'objective-c'
        elif language == 'docker':
            language = 'dockerfile'
        elif language == '{r}':
            language = 'r'
            

        stop_words = [' ', '?', '/', '[', '!', '`']
        if any(stop_word in language for stop_word in stop_words):
            language = None

    # If there is nothing up untill this point, try to find python keywords
    #
    # TODO: write euristic for other languages
    if language is None:
        for message in example['conversations']:
            keywords = ['python', 'pip', 'elif']
            if any(keyword in message['value'].lower() for keyword in keywords):
                language = 'python'
                break

    # Let's just give up
    if language is None or language == '':
        language = 'unknown'

    example['language'] = language.lower()

    return example

def main():
    dataset = load_dataset('ajibawa-2023/Code-290k-ShareGPT', split='train').map(cleanup)

    # languages = {}
    # for example in dataset:
    #     language = example['language']
    #     if language not in languages:
    #         languages[language] = 0
    #     languages[language] += 1
    #
    # languages = dict(sorted(languages.items()))
    # print(languages)

    dataset.push_to_hub('StarfleetAI/Code-290k-ShareGPT-MarkedLanguage')

if __name__ == '__main__':
    main()
