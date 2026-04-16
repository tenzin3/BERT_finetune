import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example



def custom_transform(example):

    text = example["text"]
    words = text.split()

    keyboard_adj = {
        'a': ['q', 'w', 's', 'z'],
        'b': ['v', 'g', 'h', 'n'],
        'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'r', 'f', 'c', 'x'],
        'e': ['w', 's', 'd', 'r'],
        'f': ['d', 'r', 't', 'g', 'v', 'c'],
        'g': ['f', 't', 'y', 'h', 'b', 'v'],
        'h': ['g', 'y', 'u', 'j', 'n', 'b'],
        'i': ['u', 'j', 'k', 'o'],
        'j': ['h', 'u', 'i', 'k', 'm', 'n'],
        'k': ['j', 'i', 'o', 'l', 'm'],
        'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k'],
        'n': ['b', 'h', 'j', 'm'],
        'o': ['i', 'k', 'l', 'p'],
        'p': ['o', 'l'],
        'q': ['w', 'a'],
        'r': ['e', 'd', 'f', 't'],
        's': ['a', 'w', 'e', 'd', 'x', 'z'],
        't': ['r', 'f', 'g', 'y'],
        'u': ['y', 'h', 'j', 'i'],
        'v': ['c', 'f', 'g', 'b'],
        'w': ['q', 'a', 's', 'e'],
        'x': ['z', 's', 'd', 'c'],
        'y': ['t', 'g', 'h', 'u'],
        'z': ['a', 's', 'x']
    }

    protected_words = {"not", "no", "never"}

    def split_punct(token):
        start = 0
        end = len(token)

        while start < end and not token[start].isalnum():
            start += 1
        while end > start and not token[end - 1].isalnum():
            end -= 1

        prefix = token[:start]
        core = token[start:end]
        suffix = token[end:]
        return prefix, core, suffix

    def typo_replace(chars):
        alpha_positions = [i for i, ch in enumerate(chars) if ch.isalpha()]
        if not alpha_positions:
            return chars

        i = random.choice(alpha_positions)
        c = chars[i].lower()
        if c in keyboard_adj:
            repl = random.choice(keyboard_adj[c])
            chars[i] = repl.upper() if chars[i].isupper() else repl
        return chars

    def typo_swap(chars):
        possible = [
            i for i in range(len(chars) - 1)
            if chars[i].isalpha() and chars[i + 1].isalpha()
        ]
        if not possible:
            return chars

        i = random.choice(possible)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return chars

    def typo_delete(chars):
        alpha_positions = [i for i, ch in enumerate(chars) if ch.isalpha()]
        if not alpha_positions:
            return chars
        
        i = random.choice(alpha_positions)
        del chars[i]
        return chars

    new_words = []

    for word in words:
        prefix, core, suffix = split_punct(word)

        if len(core) <= 3 or not any(ch.isalpha() for ch in core):
            new_words.append(word)
            continue
        if core.lower() in protected_words:
            new_words.append(word)
            continue

        if random.random() < 0.32:
            chars = list(core)

            typo_type = random.choices(
                population=["replace", "swap", "delete"],
                weights=[0.6, 0.3, 0.10],
                k=1
            )[0]

            if typo_type == "replace":
                chars = typo_replace(chars)
            elif typo_type == "swap":
                chars = typo_swap(chars)
            elif typo_type == "delete":
                chars = typo_delete(chars)

            core = "".join(chars)

        new_words.append(prefix + core + suffix)

    example["text"] = " ".join(new_words)

    return example