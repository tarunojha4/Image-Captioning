"""Vocabulary and Tokenizer for Image Captioning.

Supports frequency filtering, special tokens, serialization/deserialization to JSON,
and works seamlessly across PyTorch and TensorFlow data pipelines.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union

import config


def clean_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words and punctuation cleanly.

    Does not depend on external NLTK punkt downloads, making it robust in any environment.
    """
    text = text.strip().lower()
    # Match words and individual punctuation marks
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens


class Vocabulary:
    """Manages mapping between vocabulary words and integer indices."""

    def __init__(self, freq_threshold: int = config.FREQ_THRESHOLD):
        self.freq_threshold = freq_threshold
        self.itos: Dict[int, str] = {
            config.PAD_IDX: config.SPECIAL_TOKENS["pad_token"],
            config.START_IDX: config.SPECIAL_TOKENS["start_token"],
            config.END_IDX: config.SPECIAL_TOKENS["end_token"],
            config.UNK_IDX: config.SPECIAL_TOKENS["unk_token"],
        }
        self.stoi: Dict[str, int] = {v: k for k, v in self.itos.items()}
        self.word_counts: Counter = Counter()

    def __len__(self) -> int:
        return len(self.itos)

    def add_word(self, word: str) -> None:
        """Add a single word to the vocabulary if it meets the frequency threshold."""
        if word not in self.stoi:
            idx = len(self.itos)
            self.itos[idx] = word
            self.stoi[word] = idx

    def build_vocabulary(self, sentence_list: List[str]) -> None:
        """Build vocabulary from a collection of raw caption strings."""
        for sentence in sentence_list:
            tokens = clean_tokenize(sentence)
            self.word_counts.update(tokens)

        for word, count in self.word_counts.items():
            if count >= self.freq_threshold:
                self.add_word(word)

    def numericalize(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert a sentence into a list of vocabulary token indices.

        Args:
            text: Input caption string.
            add_special_tokens: Whether to prepend <start> and append <end>.

        Returns:
            List of integer token IDs.
        """
        tokens = clean_tokenize(text)
        result = []
        if add_special_tokens:
            result.append(config.START_IDX)

        unk_idx = config.UNK_IDX
        for token in tokens:
            result.append(self.stoi.get(token, unk_idx))

        if add_special_tokens:
            result.append(config.END_IDX)

        return result

    def decode(
        self, indices: List[int], remove_special_tokens: bool = True
    ) -> str:
        """Convert a list of token indices back into a human-readable string.

        Args:
            indices: List of token IDs.
            remove_special_tokens: If True, filters out <pad>, <start>, <end>, <unk>.

        Returns:
            Decoded string caption.
        """
        words = []
        special = {config.PAD_IDX, config.START_IDX, config.END_IDX, config.UNK_IDX}

        for idx in indices:
            if idx == config.END_IDX and remove_special_tokens:
                break
            if remove_special_tokens and idx in special:
                continue
            word = self.itos.get(idx, config.SPECIAL_TOKENS["unk_token"])
            words.append(word)

        # Clean punctuation spacing (e.g. "a dog , sitting ." -> "a dog, sitting.")
        sentence = " ".join(words)
        sentence = re.sub(r'\s+([.,!?;:])', r'\1', sentence)
        return sentence.capitalize()

    def save(self, filepath: Union[str, Path]) -> None:
        """Save vocabulary mappings to a JSON file."""
        data = {
            "freq_threshold": self.freq_threshold,
            "itos": self.itos,
            "stoi": self.stoi,
            "word_counts": dict(self.word_counts),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Vocabulary":
        """Load vocabulary mappings from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(freq_threshold=data.get("freq_threshold", config.FREQ_THRESHOLD))
        vocab.itos = {int(k): v for k, v in data["itos"].items()}
        vocab.stoi = data["stoi"]
        vocab.word_counts = Counter(data.get("word_counts", {}))
        return vocab
