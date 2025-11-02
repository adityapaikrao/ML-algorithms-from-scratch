# Tokenization in Large Language Models (LLMs)

---

## 1. Introduction

Tokenization is a **fundamental yet nuanced process** in the pipeline of large language models (LLMs). It converts raw text into discrete numerical units—**tokens**—that neural networks can process. Despite its apparent simplicity, tokenization influences nearly every aspect of model performance, from **context efficiency** to **language generalization** and **numerical reasoning**.

Early models used *naive character-level tokenization*, mapping each character to a unique integer. Modern models, however, employ **subword tokenization algorithms** such as *Byte Pair Encoding (BPE)* or *SentencePiece*, which produce more semantically meaningful and computationally efficient token sequences.

---

## 2. Naive Tokenization: The Starting Point

In character-level tokenization:

* Each character corresponds to a unique integer (token).
* The vocabulary is fixed to all characters in the dataset (e.g., 65 unique characters for Shakespeare’s corpus).
* Each token is embedded using a learned embedding table before being fed into the Transformer.

While simple, this approach scales poorly and lacks efficiency for long sequences. Hence, state-of-the-art models like GPT-2 and GPT-4 use **chunk-based subword tokenization** for better compression and generalization.

> **Example:**
> GPT-2 uses a byte-level BPE tokenizer with a vocabulary of ~50,257 tokens and a maximum context window of 1,024 tokens.

---

## 3. Practical Oddities and Tokenizer Differences

### 3.1. Tokenization Behavior in Practice

Using tools like **tiktoken**, one observes that:

* **Spaces are integral tokens**; `" token"` and `"token"` are treated differently.
* **Numbers** are split inconsistently (e.g., `"677"` vs. `"804"`).
* **Case sensitivity** affects token boundaries: `"egg"`, `" Egg"`, and `" space egg"` produce distinct token sequences.

### 3.2. Language Efficiency

* Non-English text (e.g., Korean) tends to produce **more tokens per sentence**, reducing effective context length.
* Tokenization efficiency affects performance on multilingual tasks.

### 3.3. Code Tokenization

* In GPT-2, every space in Python code is a separate token—extremely inefficient.
* GPT-4 improves this by merging frequent whitespace and symbol patterns, effectively **doubling context efficiency** for code.

### 3.4. Vocabulary Tradeoffs

* **Larger vocabularies** reduce sequence lengths but increase model size (embedding + output layers).
* **Smaller vocabularies** require more tokens per input but lower model complexity.

---

## 4. Unicode and UTF Encodings

### 4.1. Unicode Basics

* Strings are sequences of **Unicode code points**, each representing a character.
* Over 150,000 characters exist across languages and scripts.

### 4.2. UTF Encodings

* **UTF-8:** Variable-length (1–4 bytes), backward compatible with ASCII — the default in most NLP systems.
* **UTF-16/UTF-32:** Fixed-length but less efficient for English text.

### 4.3. Tokenization Challenge

Using raw Unicode code points as tokens leads to:

* Very large vocabularies (~150K tokens).
* Instability as Unicode evolves.

A naive UTF-8 byte tokenizer (vocabulary = 256 tokens) would generate **too many tokens** per input sequence, motivating the need for **compression algorithms like BPE**.

---

## 5. Byte Pair Encoding (BPE)

### 5.1. Concept

BPE is a **compression-based tokenization algorithm** that iteratively merges the most frequent adjacent token pairs into new tokens, reducing sequence length while maintaining a manageable vocabulary.

### 5.2. Algorithm Steps

1. Start with a base vocabulary of 256 byte tokens.
2. Count all adjacent token pairs in the dataset.
3. Merge the most frequent pair into a new token.
4. Repeat until desired vocabulary size is reached.

### 5.3. Practical Implementation

* BPE training is a **preprocessing step**, separate from LLM training.
* The tokenizer learns a **merge table** mapping token pairs to new IDs.
* Once trained, it can **encode (text → tokens)** and **decode (tokens → text)** efficiently.

---

## 6. GPT-2 Tokenizer and Regex-based Chunking

GPT-2 extends BPE with **regex-based pre-segmentation** to prevent semantically inconsistent merges.

### 6.1. Regex Chunking Rules

* Splits text into groups: letters, numbers, punctuation, whitespace, etc.
* Prevents merges across boundaries (e.g., between a word and punctuation).
* Spaces are handled specially to preserve common merged forms like `" space you"`.

### 6.2. Vocabulary and Implementation

* GPT-2’s vocabulary includes:

  * BPE merges (word fragments),
  * Encoder dictionary (token → string),
  * Special tokens (e.g., ``).
* The official OpenAI tokenizer is **inference-only**; training code was not released.

---

## 7. Tiktoken and GPT-4 Tokenizer Improvements

OpenAI’s **tiktoken** library provides efficient inference tokenization across GPT models.

### 7.1. GPT-4 Enhancements

* Improved regex rules (case-insensitive, better whitespace handling).
* Vocabulary expanded to ~100,000 tokens.
* Improved handling of code, numbers, and punctuation.
* Denser tokenization allows more text per context window.

### 7.2. Special Tokens

* Special tokens (e.g., `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`) are added for structured fine-tuning tasks.
* Adding new tokens requires resizing embedding and output layers—commonly done before fine-tuning.

### 7.3. Open-Source Alternatives

The **MBP repository** provides a GPT-4-style tokenizer, including both training and inference functionality.

---

## 8. SentencePiece Tokenizer

**SentencePiece**, used by models like *LLaMA* and *Mistral*, tokenizes **Unicode code points** rather than UTF-8 bytes.

### 8.1. Key Features

* Supports BPE and Unigram models.
* Works directly on Unicode code points.
* Includes byte fallback for unseen or rare characters.
* Offers preprocessing options like normalization and dummy prefixes.

### 8.2. Limitations

* Aggressive normalization may lose raw information.
* Requires careful configuration and tuning.
* Mandatory special tokens (e.g., `<unk>`, `<pad>`, `<s>`, `</s>`).
* Complex internal ordering: special tokens → byte tokens → merges → code points.

---

## 9. Vocabulary Size and Extending Tokenizers

### 9.1. Model Architecture Impact

Vocabulary size affects two major Transformer components:

* **Embedding layer:** Maps tokens to vectors.
* **LM head:** Projects model outputs back to token logits.

Larger vocabularies increase both memory and compute costs linearly.

### 9.2. Extending Vocabularies

* Common in fine-tuning to introduce domain-specific tokens or markers.
* Typically involves:

  * Resizing embedding and output layers.
  * Freezing pretrained weights.
  * Training new tokens from scratch.

### 9.3. Emerging Concepts

* **Gist tokens:** Compact learned tokens that summarize prompts efficiently.
* **Multimodal tokenization:** Unified token streams for text, image, video, and audio.

---

## 10. Practical Tokenization Issues

### 10.1. Character and Spelling Tasks

* Long subword tokens obscure character-level understanding (e.g., counting letters).

### 10.2. String Reversal

* Easier when tokenized per character; nearly impossible with long subwords.

### 10.3. Arithmetic Difficulty

* Arbitrary splitting of numbers disrupts numeric reasoning.

### 10.4. Non-English Efficiency

* Token inflation reduces effective context window and harms cross-lingual consistency.

### 10.5. Code Inefficiency

* GPT-2 wasted tokens on whitespace; GPT-4 fixed this, drastically improving code handling.

### 10.6. Special Token Hazards

* User-inserted special tokens (e.g., ``) can terminate generation prematurely.

### 10.7. Whitespace and Rare Tokens

* Trailing spaces create rarely seen tokens, destabilizing model predictions.

### 10.8. Partial Tokens

* Truncated tokens (e.g., `"default cel"`) cause unstable completions.

### 10.9. “Solid Gold Magikarp” Phenomenon

* Rare tokens (from training artifacts like Reddit usernames) produce untrained embeddings, leading to unpredictable behavior.

### 10.10. Data Format Efficiency

* YAML tokenizes more efficiently than JSON, affecting inference cost and context utilization.

---

## 11. Recommendations and Future Directions

* Use **tested tokenizers** (e.g., GPT-4’s tiktoken) for inference.
* When training from scratch, prefer **BPE-based approaches** using MBP or SentencePiece.
* Avoid ad-hoc tokenization schemes.
* Understand tokenization deeply—it explains many LLM quirks and performance anomalies.
* Future directions include:

  * **Tokenization-free models** that operate directly on bytes.
  * **Unified multimodal tokenization** for cross-domain learning.

---

## 12. Conclusion

Tokenization remains one of the most **critical, subtle, and error-prone components** in the design of LLMs. Its effects ripple through every stage of training, inference, and model behavior.
While future models may one day eliminate the need for tokenization entirely, today’s systems depend on it profoundly—making it an essential topic for any practitioner to understand deeply.

> *“Eternal glory awaits anyone who can remove tokenization.”* — Andrej Karpathy
