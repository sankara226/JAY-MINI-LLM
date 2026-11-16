"""Everything needed to talk to Jay: loading, prompt building, trimming, streaming.

Chat format (one flat token sequence, the history is replayed every turn):
    <|user|>{message}<|assistant|>{reply}<|end|><|user|>{message}<|assistant|>
"""
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from jay.model import BLOCK_SIZE, GPT

SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|end|>"]
EOT, USER, ASSISTANT, END = 0, 1, 2, 3   # ids of the special tokens, in this order

REPLY_ROOM = 120  # tokens kept free for Jay's answer when trimming the history

# The web app can serve several visitors at once, but they share one model, and each forward
# pass overwrites the model's saved attention weights (last_weights). The lock makes visitors
# take turns, one forward pass at a time, so Peek inside never reads someone else's weights.
MODEL_LOCK = threading.RLock()

# "chat_model.pt" is v1; the newer ones carry their version in the name.
MODEL_FILES = {"v1": "chat_model.pt", "v2": "chat_model_v2.pt", "v3": "chat_model_v3.pt"}


@dataclass
class Jay:
    """A loaded model plus its tokenizer."""
    model: GPT
    tokenizer: Tokenizer
    device: str
    version: str


# ---------------------------------------------------------------- loading

def available_versions(weights_dir):
    """Return the versions whose weight files exist, newest first, e.g. ["v3", "v2"]."""
    weights_dir = Path(weights_dir)
    found = [v for v, name in MODEL_FILES.items() if (weights_dir / name).exists()]
    return sorted(found, key=lambda v: int(v[1:]), reverse=True)


def load_tokenizer(weights_dir):
    tokenizer = Tokenizer.from_file(str(Path(weights_dir) / "tokenizer.json"))
    ids = [tokenizer.token_to_id(t) for t in SPECIAL_TOKENS]
    if ids != [EOT, USER, ASSISTANT, END]:
        raise ValueError(f"Unexpected special token ids {ids}: wrong tokenizer.json?")
    return tokenizer


def load_jay(weights_dir, model_file=None, device="cpu"):
    """Load the tokenizer and a chat model (by default the newest chat_model_v*.pt)."""
    weights_dir = Path(weights_dir)
    if model_file is None:
        versions = available_versions(weights_dir)
        if not versions:
            raise FileNotFoundError(f"No chat_model*.pt file found in {weights_dir}")
        model_file = MODEL_FILES[versions[0]]

    path = weights_dir / model_file
    model = GPT().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()  # turns dropout off

    match = re.search(r"_v(\d+)\.pt$", path.name)
    version = f"v{match.group(1)}" if match else "v1"
    return Jay(model=model, tokenizer=load_tokenizer(weights_dir), device=device, version=version)


# ---------------------------------------------------------------- tokens

def encode(tokenizer, text):
    return tokenizer.encode(text).ids


def decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False)


def decode_partial(tokenizer, ids):
    """Decode a reply that is still being written.

    One character (an emoji, an accent) can be split over two tokens; until the second
    token arrives the first one decodes to "�". Hide that half-character for now."""
    return decode(tokenizer, ids).rstrip("�").strip()


# ---------------------------------------------------------------- prompt

def turn_ids(tokenizer, user_msg, assistant_msg):
    """One finished exchange: <|user|>...<|assistant|>...<|end|>"""
    return [USER] + encode(tokenizer, user_msg) + [ASSISTANT] + encode(tokenizer, assistant_msg) + [END]


def build_prompt_ids(tokenizer, history, message):
    """history = [(user_msg, assistant_msg), ...]. The prompt ends right after <|assistant|>."""
    ids = []
    for user_msg, assistant_msg in history:
        ids += turn_ids(tokenizer, user_msg, assistant_msg)
    return ids + [USER] + encode(tokenizer, message) + [ASSISTANT]


def trim_history(tokenizer, history, message, reply_room=REPLY_ROOM):
    """Drop the oldest whole turns until prompt + reply room fits in BLOCK_SIZE tokens."""
    budget = BLOCK_SIZE - reply_room
    new_turn = len(encode(tokenizer, message)) + 2         # <|user|> message <|assistant|>
    turn_sizes = [len(turn_ids(tokenizer, u, a)) for u, a in history]

    kept = len(history)
    while kept > 0 and new_turn + sum(turn_sizes[-kept:]) > budget:
        kept -= 1                                            # forget the oldest turn
    return list(history[len(history) - kept:])


def prepare_prompt(tokenizer, history, message, reply_room=REPLY_ROOM):
    """Trim the history, build the prompt, and (rarely) shorten a very long message."""
    history = trim_history(tokenizer, history, message, reply_room)
    ids = build_prompt_ids(tokenizer, history, message)
    budget = BLOCK_SIZE - reply_room
    if len(ids) > budget:                                    # the message alone is too long:
        ids = [USER] + ids[-(budget - 1):]                   # keep its end, it holds the question
    return ids, history


# ---------------------------------------------------------------- generation

def next_token_probs(jay, ids, temperature=1.0, top_k=50):
    """Probabilities for the token that comes after `ids` (same sampling rules as GPT.generate)."""
    idx = torch.tensor([ids[-BLOCK_SIZE:]], device=jay.device)
    with MODEL_LOCK:
        logits, _ = jay.model(idx)
    logits = logits[:, -1, :] / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
    return F.softmax(logits, dim=-1)[0]


@torch.no_grad()
def stream_reply_ids(jay, prompt_ids, temperature=0.7, max_new_tokens=150, top_k=50):
    """Yield Jay's reply one token id at a time. Stops at <|end|> or <|endoftext|>."""
    jay.model.eval()
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        probs = next_token_probs(jay, ids, temperature, top_k)
        next_id = torch.multinomial(probs, num_samples=1).item()
        if next_id in (END, EOT):
            return
        ids.append(next_id)
        yield next_id


def stream_reply(jay, message, history=(), temperature=0.7, max_new_tokens=150):
    """Yield the reply text so far, growing token by token (handy for a typing effect)."""
    prompt_ids, _ = prepare_prompt(jay.tokenizer, list(history), message)
    reply = []
    for token_id in stream_reply_ids(jay, prompt_ids, temperature, max_new_tokens):
        reply.append(token_id)
        yield decode_partial(jay.tokenizer, reply)


# ---------------------------------------------------------------- peek inside

@torch.no_grad()
def peek_inside(jay, prompt_ids, reply_ids, layer=0, head=0, k=5, max_heatmap_tokens=40):
    """Top-k guesses for Jay's first reply token, and one head's attention over the reply.

    Returns (top, tokens, weights):
      top     = [(token_text, probability), ...] at the first generated position
      tokens  = the token pieces shown on the heatmap axes
      weights = a (T, T) list of lists: how much each token looked at each earlier one
    """
    jay.model.eval()
    probs = next_token_probs(jay, prompt_ids, temperature=1.0, top_k=None)
    values, indices = torch.topk(probs, k)
    top = [(decode(jay.tokenizer, [i]), p) for i, p in zip(indices.tolist(), values.tolist())]

    # Heatmap: the last user message + the reply (a full 256x256 grid is unreadable).
    start = max(i for i, t in enumerate(prompt_ids) if t == USER)
    ids = (list(prompt_ids) + list(reply_ids))[-BLOCK_SIZE:]
    start = max(0, start - (len(prompt_ids) + len(reply_ids) - len(ids)))
    stop = start + max_heatmap_tokens
    with MODEL_LOCK:  # run the model and read its attention before anyone else uses it
        jay.model(torch.tensor([ids], device=jay.device))
        weights = jay.model.blocks[layer].attn.last_weights[0, head, start:stop, start:stop].clone()
    tokens = [decode(jay.tokenizer, [i]) for i in ids[start:stop]]
    return top, tokens, weights.cpu().tolist()
