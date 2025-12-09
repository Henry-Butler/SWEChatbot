import streamlit as st
import pandas as pd
import os
import math
import copy
import json
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# optional baseline if transformers available
try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

# datasets & sentencepiece
from datasets import load_dataset
import sentencepiece as spm

CFG = {
    "vocab_size": 32000, # SentencePiece vocab size
    "seq_len": 128,   # Sequence Length per training example
    "batch_size": 8, # Batch size
    "epochs": 3, # Number of training epochs
    "d_model": 384, # Hidden size of transformer
    "n_heads": 6, # Number of attention heads
    "n_layers": 6, # Number of transformer blocks
    "d_ff": 1536, # Feed-forward inner dimension
    "dropout": 0.1, # Dropout rate
    "lr": 4e-4, # Learning rate
    "weight_decay": 0.01, # AdamW weight decay
    "warmup_steps": 400, # LR warmup steps
    "max_steps_between_checkpoints": 400, # Checkpoint frequency
    "grad_clip": 1.0, # Gradient clipping
    "grad_accum_steps": 1, # Gradient accumulation
    "seed": 42, # Random seed
    "save_dir": "./checkpoints", # Directory to save checkpoints
    "tokenizer_prefix": "spm_bpe", # Prefix for tokenizer model files
    "log_every_steps": 100, # Logging frequency
    "device": "cuda" if torch.cuda.is_available() else "cpu", # Use GPU if available
}

# Create checkpoint directory if it doesn't exist
os.makedirs(CFG["save_dir"], exist_ok=True)
device = torch.device(CFG["device"]) # Set device
torch.manual_seed(CFG["seed"]) # Set seeds so it can reprodcue
np.random.seed(CFG["seed"])
random.seed(CFG["seed"])

class Observer(ABC):
  @abstractmethod
  def update(self, subject, *args):
    pass

class TrainingObserver(Observer):
  def __init__(self, subject):
    subject.attach(self)
  def update(self, *args):
      print(args[1])

class Dataset:
    """
    Loads HF dataset and creates PyTorch DataLoaders of fixed-length sequences.
    Internally creates token sequences and DataLoaders.
    """
    def __init__(self, name: str = "wikitext", subset: str = "wikitext-2-raw-v1"):
        self.name = name
        self.subset = subset
        self.hf = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def accessData(self):
      # Load Hugging Face dataset
        self.hf = load_dataset(self.name, self.subset)
        print("[Dataset] Loaded HF dataset:", self.name, self.subset)
        return self.hf

    def build_dataloaders(self, tokenized_train: List[List[int]], tokenized_val: List[List[int]]):
        """
        Converts tokenized lists into fixed length sequences, pads short sequences,
        and produces TensorDatasets and DataLoaders.
        """
        seq_len = CFG["seq_len"]
        def make_sequences(token_lists: List[List[int]]) -> List[List[int]]:
            seqs = []
            for tokens in token_lists:
                if len(tokens) == 0:
                    continue
                # slice into non-overlapping chunks of seq_len
                for i in range(0, len(tokens), seq_len):
                    chunk = tokens[i:i+seq_len]
                    if len(chunk) < seq_len:
                        pad_id = 0  # SentencePiece pad id usually 0 (we set pad '<pad>')
                        chunk = chunk + [pad_id] * (seq_len - len(chunk))
                    seqs.append(chunk)
            return seqs

        # Create train and validation sequences
        train_seqs = make_sequences(tokenized_train)
        val_seqs = make_sequences(tokenized_val)

        if len(train_seqs) == 0:
            raise ValueError("No training sequences generated. Check tokenization and seq_len.")
        print(f"[Dataset] Train sequences: {len(train_seqs)} | Val sequences: {len(val_seqs)}")

        # convert to tensors: inputs = seq[:-1], labels = seq[1:]
        def seqs_to_tensors(seqs):
            arr_in = []
            arr_lbl = []
            for s in seqs:
                arr_in.append(s[:-1])
                arr_lbl.append(s[1:])
            return torch.tensor(arr_in, dtype=torch.long), torch.tensor(arr_lbl, dtype=torch.long)

        x_train, y_train = seqs_to_tensors(train_seqs)
        x_val, y_val = seqs_to_tensors(val_seqs)

        # Create PyTorch datasets
        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)

        # Create DataLoaders
        self.train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False)

        print("[Dataset] DataLoaders created.")
        return self.train_loader, self.val_loader


class PreProcessor:
    """
    Cleans Hugging Face dataset text for tokenizer training.
    Also, removes empty lines and normalizes text.
    """
    def __init__(self):
        self.raw_train = None # Will store raw text from HF dataset

    def loadData(self, hf_dataset):
        self.raw_train = hf_dataset["train"]
        return self.raw_train

    def cleanData(self) -> List[str]:
        cleaned = []
        for ex in self.raw_train:
            txt = ex.get("text", "").strip()
            if not txt:
                continue
            cleaned.append(txt.replace("\n", " "))
        print(f"[PreProcessor] Cleaned {len(cleaned)} documents.")
        return cleaned



class Tokenizer:
    """
    SentencePiece BPE wrapper. Trains and loads tokenizer as well as deals with encoding and decoding.
    """
    def __init__(self, model_prefix: str = CFG["tokenizer_prefix"], vocab_size: int = CFG["vocab_size"]):
        self.prefix = model_prefix
        self.vocab_size = vocab_size
        self.processor = None

    def train(self, texts: List[str]):
      # Save texts to a temporary file for SentencePiece training
        src_file = self.prefix + "_train.txt"
        with open(src_file, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(t + "\n")
                # Train SentencePiece BPE model
        spm.SentencePieceTrainer.Train(
            f"--input={src_file} --model_prefix={self.prefix} --vocab_size={self.vocab_size} "
            f"--model_type=bpe --character_coverage=1.0 --unk_piece=<unk> --pad_piece=<pad> --bos_piece=<bos> --eos_piece=<eos>"
        )
        # Load trained model
        self.processor = spm.SentencePieceProcessor()
        self.processor.Load(self.prefix + ".model")
        print(f"[Tokenizer] Trained SentencePiece model: {self.prefix}.model (vocab_size={self.processor.GetPieceSize()})")

    def load(self):
       # Load existing tokenizer model
        self.processor = spm.SentencePieceProcessor()
        self.processor.Load(self.prefix + ".model")
        print(f"[Tokenizer] Loaded {self.prefix}.model")

    def encode(self, text: str) -> List[int]:
       # Encode text to token IDs
        return self.processor.EncodeAsIds(text)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
      # Encode a batch of texts
        return [self.encode(t) for t in texts]

    def decode(self, ids: List[int]) -> str:
      # Convert token IDs back to text
        return self.processor.DecodeIds(ids)

class ModelBuilder(nn.Module, ABC):
  # Removed the abstract __init__ method
  @abstractmethod
  def _multi_head_attention(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
    pass
  @abstractmethod
  def _feed_forward(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
    pass
  @abstractmethod
  def forward(self, idx: torch.LongTensor) -> torch.Tensor:
    pass
  @abstractmethod
  def save_weights(self, path: str):
    pass
  @abstractmethod
  def load_weights(self, path: str, map_location=None):
    pass
  @abstractmethod
  def clone(self):
    pass

class mhaModel(ModelBuilder):
    """
    Decoder-only transformer. All attention/ffn/block internals are implemented here.
    """
    def __init__(self,
                 vocab_size: int,
                 max_seq_len: int,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__() # This will now correctly call nn.Module.__init__()
        assert d_model % n_heads == 0
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # transformer weights (store per-layer weights in ModuleList)
        self.qkv = nn.ModuleList([nn.Linear(d_model, 3*d_model) for _ in range(n_layers)])
        self.out_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ff_w1 = nn.ModuleList([nn.Linear(d_model, d_ff) for _ in range(n_layers)])
        self.ff_w2 = nn.ModuleList([nn.Linear(d_ff, d_model) for _ in range(n_layers)])
        self.ff_dropout = nn.Dropout(dropout)

        # final layer norm + head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # small caches are not necessary for training

    def _multi_head_attention(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C] -> returns [B, T, C]
        Implemented using linear qkv per-layer which manual split into heads and causal masking.
        """
        B, T, C = x.size()
        qkv = self.qkv[layer_idx](x)               # [B,T,3C]
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape -> [B, H, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, T, T]
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj[layer_idx](out)
        return out

    def _feed_forward(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
      # Linear -> GELU -> Dropout -> Linear
        h = F.gelu(self.ff_w1[layer_idx](x))
        h = self.ff_dropout(h)
        return self.ff_w2[layer_idx](h)

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        """
        idx: [B, T] -> logits [B, T, V]
        T must be <= max_seq_len
        """
        B, T = idx.size()
        assert T <= self.max_seq_len, f"T ({T}) > model max {self.max_seq_len}"
        tok = self.tok_emb(idx)                     # [B, T, C]
        pos = self.pos_emb(torch.arange(T, device=idx.device).unsqueeze(0))
        x = self.drop(tok + pos)
        # transformer layers
        for i in range(self.n_layers):
            x = x + self._multi_head_attention(i, self.ln1[i](x))
            x = x + self._feed_forward(i, self.ln2[i](x))
        x = self.ln_f(x)
        logits = self.head(x)                       # [B, T, V]
        return logits

    def save_weights(self, path: str):
      # wSave model weight
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, map_location=None):
      # Load model weight
        sd = torch.load(path, map_location=map_location)
        self.load_state_dict(sd)
    def clone(self):
      return copy.deepcopy(self)

class ModelFactory:
  def getModel(self, model_type: str, *args, **kwargs):
    if model_type == "mha":
      return mhaModel(*args, **kwargs)
    raise ValueError(f"Unknown model type: {model_type}")

class Trainer():
    """
    Trainer handles training loop, gradient accumulation, LR scheduling, precision training, val & checkpointing as well as loss plotting.
    """
    def __init__(self, model: mhaModel, train_dl: DataLoader, val_dl: DataLoader, cfg: Dict = CFG):
        self.observers = set()
        self.model = model.to(device)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.cfg = cfg
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        self.criterion = nn.CrossEntropyLoss()
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
        # estimate total steps
        self.total_train_steps = max(1, int((len(self.train_dl) * cfg["epochs"]) / max(1, cfg["grad_accum_steps"])))
        print("[Trainer] total_train_steps estimate:", self.total_train_steps)
    def attach(self, observer):
      self.observers.add(observer)
    def unattach(self, observer):
      self.observers.discard(observer)
    def update(self, *args):
      for observer in self.observers:
        observer.update(self, *args)
    def _lr_lambda(self, step):
        """
        Cosine decay with linear warmup
        """
        if step < self.cfg["warmup_steps"]:
            return float(step) / float(max(1, self.cfg["warmup_steps"]))
        progress = float(step - self.cfg["warmup_steps"]) / float(max(1, self.total_train_steps - self.cfg["warmup_steps"]))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def train(self):
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
        grad_accum = max(1, self.cfg["grad_accum_steps"])
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(enumerate(self.train_dl), total=len(self.train_dl), desc=f"Epoch {epoch}")
            self.optimizer.zero_grad()
            for step, batch in pbar:
                x, y = batch  # TensorDataset returns (input, label)
                x = x.to(device)
                y = y.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = self.model(x)  # [B, T, V]
                    B, T, V = logits.shape
                    loss = self.criterion(logits.view(B*T, V), y.view(B*T))
                    loss = loss / grad_accum

                scaler.scale(loss).backward()
                total_loss += loss.item() * grad_accum

                if ((step + 1) % grad_accum) == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    scheduler.step()

                    if (self.global_step % self.cfg["log_every_steps"]) == 0:
                        mean_loss = total_loss / (step + 1)
                        self.update(f"[Step {self.global_step}] approx train loss: {mean_loss:.4f} lr={self.optimizer.param_groups[0]['lr']:.2e}")
                        #print(f"[Step {self.global_step}] approx train loss: {mean_loss:.4f} lr={self.optimizer.param_groups[0]['lr']:.2e}")

                    if (self.global_step % self.cfg["max_steps_between_checkpoints"]) == 0:
                        self.save_checkpoint(self.global_step)

            avg_epoch_loss = total_loss / len(self.train_dl)
            self.train_losses.append(avg_epoch_loss)
            self.update(f"=== Epoch {epoch} complete | Train Loss: {avg_epoch_loss:.4f}")
            # print(f"=== Epoch {epoch} complete | Train Loss: {avg_epoch_loss:.4f}")

            val_loss, val_ppl = self.validate()
            self.val_losses.append(val_loss)
            # checkpoint at epoch end
            self.save_checkpoint(self.global_step, suffix=f"epoch{epoch}")

        final_path = os.path.join(self.cfg["save_dir"], "final_model.pt")
        self.model.save_weights(final_path)
        print("[Trainer] Final model saved to:", final_path)
        self.plot_losses()

    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in self.val_dl:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                logits = self.model(x)
                B, T, V = logits.shape
                loss = self.criterion(logits.view(B*T, V), y.view(B*T))
                losses.append(loss.item())
        avg = float(np.mean(losses)) if len(losses) > 0 else 0.0
        ppl = math.exp(avg) if avg < 100 else float("inf")
        print(f"[Trainer] Validation Loss: {avg:.4f} | Perplexity: {ppl:.2f}")
        return avg, ppl

    def save_checkpoint(self, step: int, suffix: Optional[str] = None):
        ckpt = {
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "cfg": self.cfg,
            "global_step": self.global_step,
        }
        name = f"ckpt_step_{step}.pt" if suffix is None else f"ckpt_step_{step}_{suffix}.pt"
        path = os.path.join(self.cfg["save_dir"], name)
        torch.save(ckpt, path)
        print("[Trainer] Checkpoint saved:", path)

    def plot_losses(self):
        plt.figure(figsize=(6,4))
        plt.plot(self.train_losses, label="train_loss")
        plt.plot(self.val_losses, label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        out = os.path.join(CFG["save_dir"], "loss_curve.png")
        plt.savefig(out)
        plt.close()
        # display inline if possible
        try:
            from PIL import Image
            img = Image.open(out)
            img.show()
        except Exception:
            pass
        print("[Trainer] Loss curve saved:", out)


class Chatbot:
    """
    Wraps a Model and Tokenizer for generation:
    - greedy
    - top-k
    - top-p (nucleus)
    """
    # Singleton Stuff
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
          cls._instance = super().__new__(cls)
        return cls._instance
    def __init__(self, model: mhaModel, tokenizer: Tokenizer):
        self.model = model.to(device)
        self.tokenizer = tokenizer

    def greedy(self, prompt: str, max_new_tokens: int = 50) -> str:
        self.model.eval()
        ids = self.tokenizer.encode(prompt)
        ids = ids[-(CFG["seq_len"] - 1):]  # keep room
        for _ in range(max_new_tokens):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = self.model(x)
            next_id = int(torch.argmax(logits[0, -1, :]).item())
            ids.append(next_id)
        return self.tokenizer.decode(ids)

    def top_k(self, prompt: str, max_new_tokens: int = 50, k: int = 50, temperature: float = 1.0) -> str:
        self.model.eval()
        ids = self.tokenizer.encode(prompt)
        ids = ids[-(CFG["seq_len"] - 1):]
        for _ in range(max_new_tokens):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = self.model(x)[0, -1, :] / max(1e-9, temperature)
            vals, idxs = torch.topk(logits, k)
            probs = F.softmax(vals, dim=-1)
            pick = torch.multinomial(probs, 1).item()
            next_id = int(idxs[pick].item())
            ids.append(next_id)
        return self.tokenizer.decode(ids)

    def top_p(self, prompt: str, max_new_tokens: int = 50, p: float = 0.9, temperature: float = 1.0) -> str:
        self.model.eval()
        ids = self.tokenizer.encode(prompt)
        ids = ids[-(CFG["seq_len"] - 1):]
        for _ in range(max_new_tokens):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = self.model(x)[0, -1, :] / max(1e-9, temperature)
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            cutoff = (cumprobs > p).nonzero(as_tuple=False)
            if cutoff.numel() > 0:
                cutoff_idx = cutoff[0].item()
                sorted_logits[cutoff_idx+1:] = float("-inf")
            probs2 = F.softmax(sorted_logits, dim=-1)
            chosen = torch.multinomial(probs2, num_samples=1).item()
            next_id = int(sorted_idx[chosen].item())
            ids.append(next_id)
        return self.tokenizer.decode(ids)
def main():
    # Dataset andPreProcessor
    dataset = Dataset()
    hf = dataset.accessData()

    pre = PreProcessor()
    raw_train = pre.loadData(hf)
    cleaned_texts = pre.cleanData()

    # Tokenizer (train + save)
    tok = Tokenizer(model_prefix=CFG["tokenizer_prefix"], vocab_size=CFG["vocab_size"])
    tok.load()

    # Tokenize train and val texts
    print("[Main] Tokenizing texts...")
    tokenized_train = tok.encode_batch(cleaned_texts)
    val_texts = [ex["text"].replace("\n", " ").strip() for ex in hf["validation"] if ex["text"].strip()]
    tokenized_val = tok.encode_batch(val_texts)

    # Build DataLoaders
    train_dl, val_dl = dataset.build_dataloaders(tokenized_train, tokenized_val)

    # Load the model
    vocab_size = tok.processor.GetPieceSize()
    PATH = r'''C:\Users\florp\Downloads\final_model_weights.pt'''
    # model = torch.load(PATH, map_location=torch.device('cpu'))
    factory = ModelFactory()
    model = factory.getModel(
        model_type="mha",
        vocab_size=vocab_size,
        max_seq_len=CFG["seq_len"] - 1,
        d_model=CFG["d_model"],
        n_layers=CFG["n_layers"],
        n_heads=CFG["n_heads"],
        d_ff=CFG["d_ff"],
        dropout=CFG["dropout"]
    )
    model.load_weights(PATH, map_location=torch.device('cpu'))
    print("[Main] Model loaded.")
    # Chatbot generation
    cbot = Chatbot(model, tok)
    prompt = st.text_input('please enter a prompt:')

    # prompts = [
    #     "The future of artificial intelligence is",
    #     "In recent years, researchers have discovered",
    #     "Economic growth will depend on"
    # ]
    # print("\nGenerated Samples: ")
    # for i, p in enumerate(prompts, 1):
    #     print(f"\nPrompt {i}: {p}")
    #     print("Greedy:\n", cbot.greedy(p, max_new_tokens=50))
    #     print("Top-K:\n", cbot.top_k(p, max_new_tokens=50, k=50, temperature=1.0))
    #     print("Top-P:\n", cbot.top_p(p, max_new_tokens=50, p=0.9, temperature=1.0))

    # # Save generations to file
    # gen = {p: {
    #     "greedy": cbot.greedy(p, max_new_tokens=50),
    #     "top_k": cbot.top_k(p, max_new_tokens=50, k=50, temperature=1.0),
    #     "top_p": cbot.top_p(p, max_new_tokens=50, p=0.9, temperature=1.0)}
    #     for p in prompts
    # }
    # with open(os.path.join(CFG["save_dir"], "generated_samples.json"), "w", encoding="utf-8") as f:
    #     json.dump(gen, f, ensure_ascii=False, indent=2)
    # print("[Main] Generated samples saved to generated_samples.json")


if __name__ == "__main__":
    main()