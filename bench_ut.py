"""
LM-eval runner for checkpoints saved by research/universal_transformer/train.py.

Supports slowrun_eval_artifact_v1 single_model and logit_ensemble artifacts,
including per-checkpoint active_n_layer metadata.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import tiktoken

import lm_eval
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM


def _sdpa_attention(q, k, v, window_size, enable_gqa):
    Tq, Tk = q.size(2), k.size(2)
    window = window_size[0]
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
    if Tq == 1:
        if 0 <= window < Tk:
            start = max(0, Tk - (window + 1))
            k, v = k[:, :, start:, :], v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
    device = q.device
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx
    if 0 <= window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    q2, k2, v2 = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    y = _sdpa_attention(q2, k2, v2, window_size, q2.size(1) != k2.size(1))
    return y.transpose(1, 2)


flash_attn = SimpleNamespace(flash_attn_func=flash_attn_func)


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 50257
    n_layer: int = 20
    initial_n_layer: int = 10
    n_head: int = 16
    n_kv_head: int = 16
    n_embd: int = 2048
    window_pattern: str = "SSSL"
    dropout: float = 0.0
    stoch_depth: float = 0.0
    use_iha: bool = True
    iha_mix_v: bool = True
    logit_cap: float = 10.0


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        y = F.rms_norm(x, (x.size(-1),), self.weight.to(dtype=x.dtype))
        return y + self.bias.to(dtype=x.dtype)


def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


class SharedCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.use_iha = config.use_iha
        if self.use_iha:
            self.q_mix = nn.Parameter(torch.zeros(self.n_head, self.n_head))
            self.k_mix = nn.Parameter(torch.zeros(self.n_kv_head, self.n_kv_head))
            self.iha_mix_v = config.iha_mix_v
            if self.iha_mix_v:
                self.v_mix = nn.Parameter(torch.zeros(self.n_kv_head, self.n_kv_head))

    def _fuse_mix(self, weight, mix, num_heads):
        d = self.head_dim
        return (mix @ weight.view(num_heads, d, -1).flatten(1)).view_as(weight)

    def forward(self, x, ve, cos_sin, window_size, q_norm, k_norm, ve_gate=None, attn_gate=None):
        B, T, _ = x.size()
        if self.use_iha:
            q = F.linear(x, self._fuse_mix(self.c_q.weight, self.q_mix, self.n_head)).view(B, T, self.n_head, self.head_dim)
            k = F.linear(x, self._fuse_mix(self.c_k.weight, self.k_mix, self.n_kv_head)).view(B, T, self.n_kv_head, self.head_dim)
            if self.iha_mix_v:
                v = F.linear(x, self._fuse_mix(self.c_v.weight, self.v_mix, self.n_kv_head)).view(B, T, self.n_kv_head, self.head_dim)
            else:
                v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        else:
            q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            assert ve_gate is not None
            v = v + (2 * torch.sigmoid(ve_gate)).unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = q_norm(q), k_norm(k)
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        if attn_gate is not None:
            y = y * torch.sigmoid(attn_gate).unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
        return self.resid_dropout(self.c_proj(y))


class SharedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 8 * config.n_embd
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.resid_dropout(self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x)))


class LayerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn_norms = nn.ModuleList([RMSNorm(config.n_embd) for _ in range(2)])
        self.mlp_norm = RMSNorm(config.n_embd)
        head_dim = config.n_embd // config.n_head
        self.q_norms = nn.ModuleList([RMSNorm(head_dim) for _ in range(2)])
        self.k_norms = nn.ModuleList([RMSNorm(head_dim) for _ in range(2)])
        self.ve_gate_channels = 32
        self.attn_gate_channels = 12
        has_value_embed = has_ve(layer_idx, config.n_layer)
        self.ve_gates = nn.ModuleList([
            nn.Linear(self.ve_gate_channels, config.n_kv_head, bias=False) if has_value_embed else nn.Identity()
            for _ in range(2)
        ])
        self.attn_gates = nn.ModuleList([
            nn.Linear(self.attn_gate_channels, config.n_head, bias=False) for _ in range(2)
        ])
        self.drop_prob = config.stoch_depth * (layer_idx / max(config.n_layer - 1, 1))


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        assert config.n_layer % 2 == 0
        assert config.initial_n_layer % 2 == 0
        self.max_n_layer = config.n_layer
        self.max_encoder_layers = config.n_layer // 2
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.encoder_attns = nn.ModuleList([SharedCausalSelfAttention(config) for _ in range(2)])
        self.decoder_attns = nn.ModuleList([SharedCausalSelfAttention(config) for _ in range(2)])
        self.encoder_mlp = SharedMLP(config)
        self.decoder_mlp = SharedMLP(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab, config.n_embd),
            "h": nn.ModuleList([LayerBlock(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.ve_projs = nn.ModuleDict({
            str(i): nn.Linear(config.n_embd, kv_dim, bias=False)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.skip_weights = nn.Parameter(torch.ones(self.max_encoder_layers))
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.set_active_layers(config.initial_n_layer)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        sizes = [{"L": (long_w, 0), "S": (short_w, 0)}[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)
        return sizes

    def refresh_rotary(self):
        head_dim = self.config.n_embd // self.config.n_head
        self.cos, self.sin = self._precompute_rotary(self.rotary_seq_len, head_dim)

    def set_active_layers(self, n_layer):
        assert n_layer % 2 == 0
        assert 0 < n_layer <= self.max_n_layer
        self.active_n_layer = n_layer
        self.active_encoder_layers = n_layer // 2
        self.active_decoder_start = self.max_n_layer - self.active_encoder_layers

    def _run_layer(self, x, x0, cos_sin, layer_idx, shared_attns, shared_mlp):
        block = self.transformer.h[layer_idx]
        x = self.resid_lambdas[layer_idx] * x + self.x0_lambdas[layer_idx] * x0
        ve = self.ve_projs[str(layer_idx)](x0) if str(layer_idx) in self.ve_projs else None
        for attn, attn_norm, q_norm, k_norm, ve_gate, attn_gate in zip(
            shared_attns, block.attn_norms, block.q_norms, block.k_norms, block.ve_gates, block.attn_gates
        ):
            attn_input = attn_norm(x)
            ve_gate_out = ve_gate(attn_input[..., :block.ve_gate_channels]) if isinstance(ve_gate, nn.Linear) else None
            attn_gate_out = attn_gate(attn_input[..., :block.attn_gate_channels])
            x = x + attn(attn_input, ve, cos_sin, self.window_sizes[layer_idx], q_norm, k_norm,
                         ve_gate=ve_gate_out, attn_gate=attn_gate_out)
        x = x + shared_mlp(block.mlp_norm(x))
        return x

    def _run_decoder_layers(self, x, x0, cos_sin, encoder_outputs, start, end):
        for i in range(start, end):
            j = self.max_n_layer - 1 - i
            if 0 <= j < len(encoder_outputs):
                x = x + self.skip_weights[i - self.max_encoder_layers] * encoder_outputs[j]
            x = self._run_layer(x, x0, cos_sin, i, self.decoder_attns, self.decoder_mlp)
        return x

    def forward(self, idx, targets=None):
        _, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = rms_norm(self.transformer.wte(idx))
        x0 = x
        encoder_outputs = []
        for i in range(self.active_encoder_layers):
            x = self._run_layer(x, x0, cos_sin, i, self.encoder_attns, self.encoder_mlp)
            encoder_outputs.append(x)
        x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs, self.active_decoder_start, self.max_n_layer)
        x = rms_norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        if self.config.logit_cap > 0:
            logits = self.config.logit_cap * torch.tanh(logits / self.config.logit_cap)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_tokens=None, max_seq_length=None):
        stop_tokens = set(stop_tokens or [])
        max_seq_length = max_seq_length or self.config.sequence_len
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= max_seq_length else idx[:, -max_seq_length:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx.size(0) == 1 and idx_next[0, 0].item() in stop_tokens:
                break
        return idx


def get_dtype(dtype_str):
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype_str, torch.bfloat16)


def clean_state_dict(state_dict):
    cleaned = {}
    meta = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            meta[k] = v
            continue
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        cleaned[k] = v
    return cleaned, meta


def extract_artifact(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get("format") == "slowrun_eval_artifact_v1":
        return ckpt
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return {"artifact_type": "single_model", "state_dict": ckpt["state_dict"], "model_config": ckpt.get("model_config", {}), "active_n_layer": ckpt.get("active_n_layer")}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return {"artifact_type": "single_model", "state_dict": ckpt["model_state_dict"], "model_config": ckpt.get("model_config", {}), "active_n_layer": ckpt.get("active_n_layer")}
    if isinstance(ckpt, dict):
        return {"artifact_type": "single_model", "state_dict": ckpt, "model_config": {}}
    raise ValueError(f"Unknown checkpoint format: {checkpoint_path}")


def infer_config(state_dict, model_config, logit_cap):
    wte = state_dict["transformer.wte.weight"]
    padded_vocab, n_embd = wte.shape
    layer_idxs = {int(m.group(1)) for k in state_dict for m in [re.match(r"transformer\.h\.(\d+)\.", k)] if m}
    n_layer = int(model_config.get("n_layer") or (max(layer_idxs) + 1))
    n_head = int(model_config.get("n_head") or state_dict["encoder_attns.0.c_q.weight"].shape[0] // (n_embd // int(model_config.get("n_head", 1))))
    if "n_head" not in model_config:
        n_head = state_dict["transformer.h.0.attn_gates.0.weight"].shape[0]
    head_dim = n_embd // n_head
    c_k_out = state_dict["encoder_attns.0.c_k.weight"].shape[0]
    n_kv_head = int(model_config.get("n_kv_head") or (c_k_out // head_dim))
    return GPTConfig(
        sequence_len=int(model_config.get("sequence_len", 2048)),
        vocab_size=int(model_config.get("vocab_size") or min(50257, padded_vocab)),
        n_layer=n_layer,
        initial_n_layer=int(model_config.get("initial_n_layer", n_layer)),
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=int(model_config.get("n_embd", n_embd)),
        window_pattern=str(model_config.get("window_pattern", "SSSL")),
        dropout=0.0,
        stoch_depth=0.0,
        use_iha=bool(model_config.get("use_iha", any(k.endswith(".q_mix") for k in state_dict))),
        iha_mix_v=bool(model_config.get("iha_mix_v", any(k.endswith(".v_mix") for k in state_dict))),
        logit_cap=logit_cap,
    )


def load_single_model(state_dict_raw, model_config, active_n_layer, device, dtype, logit_cap):
    state_dict, meta = clean_state_dict(state_dict_raw)
    active_n_layer = active_n_layer or meta.get("active_n_layer")
    config = infer_config(state_dict, model_config or {}, logit_cap)
    model = GPT(config)
    if active_n_layer is not None:
        model.set_active_layers(int(active_n_layer))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing if k not in {"cos", "sin"}]
    if real_missing:
        raise RuntimeError(f"Missing UT checkpoint keys: {real_missing[:20]}{' ...' if len(real_missing) > 20 else ''}")
    if unexpected:
        raise RuntimeError(f"Unexpected UT checkpoint keys: {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")
    model = model.to(device).to(dtype)
    model.refresh_rotary()
    model.eval()
    return model


def load_models_from_checkpoint(checkpoint_path, device, dtype, logit_cap):
    artifact = extract_artifact(checkpoint_path)
    model_config = artifact.get("model_config") or {}
    if artifact.get("artifact_type") == "logit_ensemble":
        weights = artifact.get("ensemble_weights") or [1.0] * len(artifact["checkpoints"])
        total = float(sum(weights))
        weights = [float(w) / total for w in weights]
        models = [
            load_single_model(item["state_dict"], model_config, item.get("active_n_layer"), device, dtype, logit_cap)
            for item in artifact["checkpoints"]
        ]
        return models, weights
    model = load_single_model(artifact["state_dict"], model_config, artifact.get("active_n_layer"), device, dtype, logit_cap)
    return [model], [1.0]


class DummyAccelerator:
    def __init__(self, world_size=1):
        self.world_size = world_size

    def gather(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.unsqueeze(0).expand(self.world_size, *obj.shape)
        return [obj] * self.world_size

    def wait_for_everyone(self):
        pass

    def gather_for_metrics(self, tensor):
        return self.gather(tensor)

    @property
    def is_main_process(self):
        return True

    @property
    def is_local_main_process(self):
        return True

    @property
    def num_processes(self):
        return 1


class SlowrunUTLM(LM):
    def __init__(self, checkpoint, rank=0, world_size=1, device="cuda:0", batch_size=1,
                 max_length=2048, dtype="bfloat16", logit_cap=10.0):
        super().__init__()
        self._rank = rank
        self._world_size = world_size
        self._device = torch.device(device)
        self._batch_size = int(batch_size)
        self._max_length = int(max_length)
        self._dtype = get_dtype(dtype)
        self.accelerator = DummyAccelerator(world_size)
        self.models, self.weights = load_models_from_checkpoint(checkpoint, self._device, self._dtype, logit_cap)
        self._enc = tiktoken.get_encoding("gpt2")
        self._eos_token_id = self._enc._special_tokens["<|endoftext|>"]
        print(f"[Rank {rank}/{world_size}] Loaded {len(self.models)} UT model(s) from {checkpoint}")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def eot_token_id(self):
        return self._eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, s: str, add_special_tokens: bool = False) -> List[int]:
        return self._enc.encode_ordinary(s)

    def tok_decode(self, tokens: List[int]) -> str:
        return self._enc.decode(tokens)

    @torch.no_grad()
    def _log_probs(self, input_ids):
        probs = None
        with torch.amp.autocast(device_type="cuda", dtype=self._dtype, enabled=(self._device.type == "cuda" and self._dtype != torch.float32)):
            for model, weight in zip(self.models, self.weights):
                p = F.softmax(model(input_ids).float(), dim=-1) * weight
                probs = p if probs is None else probs + p
        return torch.log(probs.clamp_min(1e-30))

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        results = []
        for req in tqdm(requests, desc=f"[r{self._rank}] loglikelihood"):
            context, continuation = req.args
            ctx_enc = self.tok_encode(context)
            cont_enc = self.tok_encode(continuation)
            full = ctx_enc + cont_enc
            if len(full) > self._max_length:
                full = full[-self._max_length:]
                ctx_len = max(0, len(full) - len(cont_enc))
            else:
                ctx_len = len(ctx_enc)
            input_ids = torch.tensor([full], device=self._device, dtype=torch.long)
            log_probs = self._log_probs(input_ids)
            total, greedy = 0.0, True
            for i, tok in enumerate(cont_enc):
                pos = ctx_len + i - 1
                if pos < 0 or pos >= log_probs.size(1):
                    continue
                total += log_probs[0, pos, tok].item()
                if log_probs[0, pos].argmax().item() != tok:
                    greedy = False
            results.append((total, greedy))
        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        results = []
        for req in tqdm(requests, desc=f"[r{self._rank}] loglikelihood_rolling"):
            toks = self.tok_encode(req.args[0])
            if not toks:
                results.append(0.0)
                continue
            toks = toks[-self._max_length:]
            input_ids = torch.tensor([toks], device=self._device, dtype=torch.long)
            log_probs = self._log_probs(input_ids)
            results.append(sum(log_probs[0, i - 1, toks[i]].item() for i in range(1, len(toks))))
        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []
        for req in tqdm(requests, desc=f"[r{self._rank}] generate_until"):
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            ctx = self.tok_encode(context)
            if len(ctx) > self._max_length - max_gen:
                ctx = ctx[-(self._max_length - max_gen):]
            input_ids = torch.tensor([ctx], device=self._device, dtype=torch.long)
            stop_ids = {self._eos_token_id}
            for s in until:
                t = self.tok_encode(s)
                if t:
                    stop_ids.add(t[0])
            out = self.models[0].generate(input_ids, max_gen, temperature=1.0, top_k=1,
                                          stop_tokens=stop_ids, max_seq_length=self._max_length)
            gen_text = self.tok_decode(out[0, len(ctx):].tolist())
            for s in until:
                if s in gen_text:
                    gen_text = gen_text.split(s)[0]
            results.append(gen_text)
        return results


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def combine_results(paths, output):
    loaded = []
    for path in sorted(paths):
        try:
            with open(path) as f:
                loaded.append(json.load(f))
            print(f"Loaded: {path}")
        except Exception as exc:
            print(f"Warning: failed to load {path}: {exc}")
    if not loaded:
        raise ValueError("No valid result files")
    combined = {"results": {}}
    for key in ("config", "configs", "versions", "git_hash", "date"):
        if key in loaded[0]:
            combined[key] = loaded[0][key]
    tasks = sorted({task for result in loaded for task in result.get("results", {})})
    for task in tasks:
        rows = [result["results"][task] for result in loaded if task in result.get("results", {})]
        out = {}
        sample_counts = [row.get("n_samples") for row in rows]
        if all(isinstance(n, (int, float)) and n > 0 for n in sample_counts):
            total = sum(sample_counts)
            weights = [n / total for n in sample_counts]
            out["n_samples"] = total
        else:
            weights = [1.0 / len(rows)] * len(rows)
        for key in sorted({k for row in rows for k in row}):
            if key == "alias":
                out[key] = rows[0][key]
            elif key != "n_samples" and not key.endswith("_stderr"):
                vals = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
                if len(vals) == len(weights):
                    out[key] = sum(v * w for v, w in zip(vals, weights))
                    if len(vals) > 1:
                        mean = sum(vals) / len(vals)
                        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                        out[f"{key}_stderr"] = math.sqrt(var) / math.sqrt(len(vals))
        combined["results"][task] = out
    with open(output, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved -> {output}")
    for task, row in combined["results"].items():
        parts = [f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float) and (k.startswith("acc,") or k.startswith("acc_norm,"))]
        print(f"  {task}: {', '.join(parts) if parts else '(no acc metrics)'}")


def eval_main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world_size", type=int, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--tasks", type=str, required=True)
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--logit_cap", type=float, default=10.0)
    args = p.parse_args(argv)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    model = SlowrunUTLM(
        checkpoint=args.checkpoint,
        rank=args.rank,
        world_size=args.world_size,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
        logit_cap=args.logit_cap,
    )
    os.environ["LMEVAL_MANUAL_SHARDING"] = "1"
    results = lm_eval.simple_evaluate(model=model, tasks=tasks, num_fewshot=args.num_fewshot, limit=args.limit)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"[Rank {args.rank}] Saved -> {args.output}")


def combine_main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    paths = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    combine_results(paths, args.output)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "combine":
        combine_main(sys.argv[2:])
    else:
        eval_main(sys.argv[1:])
