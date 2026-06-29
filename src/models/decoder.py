"""
NLLB-200 decoder with LoRA. Shared across M1/M2/M3.

The frozen WavLM encoder + bridge produce `memory` (B, T, 1024) and
`memory_mask` (B, T). This module feeds them into NLLB's decoder via
cross-attention by wrapping them as `encoder_outputs` / `encoder_attention_mask`
-- NLLB's own text encoder is bypassed entirely.

Trainable parameters: only the LoRA adapters on the decoder's cross/self
attention projections (q_proj, v_proj). The base NLLB weights stay frozen.

Key version note: use tokenizer.convert_tokens_to_ids("zho_Hans") for the
forced BOS token. tokenizer.lang_code_to_id was deprecated and REMOVED in
recent transformers -- it raises AttributeError on 4.44.x.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from peft import LoraConfig, get_peft_model


class NLLBDecoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        tgt_lang_code: str = "zho_Hans",
        src_lang_code: str = "eng_Latn",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules=("q_proj", "v_proj"),
    ):
        super().__init__()
        self.tgt_lang_code = tgt_lang_code

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, src_lang=src_lang_code, tgt_lang=tgt_lang_code
        )

        base = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # Freeze all base weights; LoRA injects the only trainable params.
        for p in base.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_target_modules),
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        self.model = get_peft_model(base, lora_cfg)

        # Resolve the forced-BOS id once. convert_tokens_to_ids is the
        # current, stable API (lang_code_to_id was removed).
        self.forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang_code)
        if self.forced_bos_token_id is None or self.forced_bos_token_id == self.tokenizer.unk_token_id:
            raise ValueError(
                f"Could not resolve target lang code {tgt_lang_code!r} to a token id."
            )

    @property
    def d_model(self) -> int:
        return self.model.config.d_model  # 1024 for distilled-600M

    def _wrap_memory(self, memory: torch.Tensor) -> BaseModelOutput:
        """Present bridge features to NLLB as if they were encoder outputs."""
        return BaseModelOutput(last_hidden_state=memory)

    def forward(self, memory, memory_mask, labels):
        """Training forward: cross-entropy against target tokens.

        Args:
            memory:      (B, T, d_model) from the bridge.
            memory_mask: (B, T) 1=real, 0=padding. Used as encoder_attention_mask.
            labels:      (B, L) target token ids; pad positions set to -100 so
                         they are ignored by the loss (see tokenize_targets).
        Returns:
            ModelOutput with .loss and .logits.
        """
        return self.model(
            encoder_outputs=self._wrap_memory(memory),
            attention_mask=memory_mask.long(),   # encoder (cross-attn) mask
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, memory, memory_mask, max_length: int = 256, num_beams: int = 5):
        """Autoregressive generation for BLEU eval. Forces Chinese BOS."""
        generated = self.model.generate(
            encoder_outputs=self._wrap_memory(memory),
            attention_mask=memory_mask.long(),
            forced_bos_token_id=self.forced_bos_token_id,
            max_length=max_length,
            num_beams=num_beams,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def tokenize_targets(self, target_texts, device=None):
        """Tokenize target strings into labels with pad -> -100 masking.

        With tgt_lang set on the tokenizer, text_target prepends the
        zho_Hans language token, matching what the decoder must learn to emit.
        """
        labels = self.tokenizer(
            text_target=list(target_texts),
            return_tensors="pt",
            padding=True,
        ).input_ids
        # Ignore pad tokens in the loss.
        labels[labels == self.tokenizer.pad_token_id] = -100
        if device is not None:
            labels = labels.to(device)
        return labels

    def print_trainable(self):
        self.model.print_trainable_parameters()
