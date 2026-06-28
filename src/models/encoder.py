"""Frozen WavLM encoder. Shared, unchanged, across M1/M2/M3."""
import torch
import torch.nn as nn
from transformers import WavLMModel, AutoFeatureExtractor


class FrozenWavLMEncoder(nn.Module):
    def __init__(self, model_name="microsoft/wavlm-large", layer=21):
        super().__init__()
        self.layer = layer
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.hidden_dim = self.model.config.hidden_size  # 1024 for large

    @torch.no_grad()
    def forward(self, audio_arrays):
        inputs = self.feature_extractor(
            audio_arrays, sampling_rate=16000,
            return_tensors="pt", padding=True,
        )
        device = next(self.model.parameters()).device
        input_values = inputs.input_values.to(device)
        attention_mask = (inputs.attention_mask.to(device)
                          if "attention_mask" in inputs else None)

        outputs = self.model(input_values, attention_mask=attention_mask,
                             output_hidden_states=True)
        features = outputs.hidden_states[self.layer]   # (B, T, 1024)

        if attention_mask is not None:
            frame_mask = self.model._get_feature_vector_attention_mask(
                features.shape[1], attention_mask)
        else:
            frame_mask = torch.ones(features.shape[:2], dtype=torch.long,
                                    device=device)
        return features, frame_mask
