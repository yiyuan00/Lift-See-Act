"""ViT-B visual encoder with LoRA and the LSA motion-prediction head."""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import ViTModel


class ViTFeatureExtractor(nn.Module):
    """Return the attention maps and hierarchical ViT representations."""

    def __init__(self, vit: nn.Module) -> None:
        super().__init__()
        self.vit = vit

    def forward(self, images: torch.Tensor):
        outputs = self.vit(images)
        return (
            outputs.attentions[:6],
            outputs.last_hidden_state,
            outputs.attentions[:6],
            outputs.hidden_states,
        )


class RobotTransformerAR(nn.Module):
    """Frozen ImageNet ViT-B with LoRA and a linear three-step motion head."""

    def __init__(
        self,
        pre_len: int = 3,
        backbone: str = "google/vit-base-patch16-224-in21k",
    ) -> None:
        super().__init__()
        self.hidden_size = 768
        self.vit = self._build_vit(backbone)
        self.embed_ln = nn.LayerNorm(self.hidden_size)
        self.predict_action = nn.Linear(self.hidden_size, 3 * pre_len)

    @staticmethod
    def _build_vit(backbone: str) -> ViTFeatureExtractor:
        vit = ViTModel.from_pretrained(
            backbone,
            output_hidden_states=True,
            output_attentions=True,
        )
        for parameter in vit.parameters():
            parameter.requires_grad = False
        lora = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        return ViTFeatureExtractor(get_peft_model(vit, lora))

    def forward(self, object_pc: torch.Tensor):
        embedding = self.embed_ln(object_pc)
        return {"action": self.predict_action(embedding)}, object_pc
