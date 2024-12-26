import torch
from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation
import torch.nn.functional as F

class AdvancedSegFormer(nn.Module):
    def __init__(self, num_classes, num_input_channels=14, model_name="nvidia/segformer-b0-finetuned-ade-512-512"):
        super(AdvancedSegFormer, self).__init__()
        # Load SegFormer configuration
        config = SegformerConfig.from_pretrained(model_name)
        config.num_labels = num_classes

        # Modify patch embedding to handle custom input channels
        self.model = SegformerForSemanticSegmentation(config)
        self.model.segformer.encoder.patch_embeddings[0] = nn.Conv2d(
            in_channels=num_input_channels, 
            out_channels=config.hidden_sizes[0], 
            kernel_size=7, 
            stride=4, 
            padding=3
        )

    def forward(self, x):
        # Forward pass through SegFormer
        outputs = self.model(pixel_values=x)
        logits = outputs.logits

        # Upsample to match input resolution
        upsampled_logits = F.interpolate(logits, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        return upsampled_logits

def build_advanced_segformer(num_classes, num_input_channels):
    return AdvancedSegFormer(num_classes=num_classes, num_input_channels=num_input_channels)
