import torch.nn as nn
import timm
class Model3Classifier(nn.Module):
    """
    1. `mobileit_xxs`           -> Hybrid CNN + Transformer (input 160 atau 224)
    2. `convenxtv2`             -> Modern Pure CNN (input 160 atau 224)
    3. `deit_tiny_patch16_224`  -> Pure Vision Transformer (input 224)
    """
    def __init__(self, arch="mobilevit_xxs", num_classes=6, pretrained=True, freeze_early=False, drop_rate=0.2, drop_path_rate=0.1):
        super(Model3Classifier, self).__init__()
        self.arch = arch
        self.model = timm.create_model(
            arch,
            num_classes=num_classes,
            pretrained=pretrained,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        if freeze_early:
            early_prefixes = (
                "stem.", 'stages.0.', "stages.1.",
                "patch_embed.", "blocks.0.", "blocks.1.",
                "blocks.2.", "blocks.3.", "blocks.4.", "blocks.5.",
            )
            for name, param in self.model.named_parameters():
                if name.startswith(early_prefixes):
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)