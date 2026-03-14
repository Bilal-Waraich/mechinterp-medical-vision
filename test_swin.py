import torch
from torchvision.models import swin_v2_b
model = swin_v2_b()
from torchvision.models.swin_transformer import PatchMergingV2
pm_layers = [(i, m) for i, m in enumerate(model.features) if isinstance(m, PatchMergingV2)]
print("Patch merging layers:")
for i, m in pm_layers:
    print(i, m)
print(f"Total PM layers: {len(pm_layers)}")
