import torch
import sys
import os

sys.path.insert(0, './UmedPT')
sys.path.insert(0, './UmedPT-sdk')
os.environ['MMM_LICENSE_ACCEPTED'] = 'i accept'

encoder = torch.load('encoder-1.0.4.pt/encoder.pt', map_location='cpu', weights_only=False)
encoder.eval()

from torchvision.models.swin_transformer import PatchMergingV2
features = encoder.model.wrapped_model.features
pm_layers = [(i, m) for i, m in enumerate(features) if isinstance(m, PatchMergingV2)]
print(f'Found {len(pm_layers)} PM layers')

x = torch.rand(1, 3, 224, 224)

for idx, (f_idx, pm) in enumerate(pm_layers):
    captured = {}
    def hook(m, i, o):
        captured['out'] = o
    handle = pm.register_forward_hook(hook)
    encoder(x)
    handle.remove()
    print(f"pm_index={idx}: shape={captured['out'].shape}")
