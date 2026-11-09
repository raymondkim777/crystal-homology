import torch

checkpoint = torch.load(
    "pre-trained/efermi.pth.tar",
    map_location="cpu",
    weights_only=True,
)

normalizer = checkpoint["normalizer"]

print("Mean:", normalizer["mean"])
print("Std:", normalizer["std"])