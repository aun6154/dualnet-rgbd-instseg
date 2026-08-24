import torch, sys
path = sys.argv[1]
try:
    ckpt = torch.load(path, map_location="cpu")
    print(f"OK: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
except Exception as e:
    print(f"손상됨: {e}")
