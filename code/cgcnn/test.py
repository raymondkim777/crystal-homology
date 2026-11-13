import torch

old_path = "out_mt_b32_mae_loss_s42/model_best_mt_b32_mae_loss_s42.pth.tar"
new_path = "checkpoints/model_best_mt_b32_mae_loss_bl_hp0_s42.pth.tar"

def test():
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    new = torch.load(new_path, map_location="cpu", weights_only=False)

    old_args = old["args"]
    new_args = new["args"]

    for key in sorted(set(old_args) | set(new_args)):
        a = old_args.get(key)
        b = new_args.get(key)

        if a != b:
            print(f"{key}:")
            print(f"  OLD: {a}")
            print(f"  NEW: {b}")


if __name__ == "__main__":
    test()