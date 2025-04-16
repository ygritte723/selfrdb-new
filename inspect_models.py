# inspect_models.py
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
import torch
from torchinfo import summary
from torchviz import make_dot

from diffusion import DiffusionBridge
from backbones.ncsnpp import NCSNpp
from backbones.discriminator import Discriminator_large

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 1) Example init params (tweak these to match your config.yaml!)
    generator_params = {
        "self_recursion": True,
        "z_emb_dim": 256,
        "ch_mult": [1,1,2,2,4,4],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.0,
        "resamp_with_conv": True,
        "image_size": 256,
        "conditional": True,
        "fir": True,
        "fir_kernel": [1,3,3,1],
        "skip_rescale": True,
        "resblock_type": "biggan",
        "progressive": "none",
        "progressive_input": "residual",
        "embedding_type": "positional",
        "combine_method": "sum",
        "fourier_scale": 16,
        "nf": 64,
        "num_channels": 2,
        "nz": 100,
        "n_mlp": 3,
        "centered": True,
        "not_use_tanh": False,
    }
    discriminator_params = {
        "nc": 1,
        "ngf": 32,
        "t_emb_dim": 256,
    }
    diffusion_params = {
        "n_steps": 10,
        "gamma": 1.0,
        "beta_start": 0.1,
        "beta_end": 3.0,
        "n_recursions": 2,
        "consistency_threshold": 0.01,
    }

    # 2) Instantiate
    print("=== Generator (NCSNpp) ===")
    gen = NCSNpp(**{**generator_params, "self_recursion": False}).to(device)    
    print(gen, "\n")

    print("=== Discriminator (large) ===")
    disc = Discriminator_large(**discriminator_params).to(device)
    print(disc, "\n")
    # patch it so start_conv now expects 2*nc input channels:
    disc.start_conv = torch.nn.Conv2d(
        in_channels=discriminator_params["nc"] * 2,
        out_channels=discriminator_params["ngf"] * 2,
        kernel_size=1,
        padding=0,
    ).to(device)

    print("=== DiffusionBridge ===")
    diff = DiffusionBridge(**diffusion_params).to(device)
    print(diff, "\n")

    # 3) torchinfo summaries
    print("=== torchinfo.summary: Generator ===")
    summary(gen,
           input_data=[torch.randn(1, generator_params["num_channels"],
                                     generator_params["image_size"],
                                     generator_params["image_size"],
                                     device=device),
                         torch.randint(1, diffusion_params["n_steps"]+1, (1,), device=device)],
            depth=3)

    print("\n=== torchinfo.summary: Discriminator ===")
    summary(disc,
            input_data=[torch.randn(1, discriminator_params["nc"],  # x_{t-1}
                                     generator_params["image_size"],
                                     generator_params["image_size"],
                                     device=device),
                        torch.randn(1, discriminator_params["nc"],  # x_t
                                     generator_params["image_size"],
                                     generator_params["image_size"],
                                     device=device),
                        torch.randint(1, diffusion_params["n_steps"]+1, (1,), device=device)],
            depth=3)

    # 4) (Optional) visualize discriminator graph
    x_tm1 = torch.randn(1, discriminator_params["nc"],
                        generator_params["image_size"],
                        generator_params["image_size"],
                        device=device)
    x_t   = torch.randn_like(x_tm1)
    t     = torch.randint(1, diffusion_params["n_steps"]+1, (1,), device=device)
    out = disc(x_tm1, x_t, t)
    dot = make_dot(out, params=dict(disc.named_parameters()))
    dot.format = "png"
    dot.render("discriminator_graph", cleanup=True)
    print("\nWrote `discriminator_graph.png`")

if __name__ == "__main__":
    main()