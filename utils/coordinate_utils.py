import torch

def make_coord_grid(H, W, device='cpu', Dp=2):
    """
    Create an absolute coordinate grid O of shape [Dp, H, W]:
      - O[0, i, j] = i
      - O[1, i, j] = j
      - If Dp > 2, pad extra channels with zeros.

    Args:
      H (int): height of the image
      W (int): width of the image
      device: torch device
      Dp (int): dimensionality of positional embedding

    Returns:
      Tensor O ∈ [Dp, H, W]
    """
    i_range = torch.arange(H, device=device).view(H, 1).expand(H, W)
    j_range = torch.arange(W, device=device).view(1, W).expand(H, W)
    O = torch.stack([i_range, j_range], dim=0).float()  # [2, H, W]
    if Dp > 2:
        pad = torch.zeros(Dp - 2, H, W, device=device)
        O = torch.cat([O, pad], dim=0)
    return O


# Sample usage:
if __name__ == "__main__":
    H, W = 256, 256
    Dp = 2
    device = 'cpu'
    O = make_coord_grid(H, W, device, Dp)
    print(O.shape)  # Should print: torch.Size([2, 256, 256])
    print(O[0])     # First channel (i-coordinates)
    print(O[1])     # Second channel (j-coordinates)

    # Vilusalize the coordinate grid
    import matplotlib.pyplot as plt
    plt.imshow(O[0].cpu().numpy(), cmap='gray')
    plt.title("i-coordinates")
    plt.axis('off')
    plt.show()
    plt.imshow(O[1].cpu().numpy(), cmap='gray')
    plt.title("j-coordinates")
    plt.axis('off')