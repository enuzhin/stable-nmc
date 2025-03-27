import torch
import hydra
from omegaconf import DictConfig
import torch.nn as nn
from models import NoisyFeedForward,NoisyResNet
from utils import create_dataloaders,eval,find_latest_checkpoint



@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """Loads config using Hydra and starts training"""

    # Use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model
    model_classes = {
        "NoisyFeedForward": NoisyFeedForward,
        "NoisyResNet": NoisyResNet,
    }
    if cfg.model.type not in model_classes:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    _, test_loader = create_dataloaders(cfg.num_workers, cfg.batch_size, cfg.dataset)
    model = model_classes[cfg.model.type](
        w_max=cfg.model.w_max,
        noise_spread=cfg.model.noise_spread,
        scale=cfg.model.scale
    ).to(device)

    checkpoint_path = find_latest_checkpoint(cfg.model.type, cfg.save.path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    criterion = nn.CrossEntropyLoss()

    print("Parameters total: ",sum(p.numel() for p in model.parameters()))

    eval(
        model=model,
        device=device,
        test_loader=test_loader,
        criterion=criterion,
        enable_noise_and_clamp = cfg.model.enable_noise_and_clamp
    )

if __name__ == "__main__":
    main()