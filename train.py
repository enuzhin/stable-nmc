import hydra
from omegaconf import DictConfig
from models import NoisyFeedForward,NoisyResNet

from utils import train

#config_name = res_net or feed_forward
@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """Loads config using Hydra and starts training"""


    # Select model
    model_classes = {
        "NoisyFeedForward": NoisyFeedForward,
        "NoisyResNet": NoisyResNet,
    }
    if cfg.model.type not in model_classes:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    model = model_classes[cfg.model.type](
        w_max=cfg.model.w_max,
        noise_spread=cfg.model.noise_spread,
        scale=cfg.model.scale
    )

    # Run training
    train(
        model=model,
        lr=cfg.model.optimizer.lr,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        cfg = cfg,
        enable_noise_and_clamp = cfg.model.enable_noise_and_clamp
    )

if __name__ == "__main__":
    main()