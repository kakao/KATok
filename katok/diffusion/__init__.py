from .cascaded import CascadedGenerator
from .flow_matching import decoupled_ode_sample, sample_ode
from .generator import Generator, LatentStats, build_sit
from .joint import JointGenerator
from .mask_prior import coords_to_grid_mask, grid_center_coords, mask_to_positions
from .sit import SiT, SiT_XL_2

__all__ = [
    "CascadedGenerator",
    "Generator",
    "JointGenerator",
    "LatentStats",
    "SiT",
    "SiT_XL_2",
    "build_sit",
    "coords_to_grid_mask",
    "decoupled_ode_sample",
    "grid_center_coords",
    "mask_to_positions",
    "sample_ode",
]


def load_generator(weights, config=None, stats=None, device="cpu"):
    """Load whichever generator variant a config names.

    Reads ``variant`` from the config (``cascaded`` or ``joint``) and dispatches to
    the matching class.
    """
    import os

    import yaml

    if config is None:
        cfg_dir = weights if os.path.isdir(weights) else os.path.dirname(os.path.abspath(weights))
        config = os.path.join(cfg_dir, "config.yaml")
    if isinstance(config, str):
        with open(config) as f:
            config = yaml.safe_load(f)

    variant = config.get("variant")
    cls = {"cascaded": CascadedGenerator, "joint": JointGenerator}.get(variant)
    if cls is None:
        raise ValueError(f"unknown variant {variant!r}; expected 'cascaded' or 'joint'")
    return cls.from_pretrained(weights, config=config, stats=stats, device=device)
