"""Construction helpers for PreJEPA — keeps save/load_pretrained agnostic of model wiring."""

from collections import OrderedDict

import stable_pretraining as spt
from omegaconf import OmegaConf
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForImageClassification,
)

from .module import CausalPredictor, Embedder
from .prejepa import PreJEPA

# fmt: off
ENCODER_CONFIGS = {
    'resnet': {
        'prefix': 'microsoft/resnet-',
        'model_class': AutoModelForImageClassification,
        'embedding_attr': lambda m: m.config.hidden_sizes[-1],
        'post_init': lambda m: setattr(m.classifier, '1', nn.LayerNorm(m.config.hidden_sizes[-1])),
        'interpolate_pos_encoding': False,
    },
    'vit':    {'prefix': 'google/vit-'},
    'dino':   {'prefix': 'facebook/dino-'},
    'dinov2':  {'prefix': 'facebook/dinov2-'},
    'dinov3':  {'prefix': 'facebook/dinov3-'},
    'webssl':  {'prefix': 'facebook/webssl-'},
    'mae':    {'prefix': 'facebook/vit-mae-'},
    'ijepa':  {'prefix': 'facebook/ijepa'},
    'vjepa2':  {'prefix': 'facebook/vjepa2-vit'},
    'siglip2': {'prefix': 'google/siglip2-'},
}
# fmt: on


def get_encoder(cfg):
    """Load a pretrained vision encoder; return (backbone, embed_dim, num_patches, interp_pos_enc)."""
    encoder_cfg = next(
        (
            c
            for c in ENCODER_CONFIGS.values()
            if cfg.backbone.name.startswith(c['prefix'])
        ),
        None,
    )
    if encoder_cfg is None:
        raise ValueError(f'Unsupported backbone: {cfg.backbone.name}')

    backbone = encoder_cfg.get('model_class', AutoModel).from_pretrained(
        cfg.backbone.name
    )
    if hasattr(backbone, 'vision_model'):
        backbone = backbone.vision_model
    if 'post_init' in encoder_cfg:
        encoder_cfg['post_init'](backbone)

    embed_dim = encoder_cfg.get(
        'embedding_attr', lambda m: m.config.hidden_size
    )(backbone)
    is_cnn = cfg.backbone.name.startswith('microsoft/resnet-')
    num_patches = 1 if is_cnn else (cfg.image_size // cfg.patch_size) ** 2
    interp_pos_enc = encoder_cfg.get('interpolate_pos_encoding', True)
    return backbone, embed_dim, num_patches, interp_pos_enc


def build_prejepa(cfg):
    """Construct a PreJEPA model from a training-style cfg (dict or OmegaConf).

    This is the canonical builder used by both `scripts/train/prejepa.py` and
    `load_pretrained` (via `_target_` in saved `config.json`).
    """
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)

    encoder, embed_dim, num_patches, interp_pos_enc = get_encoder(cfg)
    embed_dim += sum(cfg.wm.get('encoding', {}).values())

    if cfg.backbone.get('is_video_encoder', False):
        num_patches += num_patches * (cfg.n_steps // 4)

    predictor_kwargs = {k: v for k, v in cfg.predictor.items() if k != 'size'}
    predictor = CausalPredictor(
        num_patches=num_patches,
        num_frames=cfg.wm.history_size,
        dim=embed_dim,
        **predictor_kwargs,
    )

    extra_encoders = nn.ModuleDict(
        OrderedDict(
            (
                key,
                Embedder(in_chans=cfg.extra_dims[key], emb_dim=emb_dim),
            )
            for key, emb_dim in cfg.wm.get('encoding', {}).items()
        )
    )

    return PreJEPA(
        encoder=spt.backbone.EvalOnly(encoder),
        predictor=predictor,
        extra_encoders=extra_encoders,
        history_size=cfg.wm.history_size,
        num_pred=cfg.wm.num_preds,
        interpolate_pos_encoding=interp_pos_enc,
    )
