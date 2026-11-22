"""Deterministic shore-foam rendering primitives."""

from foam.shore import (
    fade_with_depth,
    foam_at_frame,
    foam_batch,
    foam_density,
    foam_mask,
    shore_distance,
    validate_params,
    wave_phase,
)

__all__ = [
    "fade_with_depth",
    "foam_at_frame",
    "foam_batch",
    "foam_density",
    "foam_mask",
    "shore_distance",
    "validate_params",
    "wave_phase",
]
