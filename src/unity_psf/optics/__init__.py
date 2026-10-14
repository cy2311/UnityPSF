"""Optics primitives shared by peak, training, and post-processing."""

from .profiles import (
    ASTIGMATISM_660NM_ANCHOR_PROFILE,
    AstigmatismAnchorProfile,
    CIRCULAR_2D_50NM_ANCHOR_PROFILE,
    CircularPSFAnchorProfile,
    resolve_circular_psf_anchor_profile,
    resolve_astigmatism_anchor_profile,
)

from .nat_field import (
    NATCoeffStack,
    NATComponent,
    NATFieldConfig,
    NATGamma,
    build_named_nat_config,
    default_order1_config,
    evaluate_zernike_coefficients_torch,
    evaluate_zernike_from_roi_positions_torch,
    full_roi_coeff_stack_torch,
    get_fov_coordinates_torch,
    order1_13_config,
    order1_21_config,
)
from .renderer import GaussianPSFRenderer, GaussianPSFRendererConfig
from .zmap_renderer import ZmapCoefficientRenderer, ZmapLUTRenderer, add_fixed_astigmatism_anchor
from .vector_psf import VectorPSFParams, build_vector_psf_context, render_vector_psf_bank, render_vector_psf_stack

__all__ = [
    "ASTIGMATISM_660NM_ANCHOR_PROFILE",
    "AstigmatismAnchorProfile",
    "CIRCULAR_2D_50NM_ANCHOR_PROFILE",
    "CircularPSFAnchorProfile",
    "GaussianPSFRenderer",
    "GaussianPSFRendererConfig",
    "ZmapLUTRenderer",
    "ZmapCoefficientRenderer",
    "add_fixed_astigmatism_anchor",
    "NATCoeffStack",
    "NATComponent",
    "NATFieldConfig",
    "NATGamma",
    "build_named_nat_config",
    "default_order1_config",
    "order1_13_config",
    "order1_21_config",
    "evaluate_zernike_coefficients_torch",
    "evaluate_zernike_from_roi_positions_torch",
    "full_roi_coeff_stack_torch",
    "get_fov_coordinates_torch",
    "VectorPSFParams",
    "build_vector_psf_context",
    "render_vector_psf_bank",
    "render_vector_psf_stack",
    "resolve_astigmatism_anchor_profile",
    "resolve_circular_psf_anchor_profile",
]
