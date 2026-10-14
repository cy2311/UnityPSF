from __future__ import annotations

import torch

from unity_psf.optics.zmap_renderer import ZmapLUTRenderer

def test_zmap_renderer_separates_field_origin_from_local_roi_origin() -> None:
    planes = torch.ones(2, 9, 9)
    renderer = ZmapLUTRenderer(
        planes, z_min_nm=-100.0, z_max_nm=100.0, pixel_size_nm_xy=(10.0, 10.0),
        roi_shape_yx=(32, 32), roi_origin_nm_xy=(100.0, 200.0),
    )
    image = renderer(torch.tensor([[120.0, 220.0, 0.0]]), torch.tensor([500.0]), torch.tensor([2.0]))
    assert float(image.max()) > 2.0
