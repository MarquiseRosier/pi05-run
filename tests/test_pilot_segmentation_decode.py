"""Segmentation readback keeps native buffers and decoded IDs in their correct dtypes."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from run_pi05_occlusion_pilot import decode_segmentation_rgb, install_segmentation_reader


def test_decode_all_three_bytes_background_and_non_geom_objects():
    # The green/blue channels must contribute without truncation or overflow.
    geoms = [NS(segid=-1, objtype=5, objid=0)] * 65537
    geoms[0] = NS(segid=0, objtype=5, objid=80)
    geoms[1] = NS(segid=255, objtype=5, objid=81)
    geoms[2] = NS(segid=65536, objtype=6, objid=82)
    scene = NS(ngeom=len(geoms), geoms=geoms)
    rgb = np.array([[[1, 0, 0], [0, 1, 0], [1, 0, 1]],
                    [[0, 0, 0], [255, 255, 255], [2, 0, 0]]], dtype=np.uint8)
    original = rgb.copy()
    decoded = decode_segmentation_rgb(rgb, scene)
    assert decoded.shape == (2, 3, 2)
    assert decoded.dtype == np.int32
    np.testing.assert_array_equal(decoded, [[[5, 80], [5, 81], [6, 82]],
                                            [[-1, -1], [-1, -1], [-1, -1]]])
    np.testing.assert_array_equal(rgb, original)


@pytest.mark.parametrize('rgb', [np.zeros((3, 4), np.uint8), np.zeros((3, 4, 2), np.uint8),
                                  np.zeros((3, 4, 3), np.uint32)])
def test_invalid_buffer_reports_shape_and_dtype(rgb):
    with pytest.raises(ValueError, match='Expected segmentation RGB buffer HxWx3 uint8'):
        decode_segmentation_rgb(rgb, NS(ngeom=0, geoms=[]))


@pytest.mark.parametrize('depth', [False, True])
def test_reader_bypasses_legacy_decode_and_preserves_rgb_depth_and_resolution(depth):
    rgb = np.zeros((128, 256, 3), np.uint8)
    rgb[12, 34] = [1, 0, 0]
    distances = np.ones((128, 256), np.float32)
    calls = []
    def old_reader(width, height, depth=False, segmentation=False):
        calls.append((width, height, depth))
        assert not segmentation, 'Legacy uint8 segmentation arithmetic must not run'
        return (rgb, distances) if depth else rgb
    context = NS(read_pixels=old_reader, scn=NS(ngeom=1, geoms=[NS(segid=0, objtype=5, objid=9)]))
    sim = NS(_render_context_offscreen=context)
    install_segmentation_reader(sim)
    first = context.read_pixels
    install_segmentation_reader(sim)
    assert context.read_pixels is first
    result = context.read_pixels(256, 128, depth=depth, segmentation=True)
    decoded = result[0] if depth else result
    assert decoded.shape == (128, 256, 2)
    assert decoded[12, 34].tolist() == [5, 9]
    assert decoded[0, 0].tolist() == [-1, -1]
    if depth:
        assert result[1] is distances
    normal = context.read_pixels(256, 128, depth=depth)
    assert (normal[0] if depth else normal) is rgb
    assert calls == [(256, 128, depth)] * 2
