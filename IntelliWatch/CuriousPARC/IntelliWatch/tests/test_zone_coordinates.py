"""
tests/test_zone_coordinates.py
Step 16 Unit Test Suite - Coordinate Transformations for Safety Zones.

Validates:
- Display viewport to original frame coordinate transformations
- Letterboxing (horizontal padding) and Pillarboxing (vertical padding)
- Aspect-ratio preserving scaling under 'contain'
- Stretched scaling under 'fill'
- Viewport and container offset handling
- Boundary clamping
- Bidirectional round-trip precision
- Polygon batch conversions
- Cross-resolution adaptation
"""
import pytest

from intelligence.zones.coordinates import (
    display_to_original,
    original_to_display,
    transform_polygon_display_to_original,
    transform_polygon_original_to_display,
    scale_polygon_between_resolutions,
    is_polygon_resolution_compatible,
)


class TestCoordinateTransformations:

    def test_01_exact_match_no_scaling(self):
        """When display and original have identical dimensions, coordinates are unchanged."""
        orig_x, orig_y = display_to_original(
            display_x=150.0,
            display_y=250.0,
            display_width=1280.0,
            display_height=720.0,
            original_width=1280.0,
            original_height=720.0,
            object_fit="contain",
        )
        assert orig_x == 150.0
        assert orig_y == 250.0

    def test_02_contain_uniform_downscale(self):
        """Uniform downscale by factor of 2.0 (1280x720 -> 640x360)."""
        # Display coordinate (320, 180) should map to center of 1280x720: (640, 360)
        orig_x, orig_y = display_to_original(
            display_x=320.0,
            display_y=180.0,
            display_width=640.0,
            display_height=360.0,
            original_width=1280.0,
            original_height=720.0,
            object_fit="contain",
        )
        assert orig_x == 640.0
        assert orig_y == 360.0

    def test_03_contain_pillarbox_vertical_padding(self):
        """
        Original is 16:9 (1280x720). Display is taller 4:3 (800x600).
        Scale is limited by width: scale = 800 / 1280 = 0.625.
        Rendered height = 720 * 0.625 = 450.
        Pillarbox vertical pad_y = (600 - 450) / 2 = 75px.
        """
        disp_w, disp_h = 800.0, 600.0
        orig_w, orig_h = 1280.0, 720.0

        # Click at top of rendered image: display_y = 75px -> should be original y = 0
        ox, oy = display_to_original(
            display_x=400.0,
            display_y=75.0,
            display_width=disp_w,
            display_height=disp_h,
            original_width=orig_w,
            original_height=orig_h,
            object_fit="contain",
        )
        assert ox == 640.0
        assert oy == 0.0

        # Click at center of rendered image: display_y = 75 + 225 = 300px -> orig y = 360
        ox_mid, oy_mid = display_to_original(
            display_x=400.0,
            display_y=300.0,
            display_width=disp_w,
            display_height=disp_h,
            original_width=orig_w,
            original_height=orig_h,
            object_fit="contain",
        )
        assert ox_mid == 640.0
        assert oy_mid == 360.0

    def test_04_contain_letterbox_horizontal_padding(self):
        """
        Original is 4:3 (800x600). Display is widescreen 16:9 (1280x720).
        Scale is limited by height: scale = 720 / 600 = 1.2.
        Rendered width = 800 * 1.2 = 960.
        Letterbox horizontal pad_x = (1280 - 960) / 2 = 160px.
        """
        disp_w, disp_h = 1280.0, 720.0
        orig_w, orig_h = 800.0, 600.0

        # Click at left edge of rendered image: display_x = 160px -> orig x = 0
        ox, oy = display_to_original(
            display_x=160.0,
            display_y=360.0,
            display_width=disp_w,
            display_height=disp_h,
            original_width=orig_w,
            original_height=orig_h,
            object_fit="contain",
        )
        assert ox == 0.0
        assert oy == 300.0

    def test_05_viewport_offset_subtraction(self):
        """Viewport offset inside parent scroll/bounding container is properly subtracted."""
        ox, oy = display_to_original(
            display_x=250.0,
            display_y=150.0,
            display_width=640.0,
            display_height=360.0,
            original_width=640.0,
            original_height=360.0,
            viewport_offset_x=50.0,
            viewport_offset_y=50.0,
            object_fit="fill",
        )
        assert ox == 200.0
        assert oy == 100.0

    def test_06_boundary_clamping(self):
        """Clicks in letterbox black bars clamp gracefully to frame edges [0, dimension]."""
        # Click deep into the top black bar
        ox, oy = display_to_original(
            display_x=400.0,
            display_y=10.0,  # below pad_y = 75px
            display_width=800.0,
            display_height=600.0,
            original_width=1280.0,
            original_height=720.0,
            clamp=True,
        )
        assert oy == 0.0

        # Click far beyond the right edge
        ox_far, oy_far = display_to_original(
            display_x=1500.0,
            display_y=300.0,
            display_width=800.0,
            display_height=600.0,
            original_width=1280.0,
            original_height=720.0,
            clamp=True,
        )
        assert ox_far == 1280.0

    def test_07_bidirectional_roundtrip_precision(self):
        """Converting original -> display -> original matches precisely."""
        orig_x, orig_y = 487.35, 312.80
        disp_w, disp_h = 960.0, 540.0
        orig_w, orig_h = 1376.0, 768.0

        dx, dy = original_to_display(
            original_x=orig_x,
            original_y=orig_y,
            display_width=disp_w,
            display_height=disp_h,
            original_width=orig_w,
            original_height=orig_h,
            object_fit="contain",
        )

        rx, ry = display_to_original(
            display_x=dx,
            display_y=dy,
            display_width=disp_w,
            display_height=disp_h,
            original_width=orig_w,
            original_height=orig_h,
            object_fit="contain",
        )

        assert abs(rx - orig_x) < 0.1
        assert abs(ry - orig_y) < 0.1

    def test_08_polygon_batch_transformation(self):
        """Polygon batch transformations correctly transform all vertices."""
        disp_poly = [
            {"x": 100, "y": 100},
            {"x": 300, "y": 100},
            {"x": 300, "y": 300},
            {"x": 100, "y": 300},
        ]
        orig_poly = transform_polygon_display_to_original(
            display_polygon=disp_poly,
            display_width=640.0,
            display_height=360.0,
            original_width=1280.0,
            original_height=720.0,
            object_fit="contain",
        )
        assert len(orig_poly) == 4
        assert orig_poly[0] == [200.0, 200.0]
        assert orig_poly[1] == [600.0, 200.0]
        assert orig_poly[2] == [600.0, 600.0]
        assert orig_poly[3] == [200.0, 600.0]

    def test_09_scale_polygon_between_resolutions(self):
        """Scales polygon coordinates between two video resolutions proportionally."""
        poly = [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0]]
        scaled = scale_polygon_between_resolutions(
            polygon=poly,
            from_resolution=(640, 360),
            to_resolution=(1280, 720),
        )
        assert scaled[0] == [200.0, 200.0]
        assert scaled[1] == [400.0, 200.0]
        assert scaled[2] == [400.0, 400.0]

    def test_10_polygon_resolution_compatibility(self):
        """Checks if polygon coordinates are within specified resolution bounds."""
        valid_poly = [[50.0, 50.0], [500.0, 50.0], [500.0, 300.0]]
        assert is_polygon_resolution_compatible(valid_poly, (640, 360)) is True

        out_of_bounds_poly = [[50.0, 50.0], [1200.0, 50.0], [1200.0, 300.0]]
        assert is_polygon_resolution_compatible(out_of_bounds_poly, (640, 360)) is False

    def test_11_invalid_dimensions_raise_error(self):
        """Zero or negative dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Display dimensions must be positive"):
            display_to_original(10, 10, 0, 720, 1280, 720)

        with pytest.raises(ValueError, match="Original dimensions must be positive"):
            display_to_original(10, 10, 1280, 720, -100, 720)
