import pytest

from traffic_violation.geometry import box_bottom_center, point_in_polygon, segments_intersect

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_point_in_polygon() -> None:
    assert point_in_polygon((5.0, 5.0), SQUARE)
    assert not point_in_polygon((15.0, 5.0), SQUARE)


@pytest.mark.parametrize("point", [(0.0, 0.0), (5.0, 0.0), (10.0, 10.0)])
def test_polygon_boundary_is_inside(point: tuple[float, float]) -> None:
    assert point_in_polygon(point, SQUARE)


def test_invalid_polygon_is_rejected() -> None:
    with pytest.raises(ValueError, match="three"):
        point_in_polygon((0.0, 0.0), [(0.0, 0.0), (1.0, 1.0)])


def test_segments_intersect() -> None:
    assert segments_intersect((0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0))
    assert not segments_intersect((0.0, 0.0), (1.0, 1.0), (5.0, 5.0), (6.0, 6.0))


def test_collinear_segments_and_touching_endpoints_intersect() -> None:
    assert segments_intersect((0.0, 0.0), (5.0, 0.0), (3.0, 0.0), (9.0, 0.0))
    assert segments_intersect((0.0, 0.0), (5.0, 0.0), (5.0, 0.0), (9.0, 3.0))


def test_box_bottom_center() -> None:
    assert box_bottom_center((0.0, 0.0, 10.0, 20.0)) == (5.0, 20.0)


def test_invalid_box_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        box_bottom_center((10.0, 0.0, 2.0, 20.0))

