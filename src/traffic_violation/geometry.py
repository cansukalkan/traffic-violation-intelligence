"""Small, dependency-free geometry primitives used by traffic rules."""

from __future__ import annotations

from collections.abc import Sequence

Point = tuple[float, float]
Box = tuple[float, float, float, float]

_EPSILON = 1e-9


def _cross(origin: Point, first: Point, second: Point) -> float:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (
        first[1] - origin[1]
    ) * (second[0] - origin[0])


def _on_segment(point: Point, start: Point, end: Point) -> bool:
    if abs(_cross(start, end, point)) > _EPSILON:
        return False
    return (
        min(start[0], end[0]) - _EPSILON
        <= point[0]
        <= max(start[0], end[0]) + _EPSILON
        and min(start[1], end[1]) - _EPSILON
        <= point[1]
        <= max(start[1], end[1]) + _EPSILON
    )


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Return whether a point lies inside or on the boundary of a polygon."""

    if len(polygon) < 3:
        raise ValueError("a polygon must contain at least three points")

    inside = False
    px, py = point
    previous = polygon[-1]
    for current in polygon:
        if _on_segment(point, previous, current):
            return True
        x1, y1 = previous
        x2, y2 = current
        if (y1 > py) != (y2 > py):
            x_at_y = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < x_at_y:
                inside = not inside
        previous = current
    return inside


def segments_intersect(
    first_start: Point,
    first_end: Point,
    second_start: Point,
    second_end: Point,
) -> bool:
    """Return whether two closed line segments intersect, including touching."""

    c1 = _cross(first_start, first_end, second_start)
    c2 = _cross(first_start, first_end, second_end)
    c3 = _cross(second_start, second_end, first_start)
    c4 = _cross(second_start, second_end, first_end)

    if ((c1 > _EPSILON and c2 < -_EPSILON) or (c1 < -_EPSILON and c2 > _EPSILON)) and (
        (c3 > _EPSILON and c4 < -_EPSILON) or (c3 < -_EPSILON and c4 > _EPSILON)
    ):
        return True
    return (
        (abs(c1) <= _EPSILON and _on_segment(second_start, first_start, first_end))
        or (abs(c2) <= _EPSILON and _on_segment(second_end, first_start, first_end))
        or (abs(c3) <= _EPSILON and _on_segment(first_start, second_start, second_end))
        or (abs(c4) <= _EPSILON and _on_segment(first_end, second_start, second_end))
    )


def box_bottom_center(box: Box) -> Point:
    """Return the ground-contact proxy point for an ``(x1, y1, x2, y2)`` box."""

    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive width and height")
    return ((x1 + x2) / 2.0, y2)

