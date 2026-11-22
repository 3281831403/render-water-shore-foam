"""Fifteen red tests for the shore-foam rendering exercise."""

import math

import pytest

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

TOL = 1e-9


def _params(**overrides):
    """Build a complete parameter mapping using the project's canonical keys."""
    params = {
        "width": 2.0,
        "power": 2.0,
        "speed": 1.0,
        "wavelength": 4.0,
        "slope": 0.5,
        "max_depth": 1.0,
        "camera_speed": 0.0,
    }
    params.update(overrides)
    return params


def test_shore_distance_is_non_negative_for_signed_depth():
    """Defect 1: a negative depth must not produce a negative distance."""
    assert shore_distance(3.0, 0.5) == pytest.approx(6.0, abs=TOL)
    assert shore_distance(-3.0, 0.5) == pytest.approx(6.0, abs=TOL)
    assert shore_distance(-1.5, -0.5) == pytest.approx(3.0, abs=TOL)


def test_foam_density_uses_requested_width():
    """Defect 2: the width argument must replace the hard-coded 1 m."""
    assert foam_density(2.0, width=4.0, power=1.0) == pytest.approx(0.5, abs=TOL)
    assert foam_density(1.0, width=4.0, power=1.0) == pytest.approx(0.75, abs=TOL)
    assert foam_density(3.0, width=4.0, power=1.0) == pytest.approx(0.25, abs=TOL)


def test_foam_density_falls_off_with_power():
    """Defect 3: density must use a power falloff, not a linear ramp."""
    assert foam_density(1.0, 2.0, 2.0) == pytest.approx(0.75, abs=TOL)
    assert foam_density(1.0, 2.0, 3.0) == pytest.approx(0.875, abs=TOL)
    assert foam_density(1.5, 2.0, 2.0) == pytest.approx(0.4375, abs=TOL)


def test_wave_phase_scales_with_wavelength_and_time():
    """Defect 4: phase speed must divide by the wavelength and stay continuous."""
    speed = 2.5
    wavelength = 5.0
    t = 0.7
    expected = math.tau * ((speed * t / wavelength) % 1.0)

    assert wave_phase(t, speed, wavelength) == pytest.approx(expected, abs=TOL)

    dt = 1e-6
    expected_step = math.tau * speed * dt / wavelength
    actual_step = wave_phase(t + dt, speed, wavelength) - wave_phase(
        t, speed, wavelength
    )
    assert actual_step == pytest.approx(expected_step, abs=TOL)


def test_foam_mask_moves_with_phase():
    """Defect 5: the noise band must change with time (phase)."""
    quiet = foam_mask(0.5, 0.0, 1.0)
    crest = foam_mask(0.5, math.pi, 1.0)

    assert quiet == pytest.approx(0.5 + 0.5 * math.cos(0.5), abs=TOL)
    assert crest == pytest.approx(0.5 + 0.5 * math.cos(math.pi + 0.5), abs=TOL)
    assert abs(quiet - crest) > 0.5


def test_foam_at_frame_reads_current_frame():
    """Defect 6: frame t must use depth_map[t], never depth_map[t - 1]."""
    deep = [5.0] * 6
    shallow = [0.2] * 6
    params = _params()

    out = foam_at_frame([deep, shallow], 1, params)

    assert len(out) == 6
    assert all(0.0 <= value <= 1.0 for value in out)
    assert max(out) > 0.05


def test_fade_with_depth_reaches_zero():
    """Defect 7: foam at max_depth must already be fully faded out."""
    assert fade_with_depth(0.0, 1.0) == pytest.approx(1.0, abs=TOL)
    assert fade_with_depth(0.5, 1.0) == pytest.approx(0.5, abs=TOL)
    assert fade_with_depth(1.0, 1.0) == pytest.approx(0.0, abs=TOL)
    assert fade_with_depth(3.0, 1.0) == pytest.approx(0.0, abs=TOL)


def test_validate_params_rejects_non_positive_width():
    """Defect 8: width <= 0 must be reported before any division."""
    assert validate_params(_params()) == []

    for bad_width in (0.0, -2.0):
        issues = validate_params(_params(width=bad_width))
        assert issues
        assert any("width" in issue.lower() for issue in issues)

    speed_issues = validate_params(_params(speed=0.0))
    assert any("speed" in issue.lower() for issue in speed_issues)
    assert any(
        "max_depth" in issue.lower()
        for issue in validate_params(_params(max_depth=-1.0))
    )


def test_foam_batch_does_not_freeze_second_frame():
    """Defect 9: a cached noise texture must not make frame 1 equal frame 0."""
    frame = [0.2] * 6
    params = _params(speed=1.0, wavelength=4.0, camera_speed=0.5)

    result = foam_batch([frame, frame], params)

    assert result["count"] == 2
    first, second = result["frames"]
    assert max(abs(a - b) for a, b in zip(first, second)) > 0.05


def test_foam_at_frame_drifts_with_camera():
    """Defect 10: camera motion must translate the whole rendered shoreline."""
    gradient = [0.1 + 0.08 * index for index in range(8)]
    depth_map = [gradient, gradient]
    static = foam_at_frame(depth_map, 1, _params(camera_speed=0.0))
    drift = foam_at_frame(depth_map, 1, _params(camera_speed=2.0))

    for i in range(6):
        assert drift[i] == pytest.approx(static[i + 2], abs=TOL)

    assert abs(static[0] - static[2]) > 1e-6


def test_foam_density_is_one_at_shore():
    """Additional check: zero water depth maps to density exactly 1."""
    assert foam_density(0.0, 2.0, 2.0) == pytest.approx(1.0, abs=TOL)
    assert foam_density(0.0, 7.5, 0.5) == pytest.approx(1.0, abs=TOL)
    assert foam_density(0.0, 1.0, 3.0) == pytest.approx(1.0, abs=TOL)
    assert foam_density(
        shore_distance(0.0, 0.5), 2.0, 2.0
    ) == pytest.approx(1.0, abs=TOL)


def test_fade_with_depth_hides_deep_water():
    """Additional check: samples at or beyond max_depth produce no foam."""
    assert fade_with_depth(2.0, 2.0) == pytest.approx(0.0, abs=TOL)
    assert fade_with_depth(10.0, 2.0) == pytest.approx(0.0, abs=TOL)
    assert fade_with_depth(1.0, 2.0) == pytest.approx(0.5, abs=TOL)

    deep_frame = foam_at_frame([[2.5] * 4], 0, _params(max_depth=2.0))
    assert all(value == pytest.approx(0.0, abs=TOL) for value in deep_frame)


def test_wave_phase_is_periodic():
    """Additional check: phase at t and t + wavelength/speed is the same."""
    speed = 2.5
    wavelength = 3.0
    period = wavelength / speed
    t = 1.3

    assert wave_phase(t + period, speed, wavelength) == pytest.approx(
        wave_phase(t, speed, wavelength), abs=TOL
    )


def test_foam_mask_is_continuous_across_pixels():
    """Additional check: adjacent mask samples differ by less than 0.1."""
    phase = 0.7
    noise = 0.8
    previous = foam_mask(0.0, phase, noise)

    for step in range(1, 100):
        current = foam_mask(step * 0.01, phase, noise)
        assert abs(current - previous) < 0.1
        previous = current


def test_foam_batch_matches_individual_frames():
    """Additional check: batched frames equal independently rendered frames."""
    first = [0.1, 0.3, 0.6, 0.9]
    second = [0.2, 0.4, 0.8, 1.0]
    frames = [first, second]
    params = _params(camera_speed=0.5)

    result = foam_batch(frames, params)

    assert result["count"] == 2
    for index, frame in enumerate(frames):
        expected = foam_at_frame(frames, index, params)
        actual = result["frames"][index]
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected):
            assert got == pytest.approx(want, abs=TOL)
