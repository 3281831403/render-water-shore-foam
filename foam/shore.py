"""Deterministic shore-foam rendering primitives.

Contract
--------
* Every depth, distance, width and wavelength is expressed in metres.
  Depths may be negative: a negative depth is simply a sample above the
  water line and must not produce a negative shore distance.
* One rendered frame advances time by exactly ``1.0`` second.
* Noise is analytic and deterministic: no random module is used and no
  texture state is shared between calls.
* Floating point results in ``tests/test_shore.py`` are asserted with an
  absolute tolerance of ``1e-9``.

Every function below is an intentional stub for the bug-fixing exercise.
"""

import math

TWO_PI = math.tau
TOLERANCE = 1e-9
NOISE_FREQ = 1.7
NOISE_PHASE = 0.3

REQUIRED_PARAMS = (
    "width",
    "power",
    "speed",
    "wavelength",
    "slope",
    "max_depth",
    "camera_speed",
)


def shore_distance(depth: float, slope: float) -> float:
    """Convert a signed water depth to a non-negative shore distance.

    ``depth`` is the water depth in metres; negative values describe points
    above the water line.  ``slope`` is the bed slope in metres of depth per
    metre of horizontal distance and must be non-zero.

    The contract is::

        abs(depth) / abs(slope)

    so the result is always non-negative, regardless of the sign of either
    argument.  Raise ``ValueError`` when ``slope == 0``.  Accuracy: 1e-9.
    """
    raise NotImplementedError("shore_distance is not implemented")


def foam_density(dist: float, width: float, power: float) -> float:
    """Return foam density in ``[0, 1]`` at ``dist`` metres from the shoreline.

    Foam is dense at the waterline and thins out with distance::

        ratio = max(dist, 0) / width
        density = clamp(1 - ratio ** power, 0, 1)

    The supplied ``width`` (metres) is the distance at which density reaches
    zero; it must not be replaced by a hard-coded 1 m.  ``power`` controls the
    falloff profile (a power falloff, not a linear ramp) and must be positive.
    ``width <= 0`` or ``power <= 0`` raises ``ValueError``.  Accuracy: 1e-9.
    """
    raise NotImplementedError("foam_density is not implemented")


def wave_phase(t: float, speed: float, wavelength: float) -> float:
    """Return the wave phase in radians at time ``t`` seconds.

    The phase advances as::

        cycles = frac(speed * t / wavelength)
        phase = 2*pi * cycles

    where ``frac(x) = x % 1.0``.  Thus phase is in ``[0, 2*pi)`` and is
    periodic: ``wave_phase(t + wavelength/speed, ...) == wave_phase(t, ...)``
    within 1e-9.  Between branch wraps the phase is continuous in ``t`` and
    advances at ``2*pi*speed/wavelength`` radians per second.  Dividing by the
    wavelength is mandatory; using ``t*speed`` alone makes the crest speed
    wrong.  ``wavelength <= 0`` raises ``ValueError``.  Accuracy: 1e-9.
    """
    raise NotImplementedError("wave_phase is not implemented")


def foam_mask(dist: float, phase: float, noise: float) -> float:
    """Return one pixel's foam mask in ``[0, 1]``.

    ``dist`` is the distance from shore in metres, ``phase`` the wave phase in
    radians and ``noise`` the deterministic noise sample for this pixel.
    ``noise`` is clamped to ``[0, 1]`` and modulates a moving wave band::

        mask = clamp(noise * (0.5 + 0.5*cos(phase + dist)), 0, 1)

    The ``phase`` term is what keeps the pattern moving from frame to frame;
    ignoring it freezes the foam.  The mask is continuous in ``dist``: a step
    of ``0.01`` m changes the result by well under ``0.1``.  Accuracy: 1e-9.
    """
    raise NotImplementedError("foam_mask is not implemented")


def foam_at_frame(depth_map, t: int, params: dict) -> list:
    """Render the foam intensities for frame ``t`` of ``depth_map``.

    ``depth_map`` is a sequence indexed by frame, so ``depth_map[t]`` is that
    frame's flat row-major sequence of signed depths.  ``t`` is a non-negative
    integer frame index; one frame is 1 second.  Let ``N`` be the number of
    pixels in the frame.  For output pixel ``i`` the camera-drifted UV is::

        u     = (i + camera_speed * t) mod N
        d     = linear interpolation of depth_map[t] at u (wrapping at N)

    and the remaining terms are::

        dist  = shore_distance(d, slope)
        phase = wave_phase(t, speed, wavelength)
        noise = 0.5 + 0.5*sin(NOISE_FREQ*u + NOISE_PHASE)
        out   = foam_density(dist, width, power)
                * foam_mask(dist, phase, noise)
                * fade_with_depth(d, max_depth)

    Return a fresh list of floats in ``[0, 1]``.  The current frame must be
    used (no one-frame lag).  Sample the depth map at the drifted UV rather
    than at the fixed screen index: with an integer camera shift the whole
    rendered frame must translate by that many pixels, so the shoreline does
    not stay nailed to the screen.  Invalid params or an out-of-range ``t``
    raise ``ValueError``.  Accuracy: 1e-9.
    """
    raise NotImplementedError("foam_at_frame is not implemented")


def fade_with_depth(depth: float, max_depth: float) -> float:
    """Return a foam fade factor in ``[0, 1]`` for a signed water depth.

    Foam fades out as water deepens and is fully gone at ``max_depth``::

        clamp(1 - max(depth, 0) / max_depth, 0, 1)

    Consequently ``depth == 0`` gives ``1.0``, ``depth == max_depth`` gives
    ``0.0`` and any deeper sample also gives ``0.0``; a deep-water sample must
    never keep a fade of 1.  ``max_depth <= 0`` raises ``ValueError``.
    Accuracy: 1e-9.
    """
    raise NotImplementedError("fade_with_depth is not implemented")


def validate_params(params: dict) -> list[str]:
    """Return a list of human-readable problems; empty means valid.

    ``params`` must contain ``width``, ``power``, ``speed``, ``wavelength``,
    ``slope``, ``max_depth`` and ``camera_speed``.  Report missing fields and
    these range errors: width, power, speed, wavelength and max_depth must be
    positive, slope must be non-zero and camera_speed must be non-negative.
    Every problem is collected; the function does not raise for ordinary bad
    data.  A ``width <= 0`` entry must be reported (otherwise later code
    divides by zero), and every issue string must contain the offending field
    name (for example ``"width"``, ``"speed"`` or ``"max_depth"``).  Accuracy:
    exact string containment is enough; numeric checks use 1e-9.
    """
    raise NotImplementedError("validate_params is not implemented")


def foam_batch(frames, params: dict) -> dict:
    """Render a batch of frames independently.

    ``frames`` is the same depth-map sequence accepted by
    :func:`foam_at_frame`.  Frame ``i`` is rendered at ``t=i``.  Return::

        {
            "count": len(frames),
            "frames": [list[float], ...],
        }

    Each frame is a fresh computation: the noise texture and wave phase must
    not be cached or reused across frames, so a repeated depth map at a later
    time still produces a different foam pattern.  Invalid params raise
    ``ValueError`` and no partial batch is returned.  Accuracy: 1e-9.
    """
    raise NotImplementedError("foam_batch is not implemented")
