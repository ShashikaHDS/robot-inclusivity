"""Accessibility report generation for cross-section ramp detection."""

from __future__ import annotations

from .detect import DetectionResult, Ramp


def generate_report(
    result: DetectionResult,
    max_slope_deg: float = 35.0,
    max_step_m: float = 0.25,
) -> str:
    """Generate a human-readable accessibility report."""
    lines = []
    lines.append("=" * 60)
    lines.append("RAMP ACCESSIBILITY REPORT (Cross-Section Analysis)")
    lines.append("=" * 60)
    lines.append(f"Profiles analyzed: {result.n_profiles}")
    lines.append(f"Raw candidates: {result.n_candidates}")
    lines.append(f"Ramps detected: {len(result.ramps)}")
    lines.append(f"Robot limits: max_slope={max_slope_deg}°, max_step={max_step_m}m")
    lines.append("")

    if not result.ramps:
        lines.append("No ramps detected.")
        return "\n".join(lines)

    for r in result.ramps:
        status = "PASS" if r.traversable else "FAIL"
        lines.append(
            f"  [{status}] Ramp #{r.ramp_id}: "
            f"angle={r.angle_deg:.1f}°, length={r.length_m:.1f}m, "
            f"width={r.width_m:.1f}m, height_diff={r.height_diff_m:.2f}m"
        )
        lines.append(
            f"         start=({r.start_xy[0]:.1f}, {r.start_xy[1]:.1f}) "
            f"→ end=({r.end_xy[0]:.1f}, {r.end_xy[1]:.1f})"
        )
        lines.append(
            f"         confidence={r.confidence:.2f}, "
            f"detected in {r.n_detections} strip(s)"
        )
        lines.append("")

    pass_count = sum(1 for r in result.ramps if r.traversable)
    fail_count = len(result.ramps) - pass_count
    lines.append(f"Summary: {pass_count} passable, {fail_count} blocked")

    if fail_count > 0:
        lines.append("\nBlocked ramps (exceed robot capability):")
        for r in result.ramps:
            if not r.traversable:
                lines.append(f"  Ramp #{r.ramp_id}: {r.angle_deg:.1f}° > {max_slope_deg}°")

    return "\n".join(lines)
