"""Pure timing checks and summaries for supplied serving-job projections."""

import math
from collections import Counter


def _duration(value):
    if type(value) not in (int, float):
        raise ValueError("invalid_first_output_timing")
    try:
        seconds = float(value)
    except OverflowError as exc:
        raise ValueError("invalid_first_output_timing") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("invalid_first_output_timing")
    return seconds


def validate_first_output_timing(duration, *, first_succeeded, load_seconds, first_raw_seconds):
    """Require null before first success, then a finite enclosing duration.

    The caller validates the rest of the receipt. Overall job failure after the
    first successful call does not erase this observation. Zero is permitted.
    """
    if type(first_succeeded) is not bool:
        raise ValueError("invalid_first_output_lifecycle")
    if not first_succeeded:
        if duration is not None:
            raise ValueError("first_output_timing_before_success")
        return
    elapsed = _duration(duration)
    minimum = _duration(load_seconds) + _duration(first_raw_seconds)
    if not math.isfinite(minimum) or elapsed + 1e-12 < minimum:
        raise ValueError("first_output_timing_not_enclosing")


def summarize_serving_jobs(assigned, repeats):
    """Summarize already-validated assignment projections and repeat counts.

    Eligibility comes from supervisor job status. This does not validate full
    receipts or infer missing samples; successful first-output samples must be
    present and finite. Overlapping spans are samples, never additive totals.
    """
    successful = [job for job in assigned if job["status"] == "succeeded"]
    samples = [{"job_id": job["job_id"], "call_index": call["index"], "seconds": call["seconds"]}
               for job in successful for call in job["receipt"]["calls"] if call["phase"] == "measured"]
    first_output_samples = []
    for job in successful:
        duration = job["receipt"]["load_to_first_output_seconds"]
        _duration(duration)
        first_output_samples.append({"job_id": job["job_id"], "seconds": duration})
    jobs = len(assigned)
    return {"assigned_jobs": jobs, "successful_jobs": len(successful),
            "unsuccessful_jobs": jobs - len(successful),
            "dispositions": dict(Counter(job["status"] for job in assigned)),
            "measured_calls_assigned": jobs * repeats,
            "measured_calls_excluded_from_success_samples": jobs * repeats - len(samples),
            "conditional_success_samples": samples,
            "load_to_first_output_samples_assigned": jobs,
            "load_to_first_output_samples_excluded_from_success_samples": jobs - len(first_output_samples),
            "conditional_success_load_to_first_output_samples": first_output_samples}
