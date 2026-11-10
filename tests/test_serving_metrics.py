"""Literal helper projections only; no native receipt or measurement evidence."""

import math
import unittest

from rlcd.serving_metrics import summarize_serving_jobs, validate_first_output_timing


class FirstOutputTimingTests(unittest.TestCase):
    def test_null_until_first_success_including_partial_failure(self):
        for load, raw in ((None, None), (1.0, None), (1.0, 2.0)):
            with self.subTest(load=load, raw=raw):
                validate_first_output_timing(None, first_succeeded=False,
                                             load_seconds=load, first_raw_seconds=raw)
        for duration in (0, 1.0, False, math.nan):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                validate_first_output_timing(duration, first_succeeded=False,
                                             load_seconds=1.0, first_raw_seconds=2.0)

    def test_finite_builtin_numbers_zero_and_enclosure_tolerance(self):
        for duration, load, raw in ((0, 0, 0), (-0.0, 0.0, 0.0), (3, 1, 2),
                                    (4.5, 1.25, 2.25), (3.0 - 5e-13, 1.0, 2.0)):
            with self.subTest(duration=duration):
                validate_first_output_timing(duration, first_succeeded=True,
                                             load_seconds=load, first_raw_seconds=raw)
        for duration, load, raw in ((3.0 - 2e-12, 1.0, 2.0), (1.7e308, 1e308, 1e308)):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                validate_first_output_timing(duration, first_succeeded=True,
                                             load_seconds=load, first_raw_seconds=raw)

    def test_invalid_scalars_and_overflow_raise_value_error(self):
        invalid = (None, True, False, "1", [], {}, 1j, math.nan, math.inf, -math.inf, -1, 10 ** 400)
        for name in ("duration", "load_seconds", "first_raw_seconds"):
            for value in invalid:
                arguments = {"duration": 10.0, "load_seconds": 1.0, "first_raw_seconds": 2.0}
                arguments[name] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    validate_first_output_timing(first_succeeded=True, **arguments)
        for flag in (None, 0, 1, "yes"):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                validate_first_output_timing(None, first_succeeded=flag,
                                             load_seconds=None, first_raw_seconds=None)


class ServingSummaryTests(unittest.TestCase):
    def test_supervisor_success_controls_samples_and_count_identities(self):
        assigned = [
            {"job_id": "accepted", "status": "succeeded", "receipt": {
                "load_to_first_output_seconds": 3.0,
                "calls": [{"index": 0, "phase": "first", "seconds": {"raw_call": 2.0}},
                          {"index": 1, "phase": "warmup", "seconds": {"raw_call": 0.5}},
                          {"index": 2, "phase": "measured", "seconds": {"raw_call": 0.25}},
                          {"index": 3, "phase": "measured", "seconds": {"raw_call": 0.2}}]}},
            {"job_id": "later-failed", "status": "nonzero_exit", "receipt": {
                "status": "failed", "load_to_first_output_seconds": 4.0}},
            {"job_id": "timed-out", "status": "timeout", "receipt": None,
             "progress": {"load_to_first_output_seconds": 5.0}},
            {"job_id": "child-success-nonzero", "status": "nonzero_exit", "receipt": {
                "status": "succeeded", "load_to_first_output_seconds": 6.0}},
            {"job_id": "no-first-output", "status": "missing_receipt", "receipt": None,
             "progress": {"load_to_first_output_seconds": None}},
        ]
        # A first-call observation survives a later job failure, without eligibility.
        validate_first_output_timing(assigned[1]["receipt"]["load_to_first_output_seconds"],
                                     first_succeeded=True, load_seconds=1.0, first_raw_seconds=2.0)
        summary = summarize_serving_jobs(assigned, 2)
        self.assertEqual(summary["assigned_jobs"], 5)
        self.assertEqual(summary["successful_jobs"], 1)
        self.assertEqual(summary["unsuccessful_jobs"], 4)
        self.assertEqual(summary["dispositions"], {"succeeded": 1, "nonzero_exit": 2,
                                                 "timeout": 1, "missing_receipt": 1})
        self.assertEqual(sum(summary["dispositions"].values()), summary["assigned_jobs"])
        self.assertEqual(summary["measured_calls_assigned"], 10)
        self.assertEqual(summary["measured_calls_excluded_from_success_samples"], 8)
        self.assertEqual(summary["conditional_success_samples"], [
            {"job_id": "accepted", "call_index": 2, "seconds": {"raw_call": 0.25}},
            {"job_id": "accepted", "call_index": 3, "seconds": {"raw_call": 0.2}}])
        self.assertEqual(summary["load_to_first_output_samples_assigned"], 5)
        self.assertEqual(summary["load_to_first_output_samples_excluded_from_success_samples"], 4)
        self.assertEqual(summary["conditional_success_load_to_first_output_samples"], [
            {"job_id": "accepted", "seconds": 3.0}])
        self.assertEqual(assigned[1]["receipt"]["load_to_first_output_seconds"], 4.0)
        self.assertEqual(assigned[2]["progress"]["load_to_first_output_seconds"], 5.0)

    def test_empty_and_unsuccessful_assignments_have_no_success_samples(self):
        for assigned in ([], [
                {"job_id": "launch", "status": "launch_failed", "receipt": None},
                {"job_id": "invalid", "status": "malformed_receipt", "receipt": None}]):
            with self.subTest(assigned=assigned):
                summary = summarize_serving_jobs(assigned, 2)
                self.assertEqual(summary["conditional_success_samples"], [])
                self.assertEqual(summary["conditional_success_load_to_first_output_samples"], [])
                self.assertEqual(summary["measured_calls_excluded_from_success_samples"], len(assigned) * 2)
                self.assertEqual(summary["load_to_first_output_samples_excluded_from_success_samples"], len(assigned))

    def test_successful_first_output_sample_cannot_be_missing_or_null(self):
        with self.assertRaises(KeyError):
            summarize_serving_jobs([{"job_id": "missing", "status": "succeeded",
                                     "receipt": {"calls": []}}], 1)
        for duration in (None, True, -1, math.inf, 10 ** 400):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                summarize_serving_jobs([{"job_id": "invalid", "status": "succeeded", "receipt": {
                    "calls": [], "load_to_first_output_seconds": duration}}], 1)
