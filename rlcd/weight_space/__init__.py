"""Contract-guided programs over neural parameters.

Exact real-arithmetic certificates cover explicit ReLU and sequential backends,
and supplied frozen-feature readouts. Numerical execution, the compiler, and
empirical utility remain separate trust/evaluation boundaries.
"""

from .model import Network, InputBox
from .artifact import TensorArtifact, WeightArtifact, StructuralPolicy
from .program import Edit, Program, apply_program, difference, compose, interpolate, rollback
from .contracts import Contract, check_contract, certify_drift, verify_certificate
from .compiler import compile_readout
from .symmetry import Gauge, align_hidden, transform, transport, transfer_edit
from .semantics import SemanticsAdapter, SemanticsRegistry, default_registry, to_tensor_artifact, as_relu_network
from .structured import StructuredProgram, Lowering, lower_structured, native_round, verify_lowering
from .features import ReadoutHead, FeatureCompilation, compile_feature_edit, certify_feature_retention, verify_feature_retention
from .sequential import sequential_artifact, sequential_forward, certify_sequential_drift, verify_sequential_certificate
from .workflow import TaskSpec, AccessPolicy, BudgetLimits, BudgetUsage, SplitSpec, ProgrammerSession, Candidate, EvaluationReceipt, ReplayProvider, validate_splits, verify_trace

__all__ = [
    "Network", "InputBox", "Edit", "Program", "apply_program", "difference",
    "compose", "interpolate", "Contract", "check_contract", "certify_drift",
    "verify_certificate", "compile_readout", "Gauge", "align_hidden", "transform",
    "transport", "rollback", "transfer_edit", "TensorArtifact", "WeightArtifact",
    "StructuralPolicy", "SemanticsAdapter", "SemanticsRegistry", "default_registry",
    "to_tensor_artifact", "as_relu_network",
    "StructuredProgram", "Lowering", "lower_structured", "native_round", "verify_lowering",
    "ReadoutHead", "FeatureCompilation", "compile_feature_edit", "certify_feature_retention", "verify_feature_retention",
    "sequential_artifact", "sequential_forward", "certify_sequential_drift", "verify_sequential_certificate",
    "TaskSpec", "AccessPolicy", "BudgetLimits", "BudgetUsage", "SplitSpec", "ProgrammerSession",
    "Candidate", "EvaluationReceipt", "ReplayProvider", "validate_splits", "verify_trace",
]
