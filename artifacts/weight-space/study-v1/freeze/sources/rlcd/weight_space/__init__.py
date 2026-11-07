"""Contract-guided programs over neural parameters.

Real-arithmetic certificates cover two-layer ReLU networks. Numerical execution,
the compiler, and empirical utility are separate trust/evaluation boundaries.
"""

from .model import Network, InputBox
from .artifact import TensorArtifact, WeightArtifact, StructuralPolicy
from .program import Edit, Program, apply_program, difference, compose, interpolate, rollback
from .contracts import Contract, check_contract, certify_drift, verify_certificate
from .compiler import compile_readout
from .symmetry import Gauge, align_hidden, transform, transport, transfer_edit
from .semantics import SemanticsAdapter, SemanticsRegistry, default_registry, to_tensor_artifact, as_relu_network

__all__ = [
    "Network", "InputBox", "Edit", "Program", "apply_program", "difference",
    "compose", "interpolate", "Contract", "check_contract", "certify_drift",
    "verify_certificate", "compile_readout", "Gauge", "align_hidden", "transform",
    "transport", "rollback", "transfer_edit", "TensorArtifact", "WeightArtifact",
    "StructuralPolicy", "SemanticsAdapter", "SemanticsRegistry", "default_registry",
    "to_tensor_artifact", "as_relu_network",
]
