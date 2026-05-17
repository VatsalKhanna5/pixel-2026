"""
src/dataset/__init__.py
PIXEL-2026 Dataset Package
"""

from src.dataset.primitives        import sample_layout, PRIMITIVE_NAMES, N_PRIMITIVES
from src.dataset.connectivity      import is_connected, check_drc
from src.dataset.em_simulation     import simulate, FREQS, SUBSTRATES
from src.dataset.physics_validator import validate_record, check_dataset_quality_gates
from src.dataset.hdf5_writer       import CheckpointWriter, resume_from_checkpoint, verify_hdf5_integrity

__all__ = [
    "sample_layout", "PRIMITIVE_NAMES", "N_PRIMITIVES",
    "is_connected", "check_drc",
    "simulate", "FREQS", "SUBSTRATES",
    "validate_record", "check_dataset_quality_gates",
    "CheckpointWriter", "resume_from_checkpoint", "verify_hdf5_integrity",
]
