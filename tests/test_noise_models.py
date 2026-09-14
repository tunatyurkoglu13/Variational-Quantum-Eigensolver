"""`build_noise_model`/`summarize_noise` must reflect FakeTorino's real calibration data,
including the disabled-coupler pathology discovered while writing this module (see
`noise/models.py`'s docstring)."""

from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeTorino

from vqe_nisq_project.noise.models import build_noise_model, summarize_noise


def test_build_noise_model_returns_a_populated_noise_model() -> None:
    noise_model = build_noise_model(FakeTorino())
    assert isinstance(noise_model, NoiseModel)
    # Real device noise touches every basis-gate instruction, not a hand-picked subset.
    assert set(noise_model.noise_instructions) >= {"x", "sx", "cz", "measure"}


def test_summarize_noise_reports_physically_sane_real_device_numbers() -> None:
    summary = summarize_noise(FakeTorino())

    assert summary.backend_name == "fake_torino"
    assert summary.n_qubits == 133

    # T2 <= 2*T1 is a hard physical bound (dephasing can't be slower than the
    # relaxation-limited bound) -- must hold even for the device-wide means.
    assert summary.mean_t2_us <= 2 * summary.mean_t1_us
    assert summary.mean_t1_us > 0
    assert summary.mean_t2_us > 0

    # Real IBM Heron-class error rates: single-qubit gates ~1e-4, two-qubit gates
    # (median, excluding disabled couplers) ~1e-3 to 1e-2, readout ~1e-2.
    assert 0 < summary.mean_one_qubit_gate_error < 1e-2
    assert 0 < summary.median_two_qubit_gate_error < 5e-2
    assert 0 < summary.median_readout_error < 0.2

    # The real, empirically-found pathology: some couplers are disabled (error==1.0
    # sentinel) in this calibration snapshot, which is why the median (not mean) is
    # used above.
    assert 0 < summary.n_disabled_two_qubit_couplers < summary.n_two_qubit_couplers
