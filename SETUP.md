# Setup

Reproduces the full development environment on a fresh machine in ~10 minutes.
Verified on Apple Silicon (macOS); should work identically on Linux/Intel since
every dependency here ships prebuilt wheels for those platforms too.

## 1. Install uv

```
brew install uv          # macOS
# or: curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Clone and sync

```
git clone https://github.com/tunatyurkoglu13/Variational-Quantum-Eigensolver.git
cd Variational-Quantum-Eigensolver
uv sync --all-groups      # installs Python 3.12 (pinned) + every runtime/dev dependency
```

## 3. Install git hooks

```
uv run pre-commit install
```

## 4. Verify

```
uv run pytest                       # should show all tests passing, >90% coverage
uv run python scripts/demo_experiment.py   # logs a run to sqlite:///mlflow.db
```

## 5. (Optional) IBM Quantum hardware access

Everything in this repo runs on local simulators by default -- an IBM Quantum
account is only needed for the Phase 6 real-hardware runs.

1. Go to <https://quantum.ibm.com> and sign in / create a free IBM account.
2. Once logged in, open your account settings and copy your **API token**.
3. Copy `.env.example` to `.env` and paste the token in:
   ```
   cp .env.example .env
   # edit .env: IBM_QUANTUM_API_TOKEN=<your token>
   ```
   `.env` is gitignored -- it is never committed.
4. Register the token once with Qiskit Runtime so future sessions don't need `.env` at all:
   ```python
   from qiskit_ibm_runtime import QiskitRuntimeService
   from vqe_nisq_project.ibm_account import load_ibm_api_token

   QiskitRuntimeService.save_account(
       channel="ibm_quantum_platform",
       token=load_ibm_api_token(),
       overwrite=True,
   )
   ```

**Free (Open) plan facts** ([source: IBM Quantum blog](https://www.ibm.com/quantum/blog/open-plan-updates)):
- **10 minutes of QPU runtime every 28 days**, at no cost.
- 2026 promotion: use 20 minutes within any 12-month period and you can opt in
  to 180 minutes for the following 12 months, no monthly cap.
- Open Plan users currently get access to `ibm_kingston` (Heron r2, 156 qubits,
  ~2.0x10^-3 median two-qubit error rate).
- IBM does not publish a queue-time guarantee for the Open Plan; in practice,
  expect anywhere from minutes to a few hours depending on system load, since
  paid tiers get scheduling priority. Plan hardware runs accordingly (see
  Phase 6: budget 2-3 well-designed runs, verify everything on `fake_<device>`
  first).

The project is backend-agnostic by design: everything that touches a backend
reads it from config (`statevector | aer_noisy | fake_<device> | ibm_hardware`),
so the entire pipeline runs identically on a noiseless simulator, a noisy
simulator, a calibration-data-based fake backend, or real hardware -- only
that one config value changes.

## 6. (Optional) Docker

```
docker build -t vqe-nisq-project .
docker run --rm vqe-nisq-project -c "import qiskit; print(qiskit.__version__)"
```

Or open the repo in VS Code / a Codespace with the Dev Containers extension --
`.devcontainer/devcontainer.json` uses the same Dockerfile.

## 7. (Optional) Build the LaTeX report

See `report/README.md` -- requires a local LaTeX install (BasicTeX), or just
read the PDF built automatically by CI (`report-pdf` artifact on every push
that touches `report/`).

## 8. (Optional) Documentation site

```
uv run mkdocs serve   # live-reloading docs at http://127.0.0.1:8000
```
