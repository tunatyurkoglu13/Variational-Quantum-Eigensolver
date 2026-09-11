# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: vqe-nisq-project
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Phase 0 setup check
#
# Confirms the environment is correctly wired end to end: package versions,
# the Qiskit/PennyLane bridge, a PySCF ground-truth calculation, and the
# shared publication plotting style. Paired to `00_setup_check.ipynb` via
# jupytext (`text_representation: percent`) so git diffs stay readable.

# %%
import sys

import matplotlib.pyplot as plt
import mitiq
import mthree
import numpy as np
import openfermion
import pennylane as qml
import pyscf
import qiskit
from pyscf import gto, scf

from vqe_nisq_project.viz import CATEGORICAL_COLORS, apply_style

print("Python:", sys.version.split()[0])
print("qiskit:", qiskit.__version__)
print("pennylane:", qml.__version__)
print("pyscf:", pyscf.__version__)
print("openfermion:", openfermion.__version__)
print("mitiq:", mitiq.__version__)
print("mthree:", mthree.__version__)

# %% [markdown]
# ## Ground-truth sanity check: H2 / STO-3G Hartree-Fock

# %%
mol = gto.M(atom="H 0 0 0; H 0 0 0.735", basis="sto-3g", unit="Angstrom", verbose=0)
e_hf = scf.RHF(mol).kernel()
print(f"RHF(H2, STO-3G, R=0.735 A) = {e_hf:.10f} Ha")
assert abs(e_hf - (-1.1169989968)) < 1e-8, "PySCF RHF result drifted from the Phase 0 baseline"

# %% [markdown]
# ## Publication plotting style smoke test

# %%
apply_style()
fig, ax = plt.subplots()
x = np.linspace(0.4, 3.0, 50)
# Illustrative Morse-like curve, NOT a fitted physical potential -- placeholder only.
y = np.exp(-2 * (x - 0.74)) - 2 * np.exp(-(x - 0.74))
ax.plot(x, y, label="illustrative curve (not physics)")
ax.set_xlabel("R (Angstrom)")
ax.set_ylabel("E (illustrative)")
ax.legend()
plt.show()
print("Categorical palette in use:", CATEGORICAL_COLORS[:3], "...")
