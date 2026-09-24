# Your quantum noise simulator is lying to you (a little)

*Draft blog post. Companion piece to the [technical report](../report/main.pdf) and
[full repo](https://github.com/tunatyurkoglu13/Variational-Quantum-Eigensolver) for the
VQE-on-NISQ-devices project.*

---

I spent the last few weeks trying to answer a question that sounds simple and isn't:
**why is it hard to get chemically accurate energies out of a variational quantum
eigensolver (VQE) on today's quantum hardware?**

The textbook answer offers three suspects. Maybe your ansatz — the parameterized circuit
standing in for the molecule's wavefunction — isn't expressive enough to represent the
true ground state. Maybe your classical optimizer gets stuck in a local minimum before it
finds the right answer. Or maybe the hardware itself is just too noisy.

I built a full pipeline to actually measure all three, on real molecules (H₂, LiH,
BeH₂), rather than guess which one dominates. Then I did something a lot of
simulation-only VQE studies skip: I ran the same circuit on real IBM Quantum hardware
and checked whether my noise simulations had been telling me the truth.

They hadn't. Not by a little.

## The setup, quickly

Nothing here is exotic. Build the molecular Hamiltonian (PySCF), map it to qubits
(Jordan-Wigner / parity / Bravyi-Kitaev, cross-validated against OpenFermion to machine
precision), taper qubits using the Hamiltonian's own symmetries, try a few ansatze
(UCCSD, a generic hardware-efficient circuit, ADAPT-VQE), optimize with four different
classical optimizers across 20 random seeds each, and measure everything with real
statistics — bootstrap confidence intervals, rank-based effect sizes, and
Holm-Bonferroni correction so that comparing many optimizer pairs at once doesn't quietly
inflate my false-positive rate. (It nearly did, once — more on that below.)

The interesting part starts once you add noise.

## Turning up the noise, honestly

Qiskit's Aer simulator can build a noise model straight from a real device's calibration
data — per-qubit relaxation times, per-gate error rates, readout error, the works. I ran
four error-mitigation techniques against that simulated noise: M3 (a fast, classical
correction for readout error), zero-noise extrapolation, Pauli twirling, and dynamical
decoupling.

Two of them worked well. M3 and ZNE both meaningfully closed the gap to the exact answer.
The other two — twirling and dynamical decoupling — did almost nothing, and I made myself
figure out *why* instead of shrugging and moving on. It turned out the simulated noise
channel was already so close to a plain depolarizing channel, and so perfectly
memory-less, that there was essentially no coherent or slowly-varying structure left for
those two techniques to remove. Not a bug — a real, mechanistic dead end, which is its
own kind of useful result.

## Then I checked against a real chip

Once the simulated study was solid, I spent an afternoon getting IBM Quantum credentials
set up (a saga involving a wrong `.env` file location, an empty file, a mysterious
"no matching instance" error, and eventually a real, working connection to `ibm_marrakesh`,
a 156-qubit Heron r2 processor) and ran the exact same circuit for real. Total cost: 55
seconds of a 600-second monthly free quota. Genuinely free.

Here's the number that made me stop and re-derive everything twice to make sure I hadn't
made a mistake: **the real hardware error was 1.5 to 3 times larger than my simulation
had predicted, at every single mitigation setting I tried.**

My first instinct was that I must be using stale calibration data — FakeTorino, my
go-to simulated backend, is a frozen snapshot from a previous chip generation. So I built
a *fresh* noise model straight from `ibm_marrakesh`'s own current calibration numbers and
reran the simulation, at zero additional hardware cost. If stale data were the problem,
this should have closed the gap.

It didn't. The live-calibrated simulation predicted an even *smaller* error than the old
snapshot had — moving further from reality, not closer.

That's the finding I actually care about. It's not "the simulator used old numbers." It's
that Qiskit Aer's standard noise-model construction — depolarizing error plus thermal
relaxation, built from whatever calibration numbers you feed it — is missing some kind of
noise structure that real hardware actually has. Crosstalk, coherent miscalibration,
something non-Markovian. The exact mechanism is future work; the existence of the gap
isn't a guess, it's five measurements pointing the same direction.

## So what actually limits chemical accuracy?

I decomposed the error budget for H₂ into four pieces, each measured on its own:

| Source | Typical size |
|---|---:|
| Ansatz choice (once the optimizer converges) | effectively zero |
| Optimizer reliability | a few mHa |
| Shot noise | a few to tens of mHa, tunable |
| Hardware noise (simulated) | ~23 mHa |
| **Hardware noise (real device)** | **~42 mHa** |

Chemical accuracy is 1.6 mHa. Ansatz expressibility was never the problem, at least not
here — a generic hardware-efficient circuit represents H₂'s ground state just fine once
you actually find the right parameters. The bottleneck is hardware noise, by a wide
margin, and today's best simulation tools underestimate exactly how big that margin is.

## The unglamorous part

A good chunk of the real work in this project wasn't physics — it was catching places
where two well-regarded libraries quietly disagreed with each other. Qiskit and
OpenFermion order spin-orbitals into qubits differently. Qiskit and PennyLane disagree on
which end of a bitstring is qubit zero. `qiskit.transpile()` isn't deterministic unless
you explicitly seed it, which meant one of my "real" findings evaporated into "oh, that
was just transpiler randomness" until I pinned it down. None of these are bugs, exactly —
they're conventions, each internally consistent, that silently stop being consistent the
moment you cross a library boundary without checking. I started keeping a running list.
It got long enough to become its own section of the report.

## Try it yourself

Everything here — every number, every figure — comes from a script you can run:
[`scripts/`](https://github.com/tunatyurkoglu13/Variational-Quantum-Eigensolver/tree/main/scripts)
in the repo reproduces the whole study, and the
[test suite](https://github.com/tunatyurkoglu13/Variational-Quantum-Eigensolver/tree/main/tests)
(147+ tests) is mostly regression tests for the real bugs mentioned above, not just
correctness checks. The full derivations, all four mitigation techniques' theory, and an
honest limitations section are in the
[technical report](https://github.com/tunatyurkoglu13/Variational-Quantum-Eigensolver/blob/main/report/main.pdf).

If you take one thing from this: if your VQE noise study is simulation-only, you don't
yet know whether it's a study of your molecule or a study of your simulator. Those are
different things, and right now, at least for this circuit and this device, the gap
between them is about a factor of two.
