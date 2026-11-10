# cayley-retraction-c

The Cayley retraction for optimising over orthogonal matrices, written in C. It is checked against a NumPy reference, timed against it at several sizes, and used to solve the orthogonal Procrustes problem, whose exact answer is known.

The project is a study in hardware-aware implementation: row-major access throughout, a skew-symmetric matrix formed from half its entries, a linear solve with no inverse, and no allocation inside a step. The reasoning is in [docs/design.md](docs/design.md).

This repository currently holds the design and documentation only. Every source file is a placeholder with a single comment, and nothing has been built, run or measured.

## The update

W is an orthogonal n × n matrix, G is the Euclidean gradient of a loss with respect to W, and η is the step size.

$$A = G W^\top - W G^\top$$

$$W^{+} = \left(I + \tfrac{\eta}{2}A\right)^{-1}\left(I - \tfrac{\eta}{2}A\right)W$$

A is skew-symmetric, and that is what keeps W⁺ orthogonal. The proof is in [docs/derivation.md](docs/derivation.md). It holds in exact arithmetic, so how far rounding error pulls W⁺ from orthogonality is one of the things to be measured.

Each step solves (I + (η/2)A) W⁺ = (I − (η/2)A) W by Gaussian elimination. No inverse is formed, and all scratch memory is allocated once, before the first step.

## Demonstration

Orthogonal Procrustes: given X and Y, find the orthogonal W that minimises ‖WX − Y‖²_F. The exact solution comes from an SVD in NumPy. The C code runs the iterations, and the aim is to see the loss fall to the optimum while W stays orthogonal. The setup is in [docs/validation.md](docs/validation.md). No run has been made.

## Layout

| Path | Purpose |
| --- | --- |
| `include/`, `src/` | Header and implementation of the update |
| `tests/` | Correctness tests |
| `reference/` | NumPy reference of the update, and the script that writes test fixtures |
| `examples/` | Procrustes driver in C, and the script that writes its data and exact solution |
| `bench/` | Timing drivers for C and NumPy |
| `docs/derivation.md` | Why the update preserves orthogonality |
| `docs/design.md` | Algorithm, memory plan and numerical choices |
| `docs/validation.md` | How correctness, speed and the demonstration will be checked |

## Progress

- [x] Derivation
- [ ] NumPy reference
- [ ] Fixtures
- [ ] Implementation
- [ ] Tests against the fixtures
- [ ] Makefile
- [ ] Procrustes demonstration
- [ ] Benchmark against NumPy

## Licence

MIT. See [LICENSE](LICENSE).
