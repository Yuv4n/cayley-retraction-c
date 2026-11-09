# cayley-retraction-c

The weight update from my orthogonal RNN (LINK-TO-RNN-REPO), rewritten in C. The plan is to check it against the original NumPy code, then time the two against each other at several hidden sizes.

This repository currently holds the design and documentation only. Every source file is a placeholder with a single comment, and nothing has been built, run or measured.

## The update

W is an orthogonal n × n weight matrix, G is the gradient of the loss with respect to W, and η is the step size.

$$A = G W^\top - W G^\top$$

$$W^{+} = \left(I + \tfrac{\eta}{2}A\right)^{-1}\left(I - \tfrac{\eta}{2}A\right)W$$

A is skew-symmetric, and that is what keeps W⁺ orthogonal. The proof is in [docs/derivation.md](docs/derivation.md). It holds in exact arithmetic, so how far rounding error pulls W⁺ away from orthogonality over many steps is one of the things to be measured.

Each step solves the system (I + (η/2)A) W⁺ = (I − (η/2)A) W by Gaussian elimination, so no inverse is ever formed. All scratch memory is allocated once, before the first step.

## Layout

| Path | Purpose |
| --- | --- |
| `include/`, `src/` | Header and implementation |
| `tests/` | Correctness tests |
| `reference/` | The NumPy original and the script that makes test fixtures from it |
| `bench/` | Timing drivers for C and NumPy |
| `docs/derivation.md` | Why the update preserves orthogonality |
| `docs/design.md` | Algorithm, memory plan and numerical choices |
| `docs/validation.md` | How correctness and speed will be checked |

## Progress

- [x] Derivation
- [ ] Reference copied from the RNN repository
- [ ] Implementation
- [ ] Tests against the NumPy fixtures
- [ ] Benchmark against NumPy
- [ ] Build instructions, once there is something to build

## Licence

MIT. See [LICENSE](LICENSE).
