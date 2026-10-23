# Numerical playground

Runnable micro, macro, and econometrics simulations that sit next to the Lean library.

MathEconLib currently machine-checks microeconomic theory: games, mechanisms, auctions, matching, and pricing. Macroeconomics and econometrics are part of the project's long-term scope, and they are not formalized here yet. The Python in this folder is ordinary numerical code. It is not checked by the Lean kernel. Where a picture restates a theorem, the docstring names the Lean declaration.

## Run

From the repository root:

```bash
python3 -m pip install -r simulations/requirements.txt
python3 -m unittest simulations.tests
python3 -m simulations
```

Figures land in `simulations/output/`. Pass `--out` to write them somewhere else.

## What is in there

| Module | Simulation | Formal counterpart |
| --- | --- | --- |
| `micro.py` | Posted price for uniform buyer values. On `[0, 1]` with zero continuation value the optimum is `1/2` and expected profit is `1/4`. | `ValueDistribution.uniformOptimalPrice` |
| `micro.py` | Becker crime: offend when the net gain clears the expected sanction `p * f`. | `MathEconLibExamples/SingleParameter/BeckerCrime.lean` |
| `micro.py` | Fictitious play in matching pennies. Frequencies drift to the unique mixed Nash, fifty-fifty. | `matchingPennies_uniform_is_mixed_nash` |
| `micro.py` | Gale-Shapley deferred acceptance. Applicant-proposing and department-proposing runs produce different stable matches. | `MarriageMarket.galeShapleyFinset` |
| `micro.py` | Cournot best-response dynamics. | numerical only |
| `macro.py` | Solow capital accumulation and the golden-rule saving rate. | numerical only |
| `macro.py` | Three-equation New Keynesian impulse responses to a policy shock and a demand shock. | numerical only |
| `econometrics.py` | OLS sampling distribution, omitted-variable bias, 2SLS, and a sharp regression discontinuity. | numerical only |

## Lean

The formal library is the rest of this repository. After [installing Lean](https://lean-lang.org/install/), from the repository root:

```bash
lake exe cache get
lake build
```

`lean-toolchain` pins `leanprover/lean4:v4.28.0`. Open the repository folder, not a single file, so the toolchain file is picked up. A local scratch file belongs in `MathEconLib/Playground/`, which git ignores.
