# Coding agent instructions for powertrace-sim

## General instructions

- Always refer to the README.md file for instructions on how to run the code.
- Before writing any code, always read the README.md file to understand the project structure and the codebase.
- After writing any code, always update the README.md file to reflect the changes.
- After writing any code, always run all tests using `uv run -m pytest -x` to ensure the code is working as expected.
- Never finish any task involving code without running all tests first or creating a new test to verify the code is working as expected.
- Strive to write code that is tight and concise, but not too dense that it's hard to understand.
- Strive to write code that is <500 lines of code per file and script.
- Default to making small changes and only making large changes when necessary.
- Default to not adding arguments to scripts and functions unless absolutely necessary.
- Do not use `try/except` to mask logic errors, missing invariants, or uncertain behavior.
- Use exceptions only at explicit process boundaries where failure is part of the interface.
- Prioritize code breaking and fixing any errors before adding new features.
- Write unsafe and correct code first, do not write defensive code first that evades failures.
- Write code that we can understand from its failures, not silent failures that are wrapped in try/except blocks or fallbacks.
- Never write code that is not correct or safe.
- Keep comments to a minimum, only use them when necessary to explain the code.
- Never use comments to explain the code, only use them to explain the intent of the code.
- Write the least amount of code necessary to achieve the desired functionality.
- If a function is used in multiple places, extract it to a separate file and import it.
- Ensure that naming conventions are consistent throughout the codebase.

## Definition of done

A task is not complete until all of the following are true:

- The code change is minimal and directly addresses the requested behavior.
- All relevant tests pass with `uv run -m pytest -x`.
- Any newly introduced behavior has a test, unless the task is purely documentation.
- `README.md` is updated if setup, outputs, assumptions, or commands changed.
- The agent has verified the primary user-visible result by running the relevant script or test.
- Dead code, obsolete branches, and superseded code paths created by the change are removed.
- The final report names the files changed, the commands run, and the observed outcome.

## Edit boundaries

- Only modify files directly relevant to the task.
- Do not rename or move files unless the task requires it.
- Do not rewrite large modules when a local fix is sufficient.
- Do not change result files, generated artifacts, or plots in `results/`, `outputs/`, or `artifacts/` unless explicitly asked.
- Do not modify environment, CI, dependency, or packaging files unless required for the task.
- Ask for a higher-confidence plan before changing public interfaces used in multiple scripts.

## Research reproducibility

- Prefer deterministic behavior when feasible.
- Do not introduce hidden randomness.
- If randomness is required, use explicit seeds and thread them through the relevant entry points.
- Do not change default experimental assumptions silently.
- If a change alters numerical results, metrics, or figure contents, state that clearly in the final report.
- Keep data processing and evaluation logic separable from plotting logic.
- Prefer scripts that can be rerun from the command line over notebook-only logic.

## Failure semantics

- Do not hide failures with fallback logic.
- Do not add permissive branches that guess missing inputs.
- Prefer explicit assertions and invariant checks at module boundaries.
- When behavior is ambiguous, inspect the calling code and existing tests before changing semantics.
- Fix root causes rather than patching symptoms.

## Compatibility and semantics

- Preserve existing CLI flags, function signatures, file formats, and output locations unless the task explicitly changes them.
- If an API rename is required, update all call sites in the same change.
- Do not keep dead compatibility layers unless explicitly requested.
- Prefer one clear code path over parallel old/new paths.

## Keep the codebase tight

- Remove obsolete helpers, unused flags, and superseded code introduced by the task.
- Do not create one-off utility files for single-use logic unless reuse is likely.
- Avoid adding new dependencies unless there is a strong justification.
- Prefer extending an existing coherent module over creating near-duplicate modules.
- Keep scripts and modules focused; split files before they become difficult to reason about.