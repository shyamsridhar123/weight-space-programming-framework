sha256sum lake-manifest.json > "$RUNNER_TEMP/wsp-lake-manifest.sha256"
lake exe cache get \
  Mathlib.Data.Real.Basic \
  Mathlib.Algebra.BigOperators.Fin \
  Mathlib.Algebra.Order.BigOperators.Group.Finset \
  Mathlib.Tactic.Ring \
  Mathlib.Tactic.Linarith \
  Mathlib.Tactic.NormNum
sha256sum --check "$RUNNER_TEMP/wsp-lake-manifest.sha256"
git diff --exit-code -- lake-manifest.json lean-toolchain lakefile.toml
