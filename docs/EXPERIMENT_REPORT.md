# Verified local execution report

Command: `python main.py`

Demo exit code: 0

```text
{
  "accuracy": 0.986013986013986,
  "f1": 0.9888888888888889,
  "auroc": 0.9976939203354298,
  "n_train": 426,
  "n_test": 143
}

```

Test exit code: 0

```text
test_drift (test_core.TestCore.test_drift) ... ok
test_import (test_core.TestCore.test_import) ... ok
test_training_and_inference (test_core.TestCore.test_training_and_inference) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.048s

OK

```

These are local baseline results on synthetic or tutorial data, not external validation.
