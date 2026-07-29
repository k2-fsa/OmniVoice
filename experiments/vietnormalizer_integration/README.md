# VietNormalizer integration experiment

This experiment compares exactly four authorized conditions on the frozen
200-case pilot development/regression corpus: identity, the existing OmniVoice
research normalizer, official VietNormalizer 0.2.3, and the numeric-only fork.

The corpus has been inspected during rule development and is not a final
held-out evaluation set. The historical 160/40 split is therefore not used to
make generalization claims.

Run from the OmniVoice repository with its project environment:

```bash
.venv/bin/python -m experiments.vietnormalizer_integration.benchmark \
  --dataset experiments/tn_value_gate/data/vi_tn_pilot_v1.jsonl \
  --smoke-cases experiments/vietnormalizer_integration/smoke_cases.jsonl \
  --upstream-python /tmp/omnivoice_vi_benchmark_env/bin/python \
  --improved-python /tmp/vietnormalizer_numeric_env/bin/python \
  --improved-repo /home/pkh257/projects/vietnormalizer \
  --artifacts experiments/vietnormalizer_integration/artifacts
```

The interpreter paths are recorded in the environment artifact. A production
dependency is intentionally not pinned until the fork has been reviewed,
committed, pushed, and assigned an immutable revision.
