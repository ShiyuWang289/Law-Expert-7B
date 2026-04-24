# P4 Learning Rate Ablation Report

| learning rate | train_loss | eval_loss | train_runtime(s) | eval_runtime(s) |
|---:|---:|---:|---:|---:|
| 2e-4 | 1.1087 | 1.0682 | 77.8786 | 9.3164 |
| 1e-4 | 1.1219 | 1.0824 | 77.8653 | 9.3213 |
| 5e-5 | 1.1399 | 1.1025 | 77.8442 | 9.3142 |

**Recommended LR:** `2e-4` (lowest eval_loss=1.0682)
