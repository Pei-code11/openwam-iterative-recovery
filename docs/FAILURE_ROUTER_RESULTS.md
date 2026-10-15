# Failure router results

The command below regenerates the recorded result:

```powershell
python scripts/run_failure_router_benchmark.py --output outputs/failure_router_benchmark
```

The run used 600 train episodes, 120 development episodes, and 300 episode-
disjoint held-out episodes. The model and local repair controller were frozen
before reading held-out outcomes. It is a custom synthetic OpenWAM recovery
stress test, not an official LIBERO score or a real-robot experiment.

| Method | Held-out success | Paired rescue | Paired harm | Clean false repair | False replan | Intervention rate |
|---|---:|---:|---:|---:|---:|---:|
| stale WAM | 23.7% | - | - | 0 | 0 | 0.0% |
| always local repair | 56.0% | 97 | 0 | 12 | 0 | 44.0% |
| always replan | 86.3% | 188 | 0 | 0 | 88 | 70.7% |
| rule hybrid | 100.0% | 229 | 0 | 0 | 0 | 76.3% |
| learned router | 100.0% | 229 | 0 | 0 | 22 | 83.0% |
| oracle route | 100.0% | 229 | 0 | 0 | 0 | 76.3% |

The learned router's source classification accuracy at the first injected event
was 75.6% on held-out episodes. This is a source-classification metric, not a
task success rate. Decision time measured in the Python/NumPy benchmark is
available in `outputs/failure_router_benchmark/results.json`; it excludes image
encoding, WAM inference, and simulator stepping.

The learned route selected a local-repair cost of `0.00`, WAM-replan cost of
`0.16`, contact-recovery cost of `0.02`, safe-stop cost of `0.03`, and an
uncertainty margin of `0.03` on development episodes. It achieved 80.7% source
classification accuracy at the first event on development and 75.6% on the
held-out set. The model has 5,899 parameters and runs with NumPy only.

The main result is a large gain over stale WAM on deliberately difficult
failure mixtures, while always-local and always-replan expose the two failure
modes of a one-size-fits-all intervention. The rule hybrid also reaches 100%,
so this run does **not** establish that the learned router is better than a
well-designed rule gate. The next evaluation should make the visual signals
harder, add held-out nuisance combinations, and compare latency and compute
against SAFE/VLA-FAIL/RAFAIL-style monitors.

## Interpretation by failure source

* Execution drift is routed to local repair when residual trend and direction
  agree, with WAM as a fallback when repair benefit is low.
* Target/world change and low WAM plan confidence are routed to WAM replan;
  local repair is not allowed to chase a moving target.
* Closed gripper plus weak contact is routed to contact recovery.
* High collision risk is routed to safe stop.
* Clean and distractor-only changes should continue; false repair and false
  replan are reported separately.

The result therefore supports the architecture as a testable routing layer,
not a claim that a small network has WAM's world knowledge.
