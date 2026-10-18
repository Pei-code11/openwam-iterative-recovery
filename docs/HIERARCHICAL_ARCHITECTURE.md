# Three-level WAM architecture

The proposed long-horizon system is a hierarchy rather than a single larger
action policy:

```text
large world model / planner
        |  task decomposition + memory
        v
short-horizon OpenWAM
        |  action chunks for one subtask
        v
time-aware repair head
        |  bounded XYZ/phase correction + verification
        v
robot execution
```

## Responsibilities

### Large planner

The large model receives the language task, scene state, and completed-subtask
memory. It returns short `SubtaskSpec` objects with a success condition and
expected event timeline. It may be an LLM/VLM or a larger WAM-like policy.
The public repository does not include such weights. `LongHorizonPlanner` is
therefore an adapter contract; `RuleBasedMacroPlanner` is only a deterministic
smoke-test fallback.

### Medium WAM

The existing OpenWAM service remains responsible for the short action chunk.
It is refreshed when the target scene changes, the current subtask is stale,
or the local repair validator fails. The middle layer does not need to solve
the complete long-horizon task.

### Small repair head

`TemporalRepairPolicy` adds the missing deadline check to the existing fast
repair path. It estimates whether a correction can finish and be validated
before the next `contact`, `grasp_close`, or `release` event. A world-change
signal always has priority over local repair. The local command is bounded and
must be validated before continuation.

## Current implementation status

The module supplies the interfaces and deterministic timing gate. It does not
claim to be a trained large model, and it does not replace OpenWAM weights.
The next experiment should connect the existing WAM RPC to
`MediumWAM.refresh`, provide event times from each WAM action chunk, and train
or fit the temporal repair head on injected position, velocity, and gripper
timing faults.

## Evaluation additions

In addition to task success and trajectory error, report:

* repair-before-event rate;
* contact and gripper timing error;
* repair-window misses;
* validated residual reduction;
* late-repair harm rate;
* target-change detection and replan latency;
* clean and distractor local-repair false-trigger rates.

