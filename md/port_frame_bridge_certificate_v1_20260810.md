# Port deterministic frame bridge certificate v1

- Status: `PORT_DETERMINISTIC_VIRTUAL_FRAME_BRIDGE_PASS`
- Lean compile exit code: `0`.
- Finite family: `31` candidates over `4` contexts.
- Exact generated-family oracle gap: `0.0`.
- Finite-family penalty bound: `P0=0.000645255`; `beta=0.0`.

| Context | Candidates | Observed duration range | Common H | Fits H |
|---|---:|---:|---:|---:|
| `public_bacasp_s` | 31 | [176.500000, 287.868750] | 671.310507 | true |
| `synthetic:balanced_heterogeneous_8` | 31 | [223.557958, 371.020605] | 2259.315796 | true |
| `synthetic:berth_crane_pressure_9` | 31 | [237.206180, 380.495731] | 2287.746366 | true |
| `synthetic:yard_gate_surge_9` | 31 | [282.614182, 392.812703] | 2825.838496 | true |

## Closed interface

Every registered trajectory fits a candidate-independent positive horizon and can be idle-padded to that horizon. Dividing the registered scalar virtual objective by the common horizon preserves ordering, and the referenced variable-duration Lean file compiles without sorry/admit/axiom.

## Open operational assumptions

The terminal coordinates are bounded virtual cost service, not physical departures. A physical-port recurrence claim still requires a stochastic vessel-arrival/service model, positive slack, and local return. This certificate does not infer those assumptions from finite replay.
