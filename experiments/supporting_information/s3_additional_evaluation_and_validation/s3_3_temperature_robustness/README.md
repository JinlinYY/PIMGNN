# SI Section S3.3: Temperature Robustness

Temperature robustness is evaluated at two complementary scales:

| Experiment | Question | Entry point |
| --- | --- | --- |
| [Local temperature perturbation](01_local_perturbation/) | How smoothly do predictions respond to finite temperature changes near observed conditions? | `scripts/analysis/run_sensitivity_analysis.py` |
| [Controlled temperature extrapolation](02_temperature_extrapolation/) | How accurately do matched PSMI models predict outside a restricted training interval, and does the temperature representation matter? | `scripts/run_temperature_encoding_sensitivity.py` and `scripts/plot_temperature_encoding_sensitivity.py` |

The local perturbation experiment probes the neighborhood of observed
conditions. The controlled extrapolation experiment instead withholds outer
temperature regions and uses system-disjoint partitions. Their metrics answer
different questions and should not be combined into a single robustness
claim.
