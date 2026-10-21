# Accuracy

Source: https://pianistpy.com/docs/guide/accuracy

Every result carries a validation report, a fixed summary with one record per accuracy check of the result.
The tier of a result’s report, `report.tier`, is the band of the largest relative error over the result’s own checks
and over the results from which the result was computed:

- `"silent"`: at most \(10^{-6}\), the result is returned silently;

- `"recorded"`: above \(10^{-6}\) and at most \(10^{-4}\), the result is returned without a warning, and
the report records the error;

- `"warned"`: above \(10^{-4}\) and at most \(10^{-2}\), the result is returned with an accuracy warning.
The accuracy warning goes to the package’s log and is issued as an `AccuracyWarning`, which
Python’s standard filter can silence or escalate. A result computed from a warned result has the tier
`"warned"` or a higher tier;

- `"refused"`: above \(10^{-2}\), the call raises `AccuracyError`, unless
`Settings(write_anyway=True)` asks for the result anyway.

The package has two tested settings levels, the default settings and the quick settings:

```python
pp.Settings()            # the default settings
pp.Settings(quick=True)  # the quick settings, faster, for exploration
```

The runtime options (verbosity, timing, cores, compilation cache, platform) set how the package runs, and the
[Runtime options](https://pianistpy.com/docs/guide/runtime) page describes the runtime options. The runtime options change no computed number, except the platform. CPU-only and
GPU results differ by less than \(10^{-10}\) in relative terms.

- [Flags and refusals](https://pianistpy.com/docs/guide/flags-refusals)
