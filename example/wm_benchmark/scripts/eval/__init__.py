"""Local eval harness for the wm_benchmark.

Two entrypoints, one shared recorder:

- ``eval_wm.py``   — world model + planner (optimizes actions to reach a goal)
- ``eval_policy.py`` — direct GC-policy rollout (no action optimization)

Both leverage swm's ``world.evaluate(...)`` / ``world.evaluate_from_dataset(...)``
internally and add a local ``RolloutRecorder`` that dumps frames / actions /
timings / per-episode stats under ``data/eval_runs/<run_id>/``.

Planners live behind a uniform ``Planner`` protocol in ``planners.py``; an
``EbmifyGradientPlanner`` stub is the future integration point for
leverage-as-EBM (deferred per the migration plan).
"""
