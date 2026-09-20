"""ALFWorld-local SwanLab identity persistence around native veRL tracking."""
from contextlib import contextmanager
import json
from pathlib import Path


@contextmanager
def swanlab_resume(config):
    """Pass an explicit run identity without changing native trainer code.

    Scoped to this task runner process. Persist the ID immediately after init,
    so exceptions during training cannot silently create a replacement run.
    """
    if "swanlab" not in config.trainer.logger:
        yield
        return
    import swanlab

    root = Path(config.trainer.default_local_dir)
    marker = root / ".swanlab_experiment.json"
    identity = json.loads(marker.read_text()) if marker.exists() else None
    name = config.trainer.experiment_name
    if identity and (identity.get("experiment_name") != name or not identity.get("run_id")):
        raise ValueError("SwanLab run identity does not match the configured experiment")
    mode = config.trainer.resume_mode
    continuing = mode == "resume_path" or (mode == "auto" and (root / "latest_checkpointed_iteration.txt").exists())
    if continuing and identity is None:
        raise ValueError("Resuming a checkpoint requires its .swanlab_experiment.json")
    if identity is not None and (mode == "disable" or not continuing):
        raise ValueError("Existing SwanLab identity requires a published checkpoint; use a new output directory")
    if mode == "resume_path":
        source = Path(config.trainer.resume_from_path).parent / marker.name
        if not source.exists() or json.loads(source.read_text()) != identity:
            raise ValueError("Checkpoint and output directory must have the same SwanLab identity")
    original_init = swanlab.init

    def init(**kwargs):
        if identity:
            kwargs.update(id=identity["run_id"], resume="must")
        run = original_init(**kwargs)
        run_id = getattr(run, "id", None)
        if not run_id or (identity and run_id != identity["run_id"]):
            swanlab.finish()
            raise RuntimeError("SwanLab did not return the requested run ID")
        root.mkdir(parents=True, exist_ok=True)
        pending = marker.with_suffix(".json.tmp")
        pending.write_text(json.dumps({"experiment_name": name, "run_id": run_id}, ensure_ascii=False) + "\n")
        pending.replace(marker)
        return run

    swanlab.init = init
    try:
        yield
    finally:
        swanlab.init = original_init
