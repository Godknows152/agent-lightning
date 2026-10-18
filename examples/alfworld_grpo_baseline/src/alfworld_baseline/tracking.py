"""ALFWorld-local SwanLab identity persistence around native veRL tracking."""
from contextlib import contextmanager
import json
from pathlib import Path
import uuid


def finish_tracking(tracking, *, error: BaseException | None = None) -> None:
    """Preserve failures when finishing native veRL's SwanLab backend.

    Native tracking forwards exit codes to some backends, but calls SwanLab
    without its terminal state. Finish that backend explicitly and let native
    tracking close the remaining backends exactly once.
    """
    if getattr(tracking, "_finished", False):
        return
    swanlab = getattr(tracking, "logger", {}).pop("swanlab", None)
    try:
        if swanlab is not None:
            if error is None:
                swanlab.finish(state="success")
            elif isinstance(error, (KeyboardInterrupt, SystemExit)):
                swanlab.finish(state="aborted")
            else:
                swanlab.finish(state="crashed", error=f"{type(error).__name__}: {error}")
    finally:
        tracking.finish(exit_code=0 if error is None else 1)


@contextmanager
def swanlab_resume(config):
    """Pass an explicit run identity without changing native trainer code.

    Scoped to this task runner process. Persist the ID immediately after init,
    so exceptions during training cannot silently create a replacement run.
    SWANLAB_NEW_RUN_ON_RESUME explicitly starts a same-name run at the restored
    step, archiving the old identity; ordinary resumes retain the current ID.
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
    new_run = bool(config.get("variables", {}).get("SWANLAB_NEW_RUN_ON_RESUME", False))
    continuing = mode == "resume_path" or (mode == "auto" and (root / "latest_checkpointed_iteration.txt").exists())
    if new_run and not continuing:
        raise ValueError("A new SwanLab run on resume requires an existing checkpoint")
    if continuing and identity is None:
        raise ValueError("Resuming a checkpoint requires its .swanlab_experiment.json")
    if identity is not None and (mode == "disable" or not continuing):
        raise ValueError("Existing SwanLab identity requires a published checkpoint; use a new output directory")
    if mode == "resume_path":
        source = Path(config.trainer.resume_from_path).parent / marker.name
        if not source.exists() or json.loads(source.read_text()) != identity:
            raise ValueError("Checkpoint and output directory must have the same SwanLab identity")
    original_init = swanlab.init
    expected_id = uuid.uuid4().hex[:8] if new_run else (identity["run_id"] if identity else None)

    def init(**kwargs):
        if expected_id:
            kwargs.update(id=expected_id, resume="never" if new_run else "must")
        run = original_init(**kwargs)
        run_id = getattr(run, "id", None)
        if not run_id or (expected_id and run_id != expected_id):
            swanlab.finish()
            raise RuntimeError("SwanLab did not return the requested run ID")
        root.mkdir(parents=True, exist_ok=True)
        if new_run:
            checkpoint = (
                Path(config.trainer.resume_from_path) if mode == "resume_path"
                else root / f"global_step_{int((root / 'latest_checkpointed_iteration.txt').read_text())}"
            )
            history = root / "swanlab_identity_history"
            history.mkdir(exist_ok=True)
            (history / f"{run_id}.json").write_text(json.dumps({
                "previous": identity, "current": {"experiment_name": name, "run_id": run_id},
                "resume_from_path": str(checkpoint),
            }, ensure_ascii=False, indent=2) + "\n")
        pending = marker.with_suffix(".json.tmp")
        pending.write_text(json.dumps({"experiment_name": name, "run_id": run_id}, ensure_ascii=False) + "\n")
        pending.replace(marker)
        if new_run:
            swanlab.log({"training/resumed_from_step": int(checkpoint.name.removeprefix("global_step_"))},
                        step=int(checkpoint.name.removeprefix("global_step_")))
        return run

    swanlab.init = init
    try:
        yield
    finally:
        swanlab.init = original_init
