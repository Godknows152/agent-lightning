# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations


def patch_transformers_vision2seq_alias() -> None:
    """Expose the Transformers 4.x multimodal auto class name on 5.x.

    VERL 0.6 still imports ``AutoModelForVision2Seq`` from ``transformers``.
    Transformers 5.x uses ``AutoModelForImageTextToText`` for the same
    multimodal auto-dispatch role needed by Qwen3.5. Install the old name
    before importing VERL modules, including inside Ray worker processes.
    """

    import sys

    import transformers

    try:
        from transformers.models.auto import modeling_auto
        from transformers.models.auto.modeling_auto import AutoModelForImageTextToText as replacement
    except ImportError:
        return

    def install_alias(transformers_module: object) -> None:
        import_structure = getattr(transformers_module, "_import_structure", None)
        if isinstance(import_structure, dict):
            modeling_auto_exports = import_structure.get("models.auto.modeling_auto")
            if isinstance(modeling_auto_exports, list) and "AutoModelForVision2Seq" not in modeling_auto_exports:
                modeling_auto_exports.append("AutoModelForVision2Seq")

        class_to_module = getattr(transformers_module, "_class_to_module", None)
        if isinstance(class_to_module, dict):
            class_to_module.setdefault("AutoModelForVision2Seq", "models.auto.modeling_auto")

        objects = getattr(transformers_module, "_objects", None)
        if isinstance(objects, dict):
            objects["AutoModelForVision2Seq"] = replacement

        all_exports = getattr(transformers_module, "__all__", None)
        if isinstance(all_exports, list) and "AutoModelForVision2Seq" not in all_exports:
            all_exports.append("AutoModelForVision2Seq")

        setattr(transformers_module, "AutoModelForVision2Seq", replacement)

    setattr(modeling_auto, "AutoModelForVision2Seq", replacement)
    install_alias(transformers)
    # VERL imports this module through ``verl.utils.dataset`` inside Ray
    # workers. Loading it once here avoids a Transformers lazy-export timing
    # issue where the top-level alias exists but VERL still observes the old
    # import table during its nested import.
    import verl.utils.model  # noqa: F401

    current_transformers = sys.modules.get("transformers")
    if current_transformers is not None:
        install_alias(current_transformers)
