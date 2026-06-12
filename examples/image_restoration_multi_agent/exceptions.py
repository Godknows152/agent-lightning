"""Domain exceptions used by the deterministic restoration workflow."""


class RestorationWorkflowError(RuntimeError):
    """Base exception for workflow failures."""


class InvalidToolCallError(RestorationWorkflowError):
    """Raised when an expert decision cannot be treated as one valid tool call."""


class UnknownActionError(InvalidToolCallError):
    """Raised when an action is absent from the shared tool registry."""


class WorkerExecutionError(RestorationWorkflowError):
    """Raised when a restoration worker cannot produce a valid output."""


class EvaluationError(RestorationWorkflowError):
    """Raised when image quality evaluation fails."""
