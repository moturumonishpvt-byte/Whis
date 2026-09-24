class WHISError(Exception):
    """Base exception for all WHIS errors."""
    pass

class ConfigurationError(WHISError):
    """Raised when there is a configuration error."""
    pass

class ToolExecutionError(WHISError):
    """Raised when a tool execution fails."""
    pass
