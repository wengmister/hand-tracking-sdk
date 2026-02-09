"""Custom exception hierarchy for SDK parsing and runtime errors."""

class HTSError(Exception):
    """Base exception for SDK errors."""


class ParseError(HTSError):
    """Raised when incoming HTS lines cannot be parsed."""
