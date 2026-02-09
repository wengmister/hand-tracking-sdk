class HTSError(Exception):
    """Base exception for SDK errors."""


class ParseError(HTSError):
    """Raised when incoming HTS lines cannot be parsed."""
