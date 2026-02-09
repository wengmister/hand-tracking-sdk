"""Custom exception hierarchy for SDK parsing and runtime errors."""

class HTSError(Exception):
    """Base exception for SDK errors."""


class ParseError(HTSError):
    """Raised when incoming HTS lines cannot be parsed."""


class TransportError(HTSError):
    """Base exception for transport-level errors."""


class TransportClosedError(TransportError):
    """Raised when operating on a transport receiver that is not open."""


class TransportTimeoutError(TransportError):
    """Raised when waiting for network I/O exceeds configured timeout."""


class TransportDisconnectedError(TransportError):
    """Raised when a connected TCP peer disconnects."""
