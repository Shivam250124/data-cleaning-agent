class DatasetLoadError(Exception):
    """Raised when a required dataset CSV file is missing at startup."""


class ColumnNotFoundError(Exception):
    """Raised when an action references a column that does not exist in the DataFrame."""


class EpisodeDoneError(Exception):
    """Raised when a step is attempted after the episode has already ended."""
