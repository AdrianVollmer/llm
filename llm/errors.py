class ModelError(Exception):
    "Models can raise this error, which will be displayed to the user"


class NeedsKeyException(ModelError):
    "Model needs an API key which has not been provided"

class FragmentNotFound(Exception):
    pass

class AttachmentError(Exception):
    """Exception raised for errors in attachment resolution."""

    pass

