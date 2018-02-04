class ImageMonkeyAPIError(Exception):
	"""Base class for exceptions raised by ImageMonkey."""

class InternalImageMonkeyAPIError(ImageMonkeyAPIError):
	"""Exception for API errors."""