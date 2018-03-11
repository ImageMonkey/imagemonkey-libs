class ImageMonkeyGeneralError(Exception):
	"""Base class for exceptions raised by ImageMonkey."""

class ImageMonkeyAPIError(ImageMonkeyGeneralError):
	"""Base class for exceptions raised by the ImageMonkey API."""

class InternalImageMonkeyAPIError(ImageMonkeyAPIError):
	"""Exception for API errors."""