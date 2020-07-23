"""@file postprocessor_factory.py
contains the Postprocessor factory"""

from . import  ivector_extractor_matlab


def factory(postprocessor):
	"""
	gets a postprocessor class

	Args:
		postprocessor: the postprocessor type

	Returns:
		a postprocessor class
	"""

	if postprocessor == 'ivector_extractor':
		return ivector_extractor_matlab.IvectorExtractor
	else:
		raise Exception('Undefined postprocessor type: %s' % postprocessor)
