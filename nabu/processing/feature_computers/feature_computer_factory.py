"""@file feature_computer_factory.py
contains the FeatureComputer factory"""

import mfcc
import fbank
import logspec
import magspec
import angspec
import spec
import raw
import frames


def factory(feature):
	"""
	create a FeatureComputer

	Args:
		feature: the feature computer type
	"""

	if feature == 'fbank':
		return fbank.Fbank
	elif feature == 'mfcc':
		return mfcc.Mfcc
	elif feature == 'logspec':
		return logspec.Logspec
	elif feature == 'magspec':
		return magspec.Magspec
	elif feature == 'powspec':
		return magspec.Powspec
	elif feature == 'angspec':
		return angspec.Angspec
	elif feature == 'spec':
		return spec.Spec
	elif feature == 'raw':
		return raw.Raw
	elif feature == 'frames':
		return frames.Frames
	else:
		raise Exception('Undefined feature type: %s' % feature)
