# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-12-05 12:01:21
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-12-05 12:19:32

from __future__ import print_function, division, absolute_import


class PcasigError(Exception):
    """A custom core Pcasig exception"""

    def __init__(self, message=None):

        message = 'There has been an error' \
            if not message else message

        super(PcasigError, self).__init__(message)


class PcasigNotImplemented(PcasigError):
    """A custom exception for not yet implemented features."""

    def __init__(self, message=None):

        message = 'This feature is not implemented yet.' \
            if not message else message

        super(PcasigNotImplemented, self).__init__(message)


class PcasigAPIError(PcasigError):
    """A custom exception for API errors"""

    def __init__(self, message=None):
        if not message:
            message = 'Error with Http Response from Pcasig API'
        else:
            message = 'Http response error from Pcasig API. {0}'.format(message)

        super(PcasigAPIError, self).__init__(message)


class PcasigApiAuthError(PcasigAPIError):
    """A custom exception for API authentication errors"""
    pass


class PcasigMissingDependency(PcasigError):
    """A custom exception for missing dependencies."""
    pass


class PcasigWarning(Warning):
    """Base warning for Pcasig."""


class PcasigUserWarning(UserWarning, PcasigWarning):
    """The primary warning class."""
    pass


class PcasigSkippedTestWarning(PcasigUserWarning):
    """A warning for when a test is skipped."""
    pass


class PcasigDeprecationWarning(PcasigUserWarning):
    """A warning for deprecated features."""
    pass
