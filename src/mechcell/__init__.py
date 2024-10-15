#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2024
#
# This file is part of mechcell
# SPDX-License-Identifier:    MIT

import importlib.metadata

from .process import ProcessImage

__version__ = importlib.metadata.version(__package__)

__all__ = [ProcessImage]
