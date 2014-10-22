# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#
"""
The galsim.wfirst module, containing information GalSim needs to simulate images for the WFIRST-AFTA
project.

This module contains numbers and routines for the WFIRST-AFTA project.  Currently, it includes the
following numbers:

    gain - The gain for all SCAs (sensor chip arrays) is expected to be the same, so this is a
           single value rather than a list of values.

    pixel_scale - The pixel scale in units of arcsec/pixel.

    dark_current - The dark current in units of e-/s.

    exposure_time - The typical exposure time in units of seconds.  (Band?)

    sky_background - The typical sky background flux in e-/pix/s.  (Band?)

    read_noise - The value of read noise in electrons.  (Band?)

    effective_diameter - The effective telescope diameter in meters.

    exptime - The typical exposure time in units of seconds.

    ... also values related to nonlinearity, etc. ...

For example, to get the gain value, use galsim.wfirst.gain.

This module also contains the following routines:

    getBandpasses() - A utility to get a dictionary containing galsim.Bandpass objects for each of
                      the WFIRST-AFTA bandpasses.  For more detail, do
                      help(galsim.wfirst.getBandpasses).
"""

gain = 1.0
pixel_scale = 0.11
dark_current = 0.01
effective_diameter = 2.4
exptime = 168.1

from wfirst_bandpass import *