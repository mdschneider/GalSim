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
import galsim

valid_value_types = {
    # The values are tuples with:
    # - the build function to call
    # - a list of types for which the type is valid
    'List' : ('_GenerateFromList', 
              [ float, int, bool, str, galsim.Angle, galsim.Shear, galsim.PositionD ]),
    'Eval' : ('_GenerateFromEval', 
              [ float, int, bool, str, galsim.Angle, galsim.Shear, galsim.PositionD ]),
    'Current' : ('_GenerateFromCurrent', 
                 [ float, int, bool, str, galsim.Angle, galsim.Shear, galsim.PositionD ]),
    'Sum' : ('_GenerateFromSum', 
             [ float, int, galsim.Angle, galsim.Shear, galsim.PositionD ]),
    'Catalog' : ('_GenerateFromCatalog', [ float, int, bool, str ]),
    'Dict' : ('_GenerateFromDict', [ float, int, bool, str ]),
    'FitsHeader' : ('_GenerateFromFitsHeader', [ float, int, bool, str ]),
    'Sequence' : ('_GenerateFromSequence', [ float, int, bool ]),
    'Random' : ('_GenerateFromRandom', [ float, int, bool, galsim.Angle ]),
    'RandomGaussian' : ('_GenerateFromRandomGaussian', [ float ]),
    'RandomDistribution' : ('_GenerateFromRandomDistribution', [ float ]),
    'RandomCircle' : ('_GenerateFromRandomCircle', [ galsim.PositionD ]),
    'NumberedFile' : ('_GenerateFromNumberedFile', [ str ]),
    'FormattedStr' : ('_GenerateFromFormattedStr', [ str ]),
    'Rad' : ('_GenerateFromRad', [ galsim.Angle ]),
    'Radians' : ('_GenerateFromRad', [ galsim.Angle ]),
    'Deg' : ('_GenerateFromDeg', [ galsim.Angle ]),
    'Degrees' : ('_GenerateFromDeg', [ galsim.Angle ]),
    'E1E2' : ('_GenerateFromE1E2', [ galsim.Shear ]),
    'EBeta' : ('_GenerateFromEBeta', [ galsim.Shear ]),
    'G1G2' : ('_GenerateFromG1G2', [ galsim.Shear ]),
    'GBeta' : ('_GenerateFromGBeta', [ galsim.Shear ]),
    'Eta1Eta2' : ('_GenerateFromEta1Eta2', [ galsim.Shear ]),
    'EtaBeta' : ('_GenerateFromEtaBeta', [ galsim.Shear ]),
    'QBeta' : ('_GenerateFromQBeta', [ galsim.Shear ]),
    'XY' : ('_GenerateFromXY', [ galsim.PositionD ]),
    'RTheta' : ('_GenerateFromRTheta', [ galsim.PositionD ]),
    'NFWHaloShear' : ('_GenerateFromNFWHaloShear', [ galsim.Shear ]),
    'NFWHaloMagnification' : ('_GenerateFromNFWHaloMagnification', [ float ]),
    'PowerSpectrumShear' : ('_GenerateFromPowerSpectrumShear', [ galsim.Shear ]),
    'PowerSpectrumMagnification' : ('_GenerateFromPowerSpectrumMagnification', [ float ]),
}
 
# Standard keys to ignore while parsing values:
standard_ignore = [ 
    'type',
    'current_val', 'current_safe', 'current_value_type',
    'current_obj_num', 'current_image_num', 'current_file_num',
    '#' # When we read in json files, there represent comments
]

def ParseValue(config, param_name, base, value_type):
    """@brief Read or generate a parameter value from config.

    @returns the tuple (value, safe).
    """
    param = config[param_name]
    #print 'ParseValue for param_name = ',param_name,', value_type = ',str(value_type)
    #print 'param = ',param
    #print 'nums = ',base.get('file_num',0), base.get('image_num',0), base.get('obj_num',0)

    # First see if we can assign by param by a direct constant value
    if isinstance(param, value_type):
        #print param_name,' = ',param
        return param, True
    elif not isinstance(param, dict):
        if value_type is galsim.Angle:
            # Angle is a special case.  Angles are specified with a final string to 
            # declare what unit to use.
            val = _GetAngleValue(param, param_name)
        elif value_type is bool:
            # For bool, we allow a few special string conversions
            val = _GetBoolValue(param, param_name)
        elif value_type is galsim.PositionD:
            # For PositionD, we allow a string of x,y
            val = _GetPositionValue(param, param_name)
        else:
            # Make sure strings are converted to float (or other type) if necessary.
            # In particular things like 1.e6 aren't converted to float automatically
            # by the yaml reader. (Although I think this is a bug.)
            val = value_type(param)
        # Save the converted type for next time.
        config[param_name] = val
        #print param_name,' = ',val
        return val, True
    elif 'type' not in param:
        raise AttributeError(
            "%s.type attribute required in config for non-constant parameter %s."%(
                param_name,param_name))
    elif ( 'current_val' in param 
           and param['current_obj_num'] == base.get('obj_num',0)
           and param['current_image_num'] == base.get('image_num',0)
           and param['current_file_num'] == base.get('file_num',0) ):
        if param['current_value_type'] != value_type:
            raise ValueError(
                "Attempt to parse %s multiple times with different value types"%param_name)
        #print base['obj_num'],'Using current value of ',param_name,' = ',param['current_val']
        return param['current_val'], param['current_safe']
    else:
        # Otherwise, we need to generate the value according to its type
        # (See valid_value_types defined at the top of the file.)

        type = param['type']
        #print 'type = ',type
        #print param['type'], value_type

        # First check if the value_type is valid.
        if type not in valid_value_types:
            raise AttributeError(
                "Unrecognized type = %s specified for parameter %s"%(type,param_name))
            
        if value_type not in valid_value_types[type][1]:
            raise AttributeError(
                "Invalid value_type = %s specified for parameter %s with type = %s."%(
                    value_type, param_name, type))

        generate_func = eval(valid_value_types[type][0])
        #print 'generate_func = ',generate_func
        val, safe = generate_func(param, param_name, base, value_type)
        #print 'returned val, safe = ',val,safe

        # Make sure we really got the right type back.  (Just in case...)
        if not isinstance(val,value_type):
            val = value_type(val)

        # Save the current value for possible use by the Current type
        param['current_val'] = val
        param['current_safe'] = safe
        param['current_value_type'] = value_type
        param['current_obj_num'] = base.get('obj_num',0)
        param['current_image_num'] = base.get('image_num',0)
        param['current_file_num'] = base.get('file_num',0)
        #print param_name,' = ',val
        return val, safe


def _GetAngleValue(param, param_name):
    """ @brief Convert a string consisting of a value and an angle unit into an Angle.
    """
    try :
        value, unit = param.rsplit(None,1)
        value = float(value)
        unit = galsim.angle.get_angle_unit(unit)
        return galsim.Angle(value, unit)
    except Exception as e:
        raise AttributeError("Unable to parse %s param = %s as an Angle."%(param_name,param))


def _GetPositionValue(param, param_name):
    """ @brief Convert a tuple or a string that looks like "a,b" into a galsim.PositionD.
    """
    try:
        x = float(param[0])
        y = float(param[1])
    except:
        try:
            x, y = param.split(',')
            x = float(x.strip())
            y = float(y.strip())
        except:
            raise AttributeError("Unable to parse %s param = %s as a PositionD."%(param_name,param))
    return galsim.PositionD(x,y)


def _GetBoolValue(param, param_name):
    """ @brief Convert a string to a bool
    """
    if isinstance(param,str):
        if param.strip().upper() in [ 'TRUE', 'YES' ]:
            return True
        elif param.strip().upper() in [ 'FALSE', 'NO' ]:
            return False
        else:
            try:
                val = bool(int(param))
                return val
            except:
                raise AttributeError("Unable to parse %s param = %s as a bool."%(param_name,param))
    else:
        try:
            val = bool(param)
            return val
        except:
            raise AttributeError("Unable to parse %s param = %s as a bool."%(param_name,param))


def CheckAllParams(param, param_name, req={}, opt={}, single=[], ignore=[]):
    """@brief Check that the parameters for a particular item are all valid
    
    @returns a dict, get, with get[key] = value_type for all keys to get.
    """
    get = {}
    valid_keys = req.keys() + opt.keys()
    # Check required items:
    for (key, value_type) in req.items():
        if key in param:
            get[key] = value_type
        else:
            raise AttributeError(
                "Attribute %s is required for %s.type = %s"%(key,param_name,param['type']))

    # Check optional items:
    for (key, value_type) in opt.items():
        if key in param:
            get[key] = value_type

    # Check items for which exacly 1 should be defined:
    for s in single: 
        if not s: # If no items in list, don't require one of them to be present.
            break
        valid_keys += s.keys()
        count = 0
        for (key, value_type) in s.items():
            if key in param:
                count += 1
                if count > 1:
                    raise AttributeError(
                        "Only one of the attributes %s is allowed for %s.type = %s"%(
                            s.keys(),param_name,param['type']))
                get[key] = value_type
        if count == 0:
            raise AttributeError(
                "One of the attributes %s is required for %s.type = %s"%(
                    s.keys(),param_name,param['type']))

    # Check that there aren't any extra keys in param aside from a few we expect:
    valid_keys += ignore
    valid_keys += standard_ignore
    for key in param.keys():
        # Generators are allowed to use item names that start with _, which we ignore here.
        if key not in valid_keys and not key.startswith('_'):
            raise AttributeError(
                "Unexpected attribute %s found for parameter %s"%(key,param_name))

    return get


def GetAllParams(param, param_name, base, req={}, opt={}, single=[], ignore=[]):
    """@brief Check and get all the parameters for a particular item

    @returns the tuple (kwargs, safe).
    """
    get = CheckAllParams(param,param_name,req,opt,single,ignore)
    kwargs = {}
    safe = True
    for (key, value_type) in sorted(get.items()):
        val, safe1 = ParseValue(param, key, base, value_type)
        safe = safe and safe1
        kwargs[key] = val
    # Just in case there are unicode strings.   python 2.6 doesn't like them in kwargs.
    kwargs = dict([(k.encode('utf-8'), v) for k,v in kwargs.iteritems()])
    return kwargs, safe


def GetCurrentValue(config, param_name):
    """@brief Return the current value of a parameter (either stored or a simple value)
    """
    param = config[param_name]
    if isinstance(param, dict):
        return param['current_val']
    else: 
        return param


#
# Now all the GenerateFrom functions:
#

def _GenerateFromG1G2(param, param_name, base, value_type):
    """@brief Return a Shear constructed from given (g1, g2)
    """
    req = { 'g1' : float, 'g2' : float }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    #print base['obj_num'],'Generate from G1G2: kwargs = ',kwargs
    return galsim.Shear(**kwargs), safe

def _GenerateFromE1E2(param, param_name, base, value_type):
    """@brief Return a Shear constructed from given (e1, e2)
    """
    req = { 'e1' : float, 'e2' : float }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    #print base['obj_num'],'Generate from E1E2: kwargs = ',kwargs
    return galsim.Shear(**kwargs), safe

def _GenerateFromEta1Eta2(param, param_name, base, value_type):
    """@brief Return a Shear constructed from given (eta1, eta2)
    """
    req = { 'eta1' : float, 'eta2' : float }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    #print base['obj_num'],'Generate from Eta1Eta2: kwargs = ',kwargs
    return galsim.Shear(**kwargs), safe

def _GenerateFromGBeta(param, param_name, base, value_type):
    """@brief Return a Shear constructed from given (g, beta)
    """
    req = { 'g' : float, 'beta' : galsim.Angle }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    #print base['obj_num'],'Generate from GBeta: kwargs = ',kwargs
    return galsim.Shear(**kwargs), safe

def _GenerateFromEBeta(param, param_name, base, value_type):
    """@brief Return a Shear constructed from given (e, beta)
    """
    req = { 'e' : float, 'beta' : galsim.Angle }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    #print base['obj_num'],'Generate from EBeta: kwargs = ',kwargs
    return galsim.Shear(**kwargs), safe

def _GenerateFromEtaBeta(param, param_name, base, value_type):
    """@brief Return a Shear constructed from given (eta, beta)
    """
    req = { 'eta' : float, 'beta' : galsim.Angle }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    #print base['obj_num'],'Generate from EtaBeta: kwargs = ',kwargs
    return galsim.Shear(**kwargs), safe

def _GenerateFromQBeta(param, param_name, base, value_type):
    """@brief Return a Shear constructed from given (q, beta)
    """
    req = { 'q' : float, 'beta' : galsim.Angle }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    #print base['obj_num'],'Generate from QBeta: kwargs = ',kwargs
    return galsim.Shear(**kwargs), safe

def _GenerateFromXY(param, param_name, base, value_type):
    """@brief Return a PositionD constructed from given (x,y)
    """
    req = { 'x' : float, 'y' : float }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    #print base['obj_num'],'Generate from XY: kwargs = ',kwargs
    return galsim.PositionD(**kwargs), safe

def _GenerateFromRTheta(param, param_name, base, value_type):
    """@brief Return a PositionD constructed from given (r,theta)
    """
    req = { 'r' : float, 'theta' : galsim.Angle }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    r = kwargs['r']
    theta = kwargs['theta']
    import math
    #print base['obj_num'],'Generate from RTheta: kwargs = ',kwargs
    return galsim.PositionD(r*math.cos(theta.rad()), r*math.sin(theta.rad())), safe

def _GenerateFromRad(param, param_name, base, value_type):
    """@brief Return an Angle constructed from given theta in radians
    """
    req = { 'theta' : float }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    #print base['obj_num'],'Generate from Rad: kwargs = ',kwargs
    return kwargs['theta'] * galsim.radians, safe

def _GenerateFromDeg(param, param_name, base, value_type):
    """@brief Return an Angle constructed from given theta in degrees
    """
    req = { 'theta' : float }
    kwargs, safe = GetAllParams(param, param_name, base, req=req)
    #print base['obj_num'],'Generate from Deg: kwargs = ',kwargs
    return kwargs['theta'] * galsim.degrees, safe

def _GenerateFromCatalog(param, param_name, base, value_type):
    """@brief Return a value read from an input catalog
    """
    if 'catalog' not in base:
        raise ValueError("No input catalog available for %s.type = Catalog"%param_name)

    if 'num' in param:
        num, safe = ParseValue(param, 'num', base, int)
    else:
        num, safe = (0, True)

    if num < 0:
        raise ValueError("Invalid num < 0 supplied for Catalog: num = %d"%num)
    if num >= len(base['catalog']):
        raise ValueError("Invalid num supplied for Catalog (too large): num = %d"%num)

    input_cat = base['catalog'][num]

    # Setup the indexing sequence if it hasn't been specified.
    # The normal thing with a Catalog is to just use each object in order,
    # so we don't require the user to specify that by hand.  We can do it for them.
    SetDefaultIndex(param, input_cat.getNObjects())

    # Coding note: the and/or bit is equivalent to a C ternary operator:
    #     input_cat.isFits() ? str : int
    # which of course doesn't exist in python.  This does the same thing (so long as the 
    # middle item evaluates to true).
    req = { 'col' : input_cat.isFits() and str or int , 'index' : int }
    kwargs, safe1 = GetAllParams(param, param_name, base, req=req, ignore=['num'])
    safe = safe and safe1

    if value_type is str:
        val = input_cat.get(**kwargs)
    elif value_type is float:
        val = input_cat.getFloat(**kwargs)
    elif value_type is int:
        val = input_cat.getInt(**kwargs)
    elif value_type is bool:
        val = _GetBoolValue(input_cat.get(**kwargs),param_name)

    #print base['file_num'],
    #print 'Catalog: col = %s, index = %s, val = %s'%(kwargs['col'],kwargs['index'],val)
    return val, safe


def _GenerateFromDict(param, param_name, base, value_type):
    """@brief Return a value read from an input dict.
    """
    if 'dict' not in base:
        raise ValueError("No input dict available for %s.type = Dict"%param_name)

    req = { 'key' : str }
    opt = { 'num' : int }
    kwargs, safe = GetAllParams(param, param_name, base, req=req, opt=opt)
    key = kwargs['key']

    num = kwargs.get('num',0)
    if num < 0:
        raise ValueError("Invalid num < 0 supplied for Dict: num = %d"%num)
    if num >= len(base['dict']):
        raise ValueError("Invalid num supplied for Dict (too large): num = %d"%num)
    d = base['dict'][num]

    val = d.get(key)
    #print base['file_num'],'Dict: key = %s, val = %s'%(key,val)
    return val, safe



def _GenerateFromFitsHeader(param, param_name, base, value_type):
    """@brief Return a value read from a FITS header
    """
    if 'fits_header' not in base:
        raise ValueError("No fits header available for %s.type = FitsHeader"%param_name)

    req = { 'key' : str }
    opt = { 'num' : int }
    kwargs, safe = GetAllParams(param, param_name, base, req=req, opt=opt)
    key = kwargs['key']

    num = kwargs.get('num',0)
    if num < 0:
        raise ValueError("Invalid num < 0 supplied for FitsHeader: num = %d"%num)
    if num >= len(base['fits_header']):
        raise ValueError("Invalid num supplied for FitsHeader (too large): num = %d"%num)
    header = base['fits_header'][num]

    if key not in header.keys():
        raise ValueError("key %s not found in the FITS header in %s"%(key,kwargs['file_name']))

    val = header.get(key)
    #print base['file_num'],'Header: key = %s, val = %s'%(key,val)
    return val, safe


def _GenerateFromRandom(param, param_name, base, value_type):
    """@brief Return a random value drawn from a uniform distribution
    """
    if 'rng' not in base:
        raise ValueError("No base['rng'] available for %s.type = Random"%param_name)
    rng = base['rng']
    ud = galsim.UniformDeviate(rng)

    # Each value_type works a bit differently:
    if value_type is galsim.Angle:
        import math
        CheckAllParams(param, param_name)
        val = ud() * 2 * math.pi * galsim.radians
        #print base['obj_num'],'Random angle = ',val
        return val, False
    elif value_type is bool:
        CheckAllParams(param, param_name)
        val = ud() < 0.5
        #print base['obj_num'],'Random bool = ',val
        return val, False
    else:
        ignore = [ 'default' ]
        req = { 'min' : value_type , 'max' : value_type }
        kwargs, safe = GetAllParams(param, param_name, base, req=req, ignore=ignore)

        min = kwargs['min']
        max = kwargs['max']

        if value_type is int:
            import math
            val = int(math.floor(ud() * (max-min+1))) + min
            # In case ud() == 1
            if val > max: val = max
        else:
            val = ud() * (max-min) + min

        #print base['obj_num'],'Random = ',val
        return val, False


def _GenerateFromRandomGaussian(param, param_name, base, value_type):
    """@brief Return a random value drawn from a Gaussian distribution
    """
    if 'rng' not in base:
        raise ValueError("No base['rng'] available for %s.type = RandomGaussian"%param_name)
    rng = base['rng']

    req = { 'sigma' : float }
    opt = { 'mean' : float, 'min' : float, 'max' : float }
    kwargs, safe = GetAllParams(param, param_name, base, req=req, opt=opt)

    sigma = kwargs['sigma']

    if 'gd' in base and base['current_gdsigma'] == sigma:
        # Minor subtlety here.  GaussianDeviate requires two random numbers to 
        # generate a single Gaussian deviate.  But then it gets a second 
        # deviate for free.  So it's more efficient to store gd than to make
        # a new one each time.  So check if we did that.
        gd = base['gd']
    else:
        # Otherwise, just go ahead and make a new one.
        gd = galsim.GaussianDeviate(rng,sigma=sigma)
        base['gd'] = gd
        base['current_gdsigma'] = sigma

    if 'min' in kwargs or 'max' in kwargs:
        # Clip at min/max.
        # However, special cases if min == mean or max == mean
        #  -- can use fabs to double the chances of falling in the range.
        mean = kwargs.get('mean',0.)
        min = kwargs.get('min',-float('inf'))
        max = kwargs.get('max',float('inf'))

        do_abs = False
        do_neg = False
        if min == mean:
            do_abs = True
            max -= mean
            min = -max
        elif max == mean:
            do_abs = True
            do_neg = True
            min -= mean
            max = -min
        else:
            min -= mean
            max -= mean
    
        # Emulate a do-while loop
        import math
        while True:
            val = gd()
            if do_abs: val = math.fabs(val)
            if val >= min and val <= max: break
        if do_neg: val = -val
        val += mean
    else:
        val = gd()
        if 'mean' in kwargs: val += kwargs['mean']

    #print base['obj_num'],'RandomGaussian: ',val
    return val, False


def _GenerateFromRandomDistribution(param, param_name, base, value_type):
    """@brief Return a random value drawn from a user-defined probability distribution
    """
    if 'rng' not in base:
        raise ValueError("No rng available for %s.type = RandomDistribution"%param_name)
    rng = base['rng']

    opt = {'function' : str, 'interpolant' : str, 'npoints' : int, 
           'x_min' : float, 'x_max' : float }
    kwargs, safe = GetAllParams(param, param_name, base, opt=opt)
    
    if '_distdev' in param:
        # The overhead for making a DistDeviate is large enough that we'd rather not do it every 
        # time, so first check if we've already made one:
        distdev = param['_distdev']
        if param['_distdev_kwargs'] != kwargs:
            distdev=galsim.DistDeviate(rng,**kwargs)
            param['_distdev'] = distdev
            param['_distdev_kwargs'] = kwargs
    else:
        # Otherwise, just go ahead and make a new one.
        distdev=galsim.DistDeviate(rng,**kwargs)
        param['_distdev'] = distdev
        param['_distdev_kwargs'] = kwargs

    # Typically, the rng will change between successive calls to this, so reset the 
    # seed.  (The other internal calculations don't need to be redone unless the rest of the
    # kwargs have been changed.)
    distdev.reset(rng)

    val = distdev()
    #print base['obj_num'],'distdev = ',val
    return val, False


def _GenerateFromRandomCircle(param, param_name, base, value_type):
    """@brief Return a PositionD drawn from a circular top hat distribution.
    """
    if 'rng' not in base:
        raise ValueError("No base['rng'] available for %s.type = RandomCircle"%param_name)
    rng = base['rng']

    req = { 'radius' : float }
    opt = { 'inner_radius' : float, 'center' : galsim.PositionD }
    kwargs, safe = GetAllParams(param, param_name, base, req=req, opt=opt)
    radius = kwargs['radius']

    ud = galsim.UniformDeviate(rng)
    max_rsq = radius**2
    if 'inner_radius' in kwargs:
        inner_radius = kwargs['inner_radius']
        min_rsq = inner_radius**2
    else:
        min_rsq = 0.
    # Emulate a do-while loop
    while True:
        x = (2*ud()-1) * radius
        y = (2*ud()-1) * radius
        rsq = x**2 + y**2
        if rsq >= min_rsq and rsq <= max_rsq: break

    pos = galsim.PositionD(x,y)
    if 'center' in kwargs:
        pos += kwargs['center']

    #print base['obj_num'],'RandomCircle: ',pos
    return pos, False


def _GenerateFromSequence(param, param_name, base, value_type):
    """@brief Return next in a sequence of integers
    """
    ignore = [ 'default' ]
    opt = { 'first' : value_type, 'last' : value_type, 'step' : value_type,
            'repeat' : int, 'nitems' : int, 'index_key' : str }
    kwargs, safe = GetAllParams(param, param_name, base, opt=opt, ignore=ignore)

    step = kwargs.get('step',1)
    first = kwargs.get('first',0)
    repeat = kwargs.get('repeat',1)
    last = kwargs.get('last',None)
    nitems = kwargs.get('nitems',None)
    index_key = kwargs.get('index_key',base.get('index_key','obj_num'))
    if repeat <= 0:
        raise ValueError(
            "Invalid repeat=%d (must be > 0) for %s.type = Sequence"%(repeat,param_name))
    if last is not None and nitems is not None:
        raise AttributeError(
            "At most one of the attributes last and nitems is allowed for %s.type = Sequence"%(
                param_name))
    if index_key not in [ 'obj_num_in_file', 'obj_num', 'image_num', 'file_num' ]:
        raise AttributeError(
            "Invalid index=%s for %s.type = Sequence."%(index_key,param_name))

    if value_type is bool:
        # Then there are only really two valid sequences: Either 010101... or 101010...
        # Aside from the repeat value of course.
        if first:
            first = 1
            step = -1
            nitems = 2
        else:
            first = 0
            step = 1
            nitems = 2

    elif value_type is float:
        if last is not None:
            nitems = int( (last-first)/step + 0.5 ) + 1
    else:
        if last is not None:
            nitems = (last - first)/step + 1

    if index_key == 'obj_num_in_file':
        k = base['obj_num'] - base.get('start_obj_num',0)
    else:
        k = base[index_key]
    k = k / repeat

    if nitems is not None and nitems > 0:
        k = k % nitems

    index = first + k*step
    #print base[index_key],'Sequence index = %s + %d*%s = %s'%(first,k,step,index)
    return index, False


def _GenerateFromNumberedFile(param, param_name, base, value_type):
    """@brief Return a file_name using a root, a number, and an extension
    """
    if 'num' not in param:
        param['num'] = { 'type' : 'Sequence' }
    req = { 'root' : str , 'num' : int }
    opt = { 'ext' : str , 'digits' : int }
    kwargs, safe = GetAllParams(param, param_name, base, req=req, opt=opt)

    template = kwargs['root']
    if 'digits' in kwargs:
        template += '%%0%dd'%kwargs['digits']
    else:
        template += '%d'
    if 'ext' in kwargs:
        template += kwargs['ext']
    s = eval("'%s'%%%d"%(template,kwargs['num']))
    #print base['obj_num'],'NumberedFile = ',s
    return s, safe

def _GenerateFromFormattedStr(param, param_name, base, value_type):
    """@brief Create a string from a format string
    """
    req = { 'format' : str }
    # Ignore items for now, we'll deal with it differently.
    ignore = [ 'items' ]
    params, safe = GetAllParams(param, param_name, base, req=req, ignore=ignore)
    format = params['format']

    # Check that items is present and is a list.
    if 'items' not in param:
        raise AttributeError("Attribute items is required for %s.type = FormattedStr"%param_name)
    items = param['items']
    if not isinstance(items,list):
        raise AttributeError("items entry for parameter %s is not a list."%param_name)

    # Figure out what types we are expecting for the list elements:
    tokens = format.split('%')
    val_types = []
    skip = False 
    for token in tokens[1:]:  # skip first one.
        # It we have set skip, then skip this one.
        if skip:
            skip = False
            continue
        # If token == '', then this is a %% in the original string.  Skip this and the next token.
        if len(token) == 0:
            skip = True
            continue
        token = token.lstrip('0123456789lLh') # ignore field size, and long/short specification
        if len(token) == 0:
            raise ValueError("Unable to parse '%s' as a valid format string"%format)
        if token[0].lower() in 'diouxX':
            val_types.append(int)
        elif token[0].lower() in 'eEfFgG':
            val_types.append(float)
        elif token[0].lower() in 'rs':
            val_types.append(str)
        else:
            raise ValueError("Unable to parse '%s' as a valid format string"%format)

    if len(val_types) != len(items):
        raise ValueError(
            "Number of items for FormatStr (%d) does not match number expected from "%len(items)+
            "format string (%d)"%len(val_types))
    vals = []
    for index in range(len(items)):
        val, safe1 = ParseValue(items, index, base, val_types[index])
        safe = safe and safe1
        vals.append(val)

    final_str = format%tuple(vals)
    #print base['obj_num'],'FormattedStr = ',final_str
    return final_str, safe


def _GenerateFromNFWHaloShear(param, param_name, base, value_type):
    """@brief Return a shear calculated from an NFWHalo object.
    """
    if 'world_pos' not in base:
        raise ValueError("NFWHaloShear requested, but no position defined.")
    pos = base['world_pos']

    if 'gal' not in base or 'redshift' not in base['gal']:
        raise ValueError("NFWHaloShear requested, but no gal.redshift defined.")
    redshift = GetCurrentValue(base['gal'],'redshift')

    if 'nfw_halo' not in base:
        raise ValueError("NFWHaloShear requested, but no input.nfw_halo defined.")

    opt = { 'num' : int }
    kwargs = GetAllParams(param, param_name, base, opt=opt)[0]

    num = kwargs.get('num',0)
    if num < 0:
        raise ValueError("Invalid num < 0 supplied for NFWHalowShear: num = %d"%num)
    if num >= len(base['nfw_halo']):
        raise ValueError("Invalid num supplied for NFWHaloShear (too large): num = %d"%num)
    nfw_halo = base['nfw_halo'][num]

    try:
        g1,g2 = nfw_halo.getShear(pos,redshift)
        shear = galsim.Shear(g1=g1,g2=g2)
    except Exception as e:
        import warnings
        warnings.warn("Warning: NFWHalo shear is invalid -- probably strong lensing!  " +
                      "Using shear = 0.")
        shear = galsim.Shear(g1=0,g2=0)
    #print base['obj_num'],'NFW shear = ',shear
    return shear, False


def _GenerateFromNFWHaloMagnification(param, param_name, base, value_type):
    """@brief Return a magnification calculated from an NFWHalo object.
    """
    if 'world_pos' not in base:
        raise ValueError("NFWHaloMagnification requested, but no position defined.")
    pos = base['world_pos']

    if 'gal' not in base or 'redshift' not in base['gal']:
        raise ValueError("NFWHaloMagnification requested, but no gal.redshift defined.")
    redshift = GetCurrentValue(base['gal'],'redshift')

    if 'nfw_halo' not in base:
        raise ValueError("NFWHaloMagnification requested, but no input.nfw_halo defined.")
 
    opt = { 'max_mu' : float, 'num' : int }
    kwargs = GetAllParams(param, param_name, base, opt=opt)[0]

    num = kwargs.get('num',0)
    if num < 0:
        raise ValueError("Invalid num < 0 supplied for NFWHaloMagnification: num = %d"%num)
    if num >= len(base['nfw_halo']):
        raise ValueError("Invalid num supplied for NFWHaloMagnification (too large): num = %d"%num)
    nfw_halo = base['nfw_halo'][num]

    mu = nfw_halo.getMagnification(pos,redshift)

    max_mu = kwargs.get('max_mu', 25.)
    if not max_mu > 0.: 
        raise ValueError(
            "Invalid max_mu=%f (must be > 0) for %s.type = NFWHaloMagnification"%(
                max_mu,param_name))

    if mu < 0 or mu > max_mu:
        import warnings
        warnings.warn("Warning: NFWHalo mu = %f means strong lensing!  Using mu=%f"%(mu,max_mu))
        mu = max_mu

    #print base['obj_num'],'NFW mu = ',mu
    return mu, False


def _GenerateFromPowerSpectrumShear(param, param_name, base, value_type):
    """@brief Return a shear calculated from a PowerSpectrum object.
    """
    if 'world_pos' not in base:
        raise ValueError("PowerSpectrumShear requested, but no position defined.")
    pos = base['world_pos']

    if 'power_spectrum' not in base:
        raise ValueError("PowerSpectrumShear requested, but no input.power_spectrum defined.")
    
    opt = { 'num' : int }
    kwargs = GetAllParams(param, param_name, base, opt=opt)[0]

    num = kwargs.get('num',0)
    if num < 0:
        raise ValueError("Invalid num < 0 supplied for PowerSpectrumShear: num = %d"%num)
    if num >= len(base['power_spectrum']):
        raise ValueError("Invalid num supplied for PowerSpectrumShear (too large): num = %d"%num)
    power_spectrum = base['power_spectrum'][num]

    try:
        g1,g2 = power_spectrum.getShear(pos)
        shear = galsim.Shear(g1=g1,g2=g2)
    except Exception as e:
        import warnings
        warnings.warn("Warning: PowerSpectrum shear is invalid -- probably strong lensing!  " +
                      "Using shear = 0.")
        shear = galsim.Shear(g1=0,g2=0)
    #print base['obj_num'],'PS shear = ',shear
    return shear, False

def _GenerateFromPowerSpectrumMagnification(param, param_name, base, value_type):
    """@brief Return a magnification calculated from a PowerSpectrum object.
    """
    if 'world_pos' not in base:
        raise ValueError("PowerSpectrumMagnification requested, but no position defined.")
    pos = base['world_pos']

    if 'power_spectrum' not in base:
        raise ValueError("PowerSpectrumMagnification requested, but no input.power_spectrum "
                         "defined.")

    opt = { 'max_mu' : float, 'num' : int }
    kwargs = GetAllParams(param, param_name, base, opt=opt)[0]

    num = kwargs.get('num',0)
    if num < 0:
        raise ValueError("Invalid num < 0 supplied for PowerSpectrumMagnification: num = %d"%num)
    if num >= len(base['power_spectrum']):
        raise ValueError(
            "Invalid num supplied for PowerSpectrumMagnification (too large): num = %d"%num)
    power_spectrum = base['power_spectrum'][num]

    mu = power_spectrum.getMagnification(pos)

    max_mu = kwargs.get('max_mu', 25.)
    if not max_mu > 0.: 
        raise ValueError(
            "Invalid max_mu=%f (must be > 0) for %s.type = PowerSpectrumMagnification"%(
                max_mu,param_name))

    if mu < 0 or mu > max_mu:
        import warnings
        warnings.warn("Warning: PowerSpectrum mu = %f means strong lensing!  Using mu=%f"%(
            mu,max_mu))
        mu = max_mu
    #print base['obj_num'],'PS mu = ',mu
    return mu, False

def _GenerateFromList(param, param_name, base, value_type):
    """@brief Return next item from a provided list
    """
    req = { 'items' : list }
    opt = { 'index' : int }
    # Only Check, not Get.  We need to handle items a bit differently, since it's a list.
    CheckAllParams(param, param_name, req=req, opt=opt)
    items = param['items']
    if not isinstance(items,list):
        raise AttributeError("items entry for parameter %s is not a list."%param_name)

    # Setup the indexing sequence if it hasn't been specified using the length of items.
    SetDefaultIndex(param, len(items))
    index, safe = ParseValue(param, 'index', base, int)

    if index < 0 or index >= len(items):
        raise AttributeError("index %d out of bounds for parameter %s"%(index,param_name))
    val, safe1 = ParseValue(items, index, base, value_type)
    safe = safe and safe1
    #print base['obj_num'],'List index = %d, val = %s'%(index,val)
    return val, safe
 
def _GenerateFromSum(param, param_name, base, value_type):
    """@brief Return next item from a provided list
    """
    req = { 'items' : list }
    # Only Check, not Get.  We need to handle items a bit differently, since it's a list.
    CheckAllParams(param, param_name, req=req)
    items = param['items']
    if not isinstance(items,list):
        raise AttributeError("items entry for parameter %s is not a list."%param_name)

    sum, safe = ParseValue(items, 0, base, value_type)

    for k in range(1,len(items)):
        val, safe1 = ParseValue(items, k, base, value_type)
        sum += val
        safe = safe and safe1
        
    return sum, safe
 
def _type_by_letter(key):
    if len(key) < 2:
        raise AttributeError("Invalid user-defined variable %r"%key)
    if key[0] == 'f':
        return float
    elif key[0] == 'i':
        return int
    elif key[0] == 'b':
        return bool
    elif key[0] == 's':
        return str
    elif key[0] == 'a':
        return galsim.Angle
    elif key[0] == 'p':
        return galsim.PositionD
    elif key[0] == 'g':
        return galsim.Shear
    else:
        raise AttributeError("Invalid Eval variable: %s (starts with an invalid letter)"%key)

def _GenerateFromEval(param, param_name, base, value_type):
    """@brief Evaluate a string as the provided type
    """
    #print 'Start Eval for ',param_name
    req = { 'str' : str }
    opt = {}
    ignore = standard_ignore
    for key in param.keys():
        if key not in (ignore + req.keys()):
            opt[key] = _type_by_letter(key)
    #print 'opt = ',opt
    #print 'base has ',base.keys()
            
    params, safe = GetAllParams(param, param_name, base, req=req, opt=opt, ignore=ignore)
    #print 'params = ',params
    string = params['str']
    #print 'string = ',string

    # Bring the user-defined variables into scope.
    for key in opt.keys():
        exec(key[1:] + ' = params[key]')
        #print key[1:],'=',eval(key[1:])

    # Also bring in any top level eval_variables
    if 'eval_variables' in base:
        #print 'found eval_variables = ',base['eval_variables']
        if not isinstance(base['eval_variables'],dict):
            raise AttributeError("eval_variables must be a dict")
        opt = {}
        for key in base['eval_variables'].keys():
            if key not in ignore:
                opt[key] = _type_by_letter(key)
        #print 'opt = ',opt
        params, safe1 = GetAllParams(base['eval_variables'], 'eval_variables', base, opt=opt,
                                     ignore=ignore)
        #print 'params = ',params
        safe = safe and safe1
        for key in opt.keys():
            exec(key[1:] + ' = params[key]')
            #print key[1:],'=',eval(key[1:])

    # Also, we allow the use of math functions
    import math
    import numpy
    import os

    # Try evaluating the string as is.
    try:
        val = value_type(eval(string))
        #print base['obj_num'],'Simple Eval(%s) = %s'%(string,val)
        return val, safe
    except:
        pass

    # Then try bringing in the allowed variables to see if that works:
    if 'image_pos' in base:
        image_pos = base['image_pos']
    if 'world_pos' in base:
        world_pos = base['world_pos']
    if 'image_center' in base:
        image_center = base['image_center']
    if 'image_origin' in base:
        image_origin = base['image_origin']
    if 'image_xsize' in base:
        image_xsize = base['image_xsize']
    if 'image_ysize' in base:
        image_ysize = base['image_ysize']
    if 'stamp_xsize' in base:
        stamp_xsize = base['stamp_xsize']
    if 'stamp_ysize' in base:
        stamp_ysize = base['stamp_ysize']
    if 'pixel_scale' in base:
        pixel_sclae = base['pixel_scale']
    if 'rng' in base:
        rng = base['rng']
    if 'file_num' in base:
        file_num = base.get('file_num',0)
    if 'image_num' in base:
        image_num = base.get('image_num',0)
    if 'obj_num' in base:
        obj_num = base['obj_num']
    if 'start_obj_num' in base:
        start_obj_num = base.get('start_obj_num',0)
    for key in galsim.config.valid_input_types.keys():
        if key in base:
            exec(key + ' = base[key]')
    try:
        val = value_type(eval(string))
        #print base['obj_num'],'Eval(%s) needed extra variables: val = %s'%(string,val)
        return val, False
    except:
        raise ValueError("Unable to evaluate string %r as a %s for %s"%(
                string,value_type,param_name))

def _GenerateFromCurrent(param, param_name, base, value_type):
    """@brief Get the current value of another config item.
    """
    req = { 'key' : str }
    params, safe = GetAllParams(param, param_name, base, req=req)

    key = params['key']

    # This next bit is basically identical to the code for Dict.get(key) in catalog.py.
    # Make a list of keys
    chain = key.split('.')
    d = base

    # We may need to make one adjustment.  If the first item in the key is 'input', then
    # the key is probably wrong relative to the current config dict.  We make each input
    # item a list, so the user can have more than one input dict for example.  But if 
    # they aren't using that, we don't want them to have to know about it if they try to 
    # take soemthing from there for a Current item.  
    # So we change, e.g., 
    #     input.fits_header.file_name 
    # --> input.fits_header.0.file_name
    if chain[0] == 'input' and len(chain) > 2:
        try:
            k = int(chain[2])
        except:
            chain.insert(2,0)

    while len(chain):
        k = chain.pop(0)

        # Try to convert to an integer:
        try: k = int(k)
        except ValueError: pass

        if chain: 
            # If there are more keys, just set d to the next in the chanin.
            d = d[k]
        else: 
            # Otherwise, parse the value for this key
            val,safe = ParseValue(d, k, base, value_type)
            #print base['obj_num'],'Current key = %s, value = %s'%(key,val)
            return val,safe

    raise ValueError("Invalid key = %s given for %s.type = Current"%(key,param_name))


def SetDefaultIndex(config, num):
    """
    When the number of items in a list is known, we allow the user to omit some of 
    the parameters of a Sequence or Random and set them automatically based on the 
    size of the list, catalog, etc.
    """
    # We use a default item (set to True) to indicate that the value of nitems, last, or max
    # has been set here, rather than by the user.  This way if the number of items in the 
    # catalog changes from one file to the next, it will be update correctly to the new
    # number of catalog entries.

    if 'index' not in config:
        config['index'] = {
            'type' : 'Sequence',
            'nitems' : num,
            'default' : True,
        }
    elif ( isinstance(config['index'],dict) 
           and 'type' in config['index'] ):
        index = config['index']
        type = index['type']
        if ( type == 'Sequence' 
             and 'nitems' in index 
             and 'default' in index ):
            index['nitems'] = num
            index['default'] = True
        elif ( type == 'Sequence' 
               and 'nitems' not in index
               and ('step' not in index or (isinstance(index['step'],int) and index['step'] > 0) )
               and ('last' not in index or 'default' in index) ):
            index['last'] = num-1
            index['default'] = True
        elif ( type == 'Sequence'
               and 'nitems' not in index
               and ('step' in index and (isinstance(index['step'],int) and index['step'] < 0) ) ):
            # Normally, the value of default doesn't matter.  Its presence is sufficient
            # to indicate True.  However, here we have three options.  
            # 1) first and last are both set by default
            # 2) first (only) is set by default
            # 3) last (only) is set by default
            # So set default to the option we are using, so we update with the correct method.
            if ( ('first' not in index and 'last' not in index)
                 or ('default' in index and index['default'] == 1) ):
                index['first'] = num-1
                index['last'] = 0
                index['default'] = 1
            elif ( 'first' not in index 
                   or ('default' in index and index['default'] == 2) ):
                index['first'] = num-1
                index['default'] = 2
            elif ( 'last' not in index 
                   or ('default' in index and index['default'] == 3) ):
                index['last'] = 0
                index['default'] = 3
        elif ( type == 'Random'
               and ('min' not in index or 'default' in index)
               and ('max' not in index or 'default' in index) ):
            index['min'] = 0
            index['max'] = num-1
            index['default'] = True

