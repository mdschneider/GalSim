/* -*- c++ -*-
 * Copyright (c) 2012-2014 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

#ifndef SBSHAPELET_H
#define SBSHAPELET_H
/** 
 * @file SBShapelet.h @brief SBProfile that implements a polar shapelet profile 
 */

#include "SBProfile.h"
#include "Laguerre.h"

namespace galsim {

    /// @brief Class for describing polar shapelet surface brightness profiles.
    class SBShapelet : public SBProfile 
    {
    public:
        /** 
         * @brief Constructor.
         *
         * @param[in] sigma    scale size of Gauss-Laguerre basis set.
         * @param[in] bvec     `bvec[n,m]` contains flux information for the `(n, m)` basis
         *                     function.
         * @param[in] gsparams GSParams object storing constants that control the accuracy of image
         *                     operations and rendering, if different from the default.
         */
        SBShapelet(double sigma, LVector bvec, const GSParamsPtr& gsparams);

        /// @brief Copy Constructor. 
        SBShapelet(const SBShapelet& rhs);

        /// @brief Destructor. 
        ~SBShapelet();

        double getSigma() const;
        const LVector& getBVec() const;

    protected:
        class SBShapeletImpl;

    private:
        // op= is undefined
        void operator=(const SBShapelet& rhs);
    };

    template <typename T>
    void ShapeletFitImage(double sigma, LVector& bvec, const BaseImage<T>& image,
                          double image_scale, const Position<double>& center);
}

#endif // SBSHAPELET_H

