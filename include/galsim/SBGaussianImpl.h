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

#ifndef SBGAUSSIAN_IMPL_H
#define SBGAUSSIAN_IMPL_H

#include "SBProfileImpl.h"
#include "SBGaussian.h"

namespace galsim {

    class SBGaussian::SBGaussianImpl : public SBProfileImpl
    {
    public:
        SBGaussianImpl(double sigma, double flux, const GSParamsPtr& gsparams);

        ~SBGaussianImpl() {}

        double xValue(const Position<double>& p) const;
        std::complex<double> kValue(const Position<double>& k) const;

        bool isAxisymmetric() const { return true; } 
        bool hasHardEdges() const { return false; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        double maxK() const;
        double stepK() const;

        Position<double> centroid() const 
        { return Position<double>(0., 0.); }

        double getFlux() const { return _flux; }

        /**
         * @brief Shoot photons through this SBGaussian.
         *
         * SBGaussian shoots photons by analytic transformation of the unit disk.  Slightly more
         * than 2 uniform deviates are drawn per photon, with some analytic function calls (sqrt,
         * etc.)
         *
         * @param[in] N Total number of photons to produce.
         * @param[in] ud UniformDeviate that will be used to draw photons from distribution.
         * @returns PhotonArray containing all the photons' info.
         */
        boost::shared_ptr<PhotonArray> shoot(int N, UniformDeviate ud) const;

        double getSigma() const { return _sigma; }

        // Overrides for better efficiency
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, int ix_zero,
                        double y0, double dy, int iy_zero) const;
        void fillXValue(tmv::MatrixView<double> val,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double x0, double dx, int ix_zero,
                        double y0, double dy, int iy_zero) const;
        void fillKValue(tmv::MatrixView<std::complex<double> > val,
                        double x0, double dx, double dxy,
                        double y0, double dy, double dyx) const;

    private:
        double _flux; ///< Flux of the Surface Brightness Profile.

        /// Characteristic size, surface brightness scales as `exp[-r^2 / (2. * sigma^2)]`.
        double _sigma;
        double _sigma_sq;
        double _inv_sigma;
        double _inv_sigma_sq;
        double _ksq_min; ///< If ksq < _kq_min, then use faster taylor approximation for kvalue
        double _ksq_max; ///< If ksq > _kq_max, then use kvalue = 0
        double _norm; ///< flux / sigma^2 / 2pi

        // Copy constructor and op= are undefined.
        SBGaussianImpl(const SBGaussianImpl& rhs);
        void operator=(const SBGaussianImpl& rhs);
    };
}

#endif // SBGAUSSIAN_IMPL_H

