/// \file SBProfile.h Contains a class definition for two-dimensional Surface Brightness Profiles.
//
/// The SBProfiles include common star, galaxy, and PSF shapes.
/// If you have not defined USE_LAGUERRE, the SBLaguerre class will be skipped.
/// If you have not defined USE_IMAGES, all of the drawing routines are disabled but you will no 
/// longer be dependent on the Image and FITS classes.



#ifndef SBPROFILE_H
#define SBPROFILE_H

#include <cmath>
#include <list>
#include <map>
#include <vector>
#include <algorithm>

#define USE_IMAGES ///< Remove this to disable the drawing routines. Also removes dependencies on Image and FITS classes.

#define USE_LAGUERRE ///< Remove this to skip the SBLaguerre classes.

#include "Std.h"
#include "Shear.h"
#include "FFT.h"
#include "Table.h"

#ifdef USE_IMAGES
#include "Image.h"
#endif

#ifdef USE_LAGUERRE
#include "Laguerre.h"
#endif

// ??? could += for SBAdd, or *= to SBConvolve
// ??? Ask for super-Nyquist sampling factor in draw??
namespace galsim {

    class SBError : public std::runtime_error 
    {
    public:
        SBError(const std::string& m="") : std::runtime_error("SB Error: " + m) {}
    };

    /// SBProfile is an abstract base class representing all of the 2D surface brightness 
    /// profiles that we know how to draw.
    //
    /// Every SBProfile knows how to draw an Image<float> of itself in real and k space.  Each also
    /// knows what is needed to prevent aliasing or truncation of itself when drawn.
    /// **Note** that when you use the SBProfile::draw() routines you will get an image of 
    /// **surface brightness** values in each pixel, not the flux that fell into the pixel.  To get
    /// flux, you must multiply the image by (dx*dx).
    /// drawK() routines are normalized such that I(0,0) is the total flux.
    /// Currently we have the following possible implementations of SBProfile:
    /// Basic shapes: SBBox, SBGaussian, SBExponential, SBAiry, SBSersic
    /// SBLaguerre: Gauss-Laguerre expansion
    /// SBDistort: affine transformation of another SBProfile
    /// SBRotate: rotated version of another SBProfile
    /// SBAdd: sum of SBProfiles
    /// SBConvolve: convolution of other SBProfiles
    class SBProfile 
    {
    protected:
        static const int MINIMUM_FFT_SIZE; ///< Constant giving minimum FFT sizes we're willing to do
        static const int MAXIMUM_FFT_SIZE; ///< Constant giving maximum FFT size we're willing to do
        static const double ALIAS_THRESHOLD; ///< A rough indicator of how good the FFTs need to be for setting maxK and stepK values

    public:

        /// Constructor (automatically generated by compiler);
        /// Copy Constructor (automatically generated by compiler);
        /// Destructor (virtual)  
        virtual ~SBProfile() {}                        

        //
        // implementation-dependent methods
        //

        /// Return a copy of self.
        virtual SBProfile* duplicate() const =0;

        /// Return value of SBProfile at a chosen 2D position in real space.
        //
        /// Assume all are real-valued.
        /// xValue() may not be implemented for derived classes (SBConvolve) that
        /// require an FFT to determine real-space values.
        /// \param _p Input: 2D position in real space.
        virtual double xValue(Position<double> _p) const =0; ///< Return value of SBProfile at a chosen 2D position in real space.

        /// Return value of SBProfile at a chosen 2D position in k space.
        //
        /// \param _p Input: 2D position in k space.
        virtual std::complex<double> kValue(Position<double> _p) const =0; ///< Return value of SBProfile at a chosen 2D position in k space.

        virtual double maxK() const =0; ///< Value of k beyond which aliasing can be neglected:
        virtual double nyquistDx() const { return M_PI / maxK(); } ///< Image pixel spacing that does not alias maxK.
        virtual double stepK() const =0; ///< Sampling in k space necessary to avoid folding of image in x space.

        virtual bool isAxisymmetric() const =0; ///< Characteristic that can affect efficiency of evaluation
        virtual bool isAnalyticX() const =0; ///< Characteristic that can affect efficiency of evaluation.  SBProfile is "analytic" in the real domain if values can be determined immediately at any position through formula or a stored table (no DFT).
        virtual bool isAnalyticK() const =0; ///< Characteristic that can affect efficiency of evaluation.  SBProfile is "analytic" in the k domain if values can be determined immediately at any position through formula or a stored table (no DFT).

        virtual double centroidX() const =0; ///<Centroid of SBProfile in X.
        virtual double centroidY() const =0; ///<Centroid of SBProfile in Y.

        /// Returns (X, Y) centroid of SBProfile.
        Position<double> centroid() const 
        { Position<double> p(centroidX(),centroidY());  return p; }


        virtual double getFlux() const =0; ///< Get flux scaling of SBProfile.

        /// Set flux scaling of SBProfile
        /// \param flux_ Input: flux
        virtual void setFlux(double flux_=1.) =0; ///< Set flux scaling of SBProfile.

        // ****Methods implemented in base class****

        // Transformations (all are special cases of affine transformations via SBDistort):

        /// Ellipse distortion transformation (affine without rotation).
        //
        /// \param e Input: Ellipse class distortion
        virtual SBProfile* distort(const Ellipse e) const; 

        /// Shear distortion transformation (affine without rotation or dilation).
        //
        /// \param e1 Input: first component of ellipticity
        /// \param e2 Input: second component of ellipticity
        virtual SBProfile* shear(double e1, double e2) const { return distort(Ellipse(e1,e2)); }
        
        /// Rotation distortion transformation
        //
        /// \param theta Input: rotation, in radians, anticlockwise
        virtual SBProfile* rotate(const double theta) const;

        /// Translation distortion transformation
        //
        /// \param dx Input: shift in X
        /// \param dy Input: shift in Y
        virtual SBProfile* shift(double dx, double dy) const;

#ifdef USE_IMAGES
        // **** Drawing routines ****


        /// Draw an image of the SBProfile in real space.
        //
        /// If input image `img` is not specified or has null dimension, a square image will be
        /// drawn which is big enough to avoid "folding."  If drawing is done using FFT,
        /// it will be scaled up to a power of 2, or 3x2^n, whicher fits.
        /// If input image has finite dimensions then these will be used, although in an FFT the 
        /// image 
        /// may be calculated internally on a larger grid to avoid folding.
        /// The default draw() routines decide internally whether image can be drawn directly
        /// in real space or needs to be done via FFT from k space.
        //
        /// \param dx Input: grid on which SBProfile is drawn has pitch dx; given dx=0. default, routine will choose dx to be at least fine enough for Nyquist sampling at maxK().  If you specify dx, image will be drawn with this dx and you will receive an image with the aliased frequencies included.
        /// \param wmult Input: specifying wmult>1 will draw an image that is wmult times larger than the default choice, i.e. it will have finer sampling in k space and have less folding.
        virtual Image<float> draw(double dx=0., int wmult=1) const;

        /// Draw the SBProfile in real space returning the summed flux in each pixel as an image.
        //
        /// If on input image `img` is not specified or has null dimension, a square image will be
        /// drawn which is big enough to avoid "folding."  If drawing is done using FFT,
        /// it will be scaled up to a power of 2, or 3x2^n, whicher fits.
        /// If input image has finite dimensions then these will be used, although in an FFT the 
        /// image 
        /// may be calculated internally on a larger grid to avoid folding.
        /// The default draw() routines decide internally whether image can be drawn directly
        /// in real space or needs to be done via FFT from k space.
        //
        /// \param img Input/Output: image
        /// \param dx Input: grid on which SBProfile is drawn has pitch dx; given dx=0. default, routine will choose dx to be at least fine enough for Nyquist sampling at maxK().  If you specify dx, image will be drawn with this dx and you will receive an image with the aliased frequencies included.
        /// \param wmult Input: specifying wmult>1 will draw an image that is wmult times larger than the default choice, i.e. it will have finer sampling in k space and have less folding.
        virtual double draw(Image<float> img, double dx=0., int wmult=1) const; 

        // Draw an image of the SBProfile in real space forcing the use of real methods where we have a formula for x values.
        //
        /// If on input image `img` is not specified or has null dimension, a square image will be
        /// drawn which is big enough to avoid "folding." 
        /// If input image has finite dimensions then these will be used, although in an FFT the 
        /// image 
        /// may be calculated internally on a larger grid to avoid folding.
        //
        /// \param img Input/Output: image
        /// \param dx Input: grid on which SBProfile is drawn has pitch dx; given dx=0. default, routine will choose dx to be at least fine enough for Nyquist sampling at maxK().  If you specify dx, image will be drawn with this dx and you will receive an image with the aliased frequencies included.
        /// \param wmult Input: specifying wmult>1 will draw an image that is wmult times larger than the default choice, i.e. it will have finer sampling in k space and have less folding.
        virtual double plainDraw(
            Image<float> img = Image<float>(), double dx=0., int wmult=1) const; 

        /// Draw an image of the SBProfile in real space forcing the use of Fourier transform from k space.
        //
        /// If on input image `img` is not specified or has null dimension, a square image will be
        /// drawn which is big enough to avoid "folding."  Drawing is done using FFT,
        /// and the image will be scaled up to a power of 2, or 3x2^n, whicher fits.
        /// If input image has finite dimensions then these will be used, although in an FFT the 
        /// image 
        /// may be calculated internally on a larger grid to avoid folding.
        //
        /// \param img Input/Output: image
        /// \param dx Input: grid on which SBProfile is drawn has pitch dx; given dx=0. default, routine will choose dx to be at least fine enough for Nyquist sampling at maxK().  If you specify dx, image will be drawn with this dx and you will receive an image with the aliased frequencies included.
        /// \param wmult Input: specifying wmult>1 will draw an image that is wmult times larger than the default choice, i.e. it will have finer sampling in k space and have less folding.
        virtual double fourierDraw(
            Image<float> img = Image<float>(), double dx=0., int wmult=1) const; 

        /// Draw an image of the SBProfile in k space.
        //
        /// For drawing in k space: routines are analagous to real space, except 2 images are needed since the SBProfile is complex.
        /// If on input either image `Re` or `Im` is not specified or has null dimension, square images will be
        /// drawn which are big enough to avoid "folding."  If drawing is done using FFT,
        /// they will be scaled up to a power of 2, or 3x2^n, whicher fits.
        /// If input image has finite dimensions then these will be used, although in an FFT the 
        /// image 
        /// may be calculated internally on a larger grid to avoid folding in real space.
        /// The default draw() routines decide internally whether image can be drawn directly
        /// in real space or needs to be done via FFT from k space.
        //
        /// \param Re Input/Output: image of real argument of SBProfile in k space
        /// \param Im Input/Output: image of imaginary argument of SBProfile in k space
        /// \param dk Input: grid on which SBProfile is drawn has pitch dk; given dk=0. default, routine will choose dk necessary to avoid folding of image in x space.  If you specify dk, image will be drawn with this dk and you will receive an image with folding artifacts included.
        /// \param wmult Input: specifying wmult>1 will expand the size drawn in k space.
        virtual void drawK(
            Image<float> Re= Image<float>(), Image<float> Im= Image<float>(),
            double dk=0., int wmult=1) const; 

        // evaluate in k space:
        virtual void plainDrawK(
            Image<float> Re= Image<float>(), Image<float> Im= Image<float>(), 
            double dk=0., int wmult=1) const; 

        // FT from x space
        virtual void fourierDrawK(
            Image<float> Re= Image<float>(), Image<float> Im= Image<float>(),
            double dk=0., int wmult=1) const; 

        // Utilities for drawing into Img data structures:
        virtual double fillXImage(Image<float> I, double dx) const;  // return flux integral

#endif

        // Utilities for drawing into FFT data structures - not intended for public
        // use, but need to be public so that derived classes can call them:
        virtual void fillKGrid(KTable& kt) const;
        virtual void fillXGrid(XTable& xt) const;

    };

    // Sum of SBProfile.  Note that this class stores duplicates of its summands,
    // so they cannot be changed after adding them.

    class SBAdd : public SBProfile 
    {
    protected:
        std::list<SBProfile*> plist;  // the plist content is a pointer to a fresh copy
        // Keep track of the cumulated properties of all summands:
    private:
        double sumflux;
        double sumfx;
        double sumfy;
        double maxMaxK;
        double minStepK;
        bool allAxisymmetric;
        bool allAnalyticX;
        bool allAnalyticK;
        void initialize();  // set all above variables to starting state
    public:
        // constructor, empty
        SBAdd() : plist() { initialize(); }

        // constructor, 1 input
        SBAdd(const SBProfile& s1) : plist() { initialize(); add(s1); }

        // constructor, 2 inputs
        SBAdd(const SBProfile& s1, const SBProfile& s2) : plist() 
        { initialize(); add(s1);  add(s2); }

        // constructor, list of inputs
        SBAdd(const std::list<SBProfile*> slist) : plist() 
        {
            std::list<SBProfile*>::const_iterator sptr;
            for (sptr = slist.begin(); sptr!=slist.end(); ++sptr)
                add(*(*sptr)->duplicate()); 
        }

        // copy constructor
        SBAdd(const SBAdd& rhs) : 
            plist(), sumflux(rhs.sumflux), sumfx(rhs.sumfx),
            sumfy(rhs.sumfy), maxMaxK(rhs.maxMaxK), minStepK(rhs.minStepK), 
            allAxisymmetric(rhs.allAxisymmetric),
            allAnalyticX(rhs.allAnalyticX), allAnalyticK(rhs.allAnalyticK)  
        {
            std::list<SBProfile*>::const_iterator sbptr;
            for (sbptr = rhs.plist.begin(); sbptr!=rhs.plist.end(); ++sbptr)
                plist.push_back((*sbptr)->duplicate());
        }

        // destructor
        ~SBAdd() 
        { 
            std::list<SBProfile*>::iterator pptr;
            for (pptr = plist.begin(); pptr!=plist.end(); ++pptr)  delete *pptr; 
        }

        // SBAdd specific method
        void add(const SBProfile& rhs, double scale=1.);

        // implementation dependent methods
        SBProfile* duplicate() const { return new SBAdd(*this); } 

        double xValue(Position<double> _p) const;
        std::complex<double> kValue(Position<double> _p) const;

        double maxK() const { return maxMaxK; }
        double stepK() const { return minStepK; }

        bool isAxisymmetric() const { return allAxisymmetric; }
        bool isAnalyticX() const { return allAnalyticX; }
        bool isAnalyticK() const { return allAnalyticK; }

        virtual double centroidX() const { return sumfx / sumflux; }
        virtual double centroidY() const { return sumfy / sumflux; }

        virtual double getFlux() const { return sumflux; }
        virtual void setFlux(double flux_=1.);

        // Overrides for better efficiency:
        virtual void fillKGrid(KTable& kt) const;
        virtual void fillXGrid(XTable& xt) const;
    };

    // SBDistort is an affine transformation of another SBProfile.
    // Stores a duplicate of its target.
    // Origin of original shape will now appear at x0.
    // Flux is NOT conserved in transformation - SB is preserved.
    class SBDistort : public SBProfile 
    {
        // keep track of all distortions in a 2x2 matrix M = (A B) (detM=1)
        // x0 is shift                                       (C D)
        // 
    private:
        SBProfile* adaptee;
        double matrixA, matrixB, matrixC, matrixD;
        // Calculate and save these:
        Position<double> x0;  // centroid position
        double absdet;            // determinant (flux magnification) of matrix
        double invdet;
        double major, minor; // major/minor axes of ellipse produced from unit circle
        bool stillIsAxisymmetric; // is output shape still circular?

    private:  
        void initialize();
        Position<double> fwd(Position<double> p) const 
        {
            Position<double> out(matrixA*p.x+matrixB*p.y,matrixC*p.x+matrixD*p.y);
            return out; 
        }

        Position<double> fwdT(Position<double> p) const 
        {
            Position<double> out(matrixA*p.x+matrixC*p.y,matrixB*p.x+matrixD*p.y);
            return out; 
        }

        Position<double> inv(Position<double> p) const 
        {
            Position<double> out(invdet*(matrixD*p.x-matrixB*p.y),
                                 invdet*(-matrixC*p.x+matrixA*p.y));
            return out; 
        }

        std::complex<double> kValNoPhase(Position<double> k) const 
        { return absdet*adaptee->kValue(fwdT(k)); }


    public:
        // general constructor:
        SBDistort(
            const SBProfile& sbin, double mA, double mB, double mC, double mD,
            Position<double> x0_=Position<double>(0.,0.));

        // Construct from Ellipse class:
        SBDistort(const SBProfile& sbin, const Ellipse e_=Ellipse());

        // copy constructor
        SBDistort(const SBDistort& rhs) 
        {
            adaptee = rhs.adaptee->duplicate();
            matrixA = (rhs.matrixA); 
            matrixB = (rhs.matrixB); 
            matrixC = (rhs.matrixC);
            matrixD = (rhs.matrixD); 
            x0 = (rhs.x0);
            initialize();
        }

        // destructor
        ~SBDistort() { delete adaptee; adaptee=0; }

        // methods
        SBProfile* duplicate() const 
        { return new SBDistort(*this); } 

        double xValue(Position<double> p) const 
        { return adaptee->xValue(inv(p-x0)); }

        std::complex<double> kValue(Position<double> k) const 
        {
            std::complex<double> phaseexp(0,-k.x*x0.x-k.y*x0.y); // phase exponent
            std::complex<double> kv(absdet*adaptee->kValue(fwdT(k))*exp(phaseexp));
            return kv; 
        }

        bool isAxisymmetric() const { return stillIsAxisymmetric; }
        bool isAnalyticX() const { return adaptee->isAnalyticX(); }
        bool isAnalyticK() const { return adaptee->isAnalyticK(); }

        double maxK() const { return adaptee->maxK() / minor; }
        double stepK() const { return adaptee->stepK() / major; }

        double centroidX() const { return (x0+fwd(adaptee->centroid())).x; }
        double centroidY() const { return (x0+fwd(adaptee->centroid())).y; }

        double getFlux() const { return adaptee->getFlux()*absdet; }
        void setFlux(double flux_=1.) { adaptee->setFlux(flux_/absdet); }

        void fillKGrid(KTable& kt) const; // optimized phase calculation
    };

    class SBConvolve : public SBProfile 
    {
    private:
        std::list<SBProfile*> plist;  // the plist content is a copy_ptr (cf. smart ptrs)
        double fluxScale;
        double x0, y0;
        bool isStillAxisymmetric;
        double minMaxK;
        double minStepK;
        double fluxProduct;

    public:
        // constructor, empty
        SBConvolve() : plist(), fluxScale(1.) {} 

        // constructor, 1 input; f is optional scaling factor for final flux
        SBConvolve(const SBProfile& s1, double f=1.) : plist(), fluxScale(f) 
        { add(s1); }

        // constructor, 2 inputs
        SBConvolve(const SBProfile& s1, const SBProfile& s2, double f=1.) : 
            plist(), fluxScale(f) 
        { add(s1);  add(s2); }

        // constructor, 3 inputs
        SBConvolve(
            const SBProfile& s1, const SBProfile& s2, const SBProfile& s3, double f=1.) :
            plist(), fluxScale(f) 
        { add(s1);  add(s2);  add(s3); }

        // constructor, list of inputs
        SBConvolve(const std::list<SBProfile*> slist, double f=1.) :
            plist(), fluxScale(f) 
        { 
            std::list<SBProfile*>::const_iterator sptr;
            for (sptr = slist.begin(); sptr!=slist.end(); ++sptr) add(**sptr); 
        }

        // copy constructor
        SBConvolve(const SBConvolve& rhs) : 
            plist(), fluxScale(rhs.fluxScale),
            x0(rhs.x0), y0(rhs.y0),
            isStillAxisymmetric(rhs.isStillAxisymmetric),
            minMaxK(rhs.minMaxK), minStepK(rhs.minStepK),
            fluxProduct(rhs.fluxProduct) 
        {
            std::list<SBProfile*>::const_iterator rhsptr;
            for (rhsptr = rhs.plist.begin(); rhsptr!=rhs.plist.end(); ++rhsptr)
                plist.push_back((*rhsptr)->duplicate()); 
        }

        // destructor
        ~SBConvolve() 
        { 
            std::list<SBProfile*>::iterator pptr;
            for (pptr = plist.begin(); pptr!=plist.end(); ++pptr)  delete *pptr; 
        }

        // SBConvolve specific method
        void add(const SBProfile& rhs); 

        // implementation dependent methods:
        SBProfile* duplicate() const { return new SBConvolve(*this); } 

        double xValue(Position<double> _p) const 
        { throw SBError("SBConvolve::xValue() not allowed"); } 

        std::complex<double> kValue(Position<double> k) const 
        {
            std::list<SBProfile*>::const_iterator pptr;
            std::complex<double> product(fluxScale,0);
            for (pptr=plist.begin(); pptr!=plist.end(); ++pptr)
                product *= (*pptr)->kValue(k);
            return product; 
        }

        bool isAxisymmetric() const { return isStillAxisymmetric; }
        bool isAnalyticX() const { return false; }
        bool isAnalyticK() const { return true; }    // convolvees must all meet this
        double maxK() const { return minMaxK; }
        double stepK() const { return minStepK; }
        double centroidX() const { return x0; }
        double centroidY() const { return y0; }
        double getFlux() const { return fluxScale * fluxProduct; }
        void setFlux(double flux_=1.) { fluxScale = flux_/fluxProduct; }

        // Overrides for better efficiency:
        virtual void fillKGrid(KTable& kt) const;
    };

    class SBGaussian : public SBProfile 
    {
    private:
        double flux;
        double sigma; // characteristic size:  exp[-r^2/(2.*sigma^2)]

    public:
        // Constructor
        SBGaussian(double flux_=1., double sigma_=1.) : flux(flux_), sigma(sigma_) {}

        // Destructor
        ~SBGaussian() {}                        

        double xValue(Position<double> _p) const;
        std::complex<double> kValue(Position<double> _p) const;

        bool isAxisymmetric() const { return true; } 
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        // Extend to 4 sigma in both domains, or more if needed to reach EE spec
        double maxK() const { return std::max(4., sqrt(-2.*log(ALIAS_THRESHOLD))) / sigma; }
        double stepK() const { return M_PI/std::max(4., sqrt(-2.*log(ALIAS_THRESHOLD))) / sigma; }
        double centroidX() const { return 0.; } 
        double centroidY() const { return 0.; } 
        double getFlux() const { return flux; }
        void setFlux(double flux_=1.) { flux=flux_; }
        SBProfile* duplicate() const { return new SBGaussian(*this); }
    };

    class SBSersic : public SBProfile 
    {
    private:
        // First define some private classes that will cache the
        // needed parameters for each Sersic index n.
        class SersicInfo 
        {
            // This private class contains all the info needed 
            // to calculate values for a given n.  
        public:
            SersicInfo(double n);
            double inv2n;   // 1/2n
            double maxK;
            double stepK;
            double xValue(double xsq) const { return norm*exp(-b*pow(xsq,inv2n)); }
            double kValue(double ksq) const; // Look up k value in table

        private:
            SersicInfo(const SersicInfo& rhs) {} // Hide the copy constructor.
            double b;
            double norm;
            double kderiv2; // quadratic dependence near k=0;
            double kderiv4; // quartic dependence near k=0;
            double logkMin;
            double logkMax;
            double logkStep;
            std::vector<double> lookup;
        };

        // A map to hold one copy of the SersicInfo for each n ever used
        // during the program run.  Make one static copy of this map.
        class InfoBarn : public std::map<double, const SersicInfo*> 
        {
        public:
            const SersicInfo* get(double n) 
            {
                const int MAX_SERSIC_TABLES = 100; // How many n's can be stored?
                const SersicInfo* info = (*this)[n];
                if (info==0) {
                    info = new SersicInfo(n);
                    (*this)[n] = info;
                    if (int(size()) > MAX_SERSIC_TABLES)
                        throw SBError("Storing Sersic info for too many n values");
                }
                return info;
            }

            ~InfoBarn() 
            {
                typedef std::map<double,const SersicInfo*>::iterator mapit;
                for (mapit pos = begin(); pos != end(); ++pos) {
                    delete pos->second;
                    pos->second = 0;
                }
            }
        };
        static InfoBarn nmap;

        // Now the parameters of this instance of SBSersic:
        double n;
        double flux;
        double re;   // half-light radius
        const SersicInfo* info; // point to info structure for this n

    public:
        // Constructor
        SBSersic(double n_, double flux_=1., double re_=1.) :
            n(n_), flux(flux_), re(re_), info(nmap.get(n)) {}

        // Default copy constructor should be fine.
        // Destructor
        ~SBSersic() {}

        double xValue(Position<double> p) const 
        {
            p /= re;
            return flux*info->xValue(p.x*p.x+p.y*p.y) / (re*re);
        }

        std::complex<double> kValue(Position<double> k) const 
        {
            k *= re;
            return std::complex<double>( flux*info->kValue(k.x*k.x+k.y*k.y), 0.);
        }

        bool isAxisymmetric() const { return true; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }  // 1d lookup table

        double maxK() const { return info->maxK / re; }
        double stepK() const { return info->stepK / re; }

        double centroidX() const { return 0.; } 
        double centroidY() const { return 0.; } 

        double getFlux() const { return flux; }
        void setFlux(double flux_=1.) { flux=flux_; }

        SBProfile* duplicate() const { return new SBSersic(*this); }

        // A method that only works for Sersic:
        double getN() const { return n; }
    };

    // Keep distinct SBExponential since FT has closed form for this:
    class SBExponential : public SBProfile 
    {
    private:
        double r0;   // characteristic size:  exp[-(r/r0)]
        double flux;
    public:
        // Constructor - note that r0 is scale length, not half-light
        SBExponential(double flux_=1., double r0_=1.) : r0(r0_), flux(flux_) {}

        // Destructor
        ~SBExponential() {}

        // Methods
        double xValue(Position<double> _p) const;
        std::complex<double> kValue(Position<double> _p) const;

        bool isAxisymmetric() const { return true; } 
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        // Set maxK where the FT is down to 0.001 or threshold, whichever is harder.
        double maxK() const { return std::max(10., pow(ALIAS_THRESHOLD, -1./3.))/r0; }
        double stepK() const;

        double centroidX() const { return 0.; } 
        double centroidY() const { return 0.; } 

        double getFlux() const { return flux; }
        void setFlux(double flux_=1.) { flux=flux_; }

        SBProfile* duplicate() const { return new SBExponential(*this); }
    };

    class SBAiry : public SBProfile 
    {
    private:
        double D;    // D = (telescope diam) / (lambda * focal length) if arg is focal plane
        // position, else D = diam / lambda if arg is in radians of field angle.
        double obscuration; // radius ratio of central obscuration
        double flux;

    public:
        // Constructor
        SBAiry(double D_=1., double obs_=0., double flux_=1.) :
            D(D_), obscuration(obs_), flux(flux_) {}

        // Destructor
        ~SBAiry() {}

        // Methods
        double xValue(Position<double> _p) const;
        std::complex<double> kValue(Position<double> _p) const;

        bool isAxisymmetric() const { return true; } 
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        double maxK() const { return 2*M_PI*D; } // hard limit for Airy

        // stepK makes transforms go to at least 5 lam/D or EE>(1-ALIAS_THRESHOLD).
        // Schroeder (10.1.18) gives limit of EE at large radius.
        // This stepK could probably be relaxed, it makes overly accurate FFTs.
        double stepK() const 
        { 
            return std::min( 
                ALIAS_THRESHOLD * 0.5 * D * pow(M_PI,3.) * (1-obscuration) ,
                M_PI * D / 5.);
        }
        double centroidX() const { return 0.; } 
        double centroidY() const { return 0.; } 

        double getFlux() const { return flux; }
        void setFlux(double flux_=1.) { flux=flux_; }

        SBProfile* duplicate() const { return new SBAiry(*this); }

    private: 
        double chord(const double r, const double h) const;
        double circle_intersection(
            double r, double s, double t) const;
        double annuli_intersect(
            double r1, double r2, double t) const;
        double annuli_autocorrelation(const double k) const;
    };

    class SBBox : public SBProfile 
    {
    private:
        double xw;  // box is xw x yw across.
        double yw;
        double flux;
        double sinc(const double u) const;
    public:
        // Constructor
        SBBox(double xw_=1., double yw_=0., double flux_=1.) :
            xw(xw_), yw(yw_), flux(flux_) 
        { if (yw==0.) yw=xw; }

        // Destructor
        ~SBBox() {}

        // Methods
        double xValue(Position<double> _p) const;
        std::complex<double> kValue(Position<double> _p) const;

        bool isAxisymmetric() const { return false; } 
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        double maxK() const { return 2. / ALIAS_THRESHOLD / std::max(xw,yw); }  
        double stepK() const { return M_PI/std::max(xw,yw)/2; } 

        double centroidX() const { return 0.; } 
        double centroidY() const { return 0.; } 

        double getFlux() const { return flux; }
        void setFlux(double flux_=1.) { flux=flux_; }

        SBProfile* duplicate() const { return new SBBox(*this); }

        // Override to put in fractional edge values:
        void fillXGrid(XTable& xt) const;

    protected:
#ifdef USE_IMAGES
        double fillXImage(Image<float> I, double dx) const;
#endif
    };

#ifdef USE_LAGUERRE
    class SBLaguerre : public SBProfile 
    {
    private:
        LVector bvec;  // bvec[n,n] contains flux information
        double sigma;
    public:
        // Constructor 
        SBLaguerre(LVector bvec_=LVector(), double sigma_=1.) : 
            bvec(bvec_.duplicate()), sigma(sigma_) {}

        // Copy Constructor 
        SBLaguerre(const SBLaguerre& rhs) :
            bvec(rhs.bvec.duplicate()), sigma(rhs.sigma) {}

        // Destructor 
        ~SBLaguerre() {}

        // implementation dependent methods
        SBProfile* duplicate() const { return new SBLaguerre(*this); }

        double xValue(Position<double> _p) const;
        std::complex<double> kValue(Position<double> _p) const;

        double maxK() const;
        double stepK() const;

        bool isAxisymmetric() const { return false; }
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }

        double centroidX() const 
        { throw SBError("SBLaguerre::centroid calculations not yet implemented"); }
        double centroidY() const 
        { throw SBError("SBLaguerre::centroid calculations not yet implemented"); }

        double getFlux() const;
        void setFlux(double flux_=1.);

        // void fillKGrid(KTable& kt) const;
        // void fillXGrid(XTable& xt) const;

    };
#endif

    class SBMoffat : public SBProfile 
    {
    private:
        double beta;
        double flux;
        double norm;
        double rD;
        // In units of rD:
        double maxRrD;
        double maxKrD;
        double stepKrD;
        double FWHMrD;
        double rerD;

        Table<double,double> ft;

    public:
        // Constructor
        SBMoffat(
            double beta_, double truncationFWHM=2., double flux_=1., double re=1.);

        // Default copy constructor should be fine.
        // Destructor
        ~SBMoffat() {}

        double xValue(Position<double> p) const 
        {
            p /= rD;
            double rsq = p.x*p.x+p.y*p.y;
            if (rsq >= maxRrD*maxRrD) return 0.;
            else return flux*norm*pow(1+rsq, -beta) / (rD*rD);
        }

        std::complex<double> kValue(Position<double> k) const; 

        bool isAxisymmetric() const { return true; } 
        bool isAnalyticX() const { return true; }
        bool isAnalyticK() const { return true; }  // 1d lookup table

        double maxK() const { return maxKrD / rD; }   
        double stepK() const { return stepKrD / rD; } 

        double centroidX() const { return 0.; } 
        double centroidY() const { return 0.; } 

        double getFlux() const { return flux; }
        void setFlux(double flux_=1.) { flux=flux_; }

        SBProfile* duplicate() const { return new SBMoffat(*this); }

        // Methods that only works for Moffat:
        double getBeta() const { return beta; }
        void setFWHM(double fwhm) { rD = fwhm / FWHMrD; }
        void setRd(double rD_) { rD = rD_; }
    };

    // This is for backwards compatibility; prefer rotate() method.
    class SBRotate : public SBDistort 
    {
    public:
        // constructor #1
        SBRotate(const SBProfile& s, const double theta) :
            SBDistort(s, cos(theta), -sin(theta), sin(theta), cos(theta)) {}
    };

    class SBDeVaucouleurs : public SBSersic 
    {
    public:
        // Constructor
        SBDeVaucouleurs(double flux_=1., double r0_=1.) : SBSersic(4., flux_, r0_) {}

        // Destructor
        ~SBDeVaucouleurs() {}

        SBProfile* duplicate() const { return new SBDeVaucouleurs(*this); }
    };


}

#endif // SBPROFILE_H

