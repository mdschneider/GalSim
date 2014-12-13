import numpy as np
import scipy.special
from scipy.spatial.distance import squareform, pdist


k_cm_to_m = 0.01

class PhaseScreen(object):
	"""
	Base class for generating atmospheric phase screens

	Attributes
	----------
	npix : int
		Number of pixels per dimension
	outer_scale : float
		Turbulence outer scale in meters
	r0 : float
		Fried parameter in centimeters at 0.5 micrometers
	inner_scale : float
		Turbulence inner scale in meters
	pixel_scale : float
		Physical size of phase screen pixels in centimeters
	"""
	def __init__(self, npix, outer_scale, r0, inner_scale=1.e-5,
				 pixel_scale=1):
		self.npix = npix
		self.outer_scale = outer_scale  # outer scale in von-Karman model
		self.r0 = r0  # Fried parameter in cm
		self.inner_scale = inner_scale  # inner scale in von-Karman model
		self.pixel_scale = pixel_scale

		# Precompute constants for von-Karman covariance
		cov_vk_term1 = (self.outer_scale / self.r0) ** (5. / 3.)
		cov_vk_term2 = (scipy.special.gamma(11. / 6.) / 
				 			 (2. ** (5. / 6.) * np.pi ** (8. / 3.)))
		cov_vk_term3 = ((24. / 5.) *
							 scipy.special.gamma(6. / 5.)) ** (5. / 6.)
		self.cov_vk_coef = cov_vk_term1 * cov_vk_term2 * cov_vk_term3
		self.cov_vk_2pi_L = 2. * np.pi / self.outer_scale

	def fried_param(self, wavelength=0.5):
		"""
		Wavelength dependent Fried parameter.

		Parameters
		----------
		wavelength: float
			Wavelength in micrometers
		"""
		return self.r0 * (wavelength / 0.5) ** (6. / 5.)

	def structure_param(self, wavelength=0.5, zenith_angle=0.0,
						delta_height=1.0):
		"""
		Index of refraction structure parameter for given Fried parameter.

		Parameters
		----------
		wavelength : float
			Wavelength in micrometers
		zenith_angle : float
			Zenith angle in radians
		delta_height : float
			Width of phase screen in meters

		Reference
		---------
			Eq. 3.51 of Hardy
		"""
		ksq = ((2. * np.pi) / wavelength) ** 2
		r0_microns = self.r0 * 1.e4
		dCNsq = r0_microns ** (-5. / 3.) / (0.423 * ksq / np.cos(zenith_angle))
		return dCNsq / (delta_height * 1e6)

	def phase_power_spectrum_kolmogorov(self, kappa):
		return kappa ** (-11. / 3.)

	def phase_power_spectrum_vonkarman(self, kappa):
		"""
		Phase power spectral density of the von-Karman model.

		Reference
		---------
			Hardy eqs. 3.14 and 3.96
		"""
		k0 = 2. * np.pi / self.outer_scale
		ki = 5.92 / self.inner_scale
		damping_term = np.exp(-(kappa / ki) ** 2)
		return (kappa ** 2 + k0 ** 2) ** (-11. / 6.) * damping_term

	def phase_covariance_vonkarman(self, r):
		"""
		Phase covariance function of the von-Karman model.

		Reference:
			eq. (5) of [1]
		"""
		# twopir_outer_scale = 2. * np.pi * r / self.outer_scale
		twopir_outer_scale = self.cov_vk_2pi_L * r
		# term1 = (self.outer_scale / self.r0) ** (5. / 3.)
		# term2 = (scipy.special.gamma(11. / 6.) / 
		# 		 (2. ** (5. / 6.) * np.pi ** (8. / 3.)))
		# term3 = ((24. / 5.) * scipy.special.gamma(6. / 5.)) ** (5. / 6.)
		term4 = twopir_outer_scale ** (5. / 6.)
		term5 = scipy.special.kv(5. / 6., twopir_outer_scale)
		return self.cov_vk_coef * term4 * term5

	def plot_covariance(self):
		import pylab as pl
		r = np.logspace(np.log10(self.pixel_scale),
						   np.log10(self.outer_scale) + 0.2)
		pl.loglog(r, self.phase_covariance_vonkarman(r), 'k-',
				 linewidth=3)
		pl.axvline(x=self.outer_scale, linestyle='dashed', color='black',
				   linewidth=2)
		pl.xlabel("$r$ (meters)", fontsize=18)
		pl.ylabel("$C_{\\phi}(r)$", fontsize=18)
		# pl.savefig("phase_screen_covariance_function.pdf")
		pl.show()

	def strucfcn(self, s, verbose=True, npad=2):
	    if verbose:
	        print '<strucfcn>'
	    n = s.shape[0]
	    m = s.shape[1]

	    sp = np.zeros((npad*n, npad*m), dtype=float)
	    sp[0:n, 0:m] = s
	    fsp = np.fft.fft2(sp)
	    nn2 = (float(n) * float(n)) ** 2

	    if verbose:
	        print 'calculating auto-correlation of data'
	    ss = nn2 * np.fft.ifft2(fsp * np.conj(fsp))

	    win = np.zeros((npad*n, npad*m), dtype=float)
	    win[0:n, 0:m] = 1.0
	    fwin = np.fft.fft2(win)

	    if verbose:
	        print 'calculating auto-correlation of window'
	    ww = np.real(nn2 * np.fft.ifft2(fwin * np.conj(fwin)))

	    if verbose:
	        print 'calculating cross-correlation window and screen^2'
	    cc = nn2 * np.fft.ifft2(fwin * np.conj(np.fft.fft2(sp ** 2)))
	    ccr = cc[:, ::-1][::-1, :]

	    ndx = np.where(np.real(ww) <= 1.e-8)
	    wwp = np.real(ww)
	    wwp[ndx] = 1.0
	    sf = np.real((cc + ccr - 2 * ss) / wwp)
	    # sf[n-1, m-1] = 0.
	    # sf[0, :] = 0.
	    # sf[:, 0] = 0.
	    if verbose:
	        print '<strucfcn> done'
	    return sf, cc, ccr, ss, ww


class PhaseScreenFFT(PhaseScreen):
	"""Generate a phase screen via FFT"""
	def __init__(self, npix, outer_scale, r0=0.1, inner_scale=1.e-5,
				 pixel_scale=0.01):
		super(PhaseScreenFFT, self).__init__(npix, outer_scale,
											 r0, inner_scale, pixel_scale)
		self.screensize_meters = self.npix * (self.pixel_scale * k_cm_to_m)
		self.fundamental_freq = 2. * np.pi / self.screensize_meters  ### m^{-1}

	def _fft_grid_indices(self):
		kndx = np.mgrid[0:self.npix, 0:self.npix]
		kfreq = np.fft.fftfreq(self.npix) * self.npix
		return kfreq[kndx[0]], kfreq[kndx[1]]

	def _power_amplitude(self):
		# return 0.023 * (2. * self.npix *
		# 	self.pixel_scale / self.r0) ** (5. / 3.)
		return 0.023 * self.r0 ** (-5. / 3.)

	def create_screen(self, iseed=None):
		""" 
		Generate a screen realization via FFT of Gaussian random field.

		Parameters
		----------
		iseed : float
			Seed for random number generator.

		Returns
		-------
	    delta1 : 2D grid of Gaussian turbulence.
	    delta2 : 2D grid of Gaussian turbulence.
		"""
		if iseed is not None:
		    np.random.seed(seed=iseed)
		### Generate random phases at each grid point
		phase = (2.0 * np.pi *
		         np.random.uniform(size=(self.npix, self.npix)))
		phiFT = np.cos(phase) + 1j * np.sin(phase)
		### Mode amplitudes drawn from Rayleigh distribution
		ikx, iky = self._fft_grid_indices()
		k = np.sqrt(ikx ** 2 + iky ** 2) * self.fundamental_freq
		r1 = np.random.uniform(size=(self.npix, self.npix))
		###
		# pkfun = self.phase_power_spectrum_vonkarman
		pkfun = self.phase_power_spectrum_kolmogorov
		###
		r = np.sqrt(pkfun(k) * self._power_amplitude() * (-2.0 * np.log(r1)))
		r[0, 0] = 0.0
		### Factor of 245 needed to match amplitude of structure function
		r = r * (2. * np.pi / self.screensize_meters) * 250. #245.
		# Combine amplitudes and phases and inverse FFT to get the density
		phiFT = r * phiFT
		phi = np.fft.ifft2(phiFT) * (float(self.npix) * float(self.npix))
		return np.real(phi), np.imag(phi)


class PhaseScreenMatMul(PhaseScreen):
	"""Compute a phase screen by matrix multiplication."""
	def __init__(self, npix, outer_scale, r0=0.1, inner_scale=1.e-5,
				 pixel_scale=0.01):
		super(PhaseScreenMatMul, self).__init__(npix, outer_scale,
												r0, inner_scale, pixel_scale)

	def _fill_corr_matrix(self, nx, ny):
		""" 
		Fill a covariance matrix of the phase screen pixels.

		Parameters
		----------
		nx : int
			Number of phase screen pixels in the x direction.
		ny : int
			Number of phase screen pixels in the y direction.

		Returns
		-------
		y : Description
		"""
		xndx = np.mgrid[0:nx, 0:ny] * self.pixel_scale / 100.
		r = squareform(pdist(np.array([xndx[0].flatten(), 
										  xndx[1].flatten()]).T))
		r[np.where(r == 0)] = 1.e-10
		return r, self.phase_covariance_vonkarman(r)

	def create_screen(self, iseed=None):
		""" 
		Generate a screen realization via matrix multiplication.

		Parameters
		----------
		iseed : float
			Seed for random number generator.

		Returns
		-------
	    delta : 2D grid of Gaussian turbulence.
		"""
		r, Sigma = self._fill_corr_matrix(self.npix, self.npix)
		mu = np.zeros(self.npix ** 2)
		x = np.random.multivariate_normal(mean=mu, cov=Sigma)
		return x.reshape((self.npix, self.npix))

	def create_conditional_screen(self, iseed=None):
		pass