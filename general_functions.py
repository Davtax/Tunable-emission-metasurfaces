import numpy as np
from scipy.integrate import romb, cumtrapz, simps
from scipy.special import erfi
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from tqdm.auto import tqdm

counter_verify = 0  # Counter for the verification in complete_dynamics


def g0_fun(k0, aho):
	"""
	Compute the renormalized fluctuation-averaged Green's tensor for r = 0. This corresponds to Eq. (23) of J. Perczel,
	et al., PRA 96, 063801 (2017)

	Parameters
	----------
	k0: float
		Wave number of the dipoles in the array.
	aho: float
		Distance associated with the cut-off in momentum.

	Returns
	-------
	g0: float
		Renormalized fluctuation averaged Green's tensor.
	"""
	g0 = k0 / (6 * np.pi) * (
			(erfi(k0 * aho / np.sqrt(2)) - 1j) / np.exp((k0 * aho) ** 2 / 2) - (-1 / 2 + (k0 * aho) ** 2) / (
			np.sqrt(np.pi / 2) * (k0 * aho) ** 3))
	return g0


def compute_energies(b1, b2, A, nmax, k0, aho, k, G0=None, polarizations=None):
	"""
	Compute the energies for all the introduced values of the momentum k.

	Parameters
	----------
	b1: array_like
		Cartesian coordinates (x and y) for the first lattice vector of the reciprocal space.
	b2: array_like
		Cartesian coordinates (x and y) for the second lattice vector of the reciprocal space.
	A: float
		Area of the unit cell.
	nmax: int
		 Number of elements in the summatory for each direction.
	k0: float
		Wave number of the dipoles in the array.
	aho: float
		Distance associated with the cut-off in momentum.
	k: array_like (n, 2)
		All the n momentum's of interest given by (k_x, k_y).
	G0: complex, optional, default: None
		Rescaled Green tensor at the origin. If None, it is calculated here.
	polarizations: List, optional, default: None
		 Polarization of lattice atoms, each element represents one of them. The polarization must be a unit vector in
		 cartersian coordinates (x, y and z), written as a (3, 1) ndarray.

	Returns
	-------
	J, Gamma: array_like (N, ...)
		Band structure and decay rates for the given momentum's. If polarization is	not provided, then the bands
		correspond to the eigenvalues of the Hamiltonian, otherwise there will be as bands as the number of
		polarizations given.
	"""
	# Compute G0 if it's not given
	if G0 is None:
		G0 = g0_fun(k0, aho)

	N_tot = len(k)  # Total number of momentum's

	hypermatrix = compute_hamiltonian(b1, b2, A, nmax, k0, aho, k, G0=G0)

	if polarizations is None:  # If not polarization is provided, compute the eigenvalues
		# Extract the elements of the hamiltonian
		a = hypermatrix[:, 0, 0]
		b = hypermatrix[:, 1, 1]
		c = hypermatrix[:, 2, 2]
		d = hypermatrix[:, 0, 1]
		e = hypermatrix[:, 1, 0]

		# Compute "by-hand" it's three eigenvalues
		energies = np.zeros((N_tot, 3), dtype=complex)
		energies[:, 0] = ((a + b) + np.sqrt((a + b) ** 2 - 4 * (a * b - e * d))) / 2
		energies[:, 1] = ((a + b) - np.sqrt((a + b) ** 2 - 4 * (a * b - e * d))) / 2
		energies[:, 2] = c

	else:
		N = len(polarizations)  # Number of polarizations
		energies = np.zeros((N_tot, N), dtype=complex)  # Initialize the bands

		# Iterate for all the polarizations
		for i in range(N):
			polarization = polarizations[i]  # Extract the polarization
			polarization = polarization.reshape((3, 1))  # Ensure that the polarization is a column vector
			temp = polarization.T.conj() @ hypermatrix @ polarization  # Compute the band
			energies[:, i] = temp[:, 0, 0]  # Save the band

	# Extract the band (real part) and the decay rates (imaginary part)
	J = np.real(energies)
	Gamma = - 2 * np.imag(energies)

	return J, Gamma


def compute_hamiltonian(b1, b2, A, nmax, k0, aho, k, G0=None):
	"""
	Compute the hamiltonian divided by Gamma_0 for a 2D infinite array of dipoles for a given values of the Bloch
	momentum. The hamiltonian is written in the cartesian basis (x, y, z). The origin of energies is set at
	omega_0/Gamma_0 by subtracting this quantity from the diagonal.

	Parameters
	----------
	b1: array_like
		Cartesian coordinates (x and y) for the first lattice vector of the reciprocal space.
	b2: array_like
		Cartesian coordinates (x and y) for the second lattice vector of the reciprocal space.
	A: float
		Area of the unit cell.
	nmax: int
		 Number of elements in the summatory for each direction.
	k0: float
		Wave number of the dipoles in the array.
	aho: float
		Distance associated with the cut-off in momentum.
	k: array_like (n, 2)
		All the n momentum's of interest given by (k_x, k_y).
	G0: complex, optional, default: None
		Rescaled Green tensor at the origin. If None, it is calculated here.

	Returns
	-------
	hamiltonian: array_like (n, 3, 3)
		hamiltonian for all the values of the momentum provided.
	"""

	n = len(k)  # Number of different momentum's provided

	# Compute G0 if it's not given
	if G0 is None:
		G0 = g0_fun(k0, aho)

	shift = -np.identity(3, dtype=complex) * 1j / 2  # Shift in the diagonal due to individual decays rates

	hamiltonian = np.zeros((n, 3, 3), dtype=complex)  # Initialize the hypermatrix for the hamiltonian of each momentum

	# Iterate all the momentum's
	for i in range(n):
		chi = green_dyadic(b1, b2, A, nmax, k[i], k0, aho, G0, matrix_return=True)  # Compute the Green dyadic
		hamiltonian[i, :, :] = shift + chi  # Sum all the contributions

	return hamiltonian


def green_dyadic(b1, b2, area, nmax, k, k0, aho, g0=None, matrix_return=False):
	"""
	Compute the Green dyadic divided by Gamma_0 for an infinite 2D array of equal dipoles with periodic boundary
	conditions. The sum is performed in the momentum space, since the convergence is faster. The array is set to have
	the same elements in both directions. The Green tensor is finally multiply by the factor (3 * pi) / k0. This
	function corresponds to Eq. (9) and Eq. (30) of J. Perczel, et al., PRA 96, 063801 (2017)

	Parameters
	----------
	b1: array_like
		Cartesian coordinates (x and y) for the first lattice vector of the reciprocal space.
	b2: array_like
		Cartesian coordinates (x and y) for the second lattice vector of the reciprocal space.
	area: float
		Area of the unit cell.
	nmax: int
		 Number of elements in the summatory for each direction.
	k0: float
		Wave number of the dipoles in the array.
	aho: float
		Distance associated with the cut-off in momentum.
	k: array_like (2)
		Momentum of interest given by (k_x, k_y).
	g0: complex, optional, default: None
		Rescaled Green tensor at the origin. If None, it is calculated here.
	matrix_return: bool, optional, default: False
		 Return the hole matrix representation. If not, just return the non-vanishing elements.

	Returns
	-------
	chi: array-like (3, 3)
		Green tensor in the basis (x, y and z). If matrix_return=False, chi is a list of four elements with g_xx, g_yy,
		 g_z and g_xy.
	"""

	#  Parameter to check if there is an overflow in the calculation due to large number of elements in the summation
	global error_overflow

	if g0 is None:
		g0 = g0_fun(k0, aho)  # Compute the rescaled Green tensor at the origin

	gx, gy = lattice_sites(b1, b2, nmax, nmax)  # Reciprocal-lattice vectors

	# Components of G_vec - k_vec
	px = gx - k[0]
	py = gy - k[1]

	# Compute the auxiliary parameters
	c = c_fun(px, py, k0, aho)
	lambda_value = lambda_fun(px, py, k0)

	# Ignore the possible overflow message due to the exponential of something close -> -inf just for the next step
	np.seterr(over='ignore')
	exponential = np.exp(-(aho * lambda_value) ** 2 / 2)
	np.seterr(over='raise')

	# If there is some overflow, and it is the first time it occurs
	if np.any(np.isinf(exponential)):
		# All the overflows are set equal to the maximum value which is not an infinite
		limit = exponential[~np.isinf(exponential)].max()
		exponential[np.isinf(exponential)] = limit

		if not error_overflow:
			print('Overflow in exponential, enough terms in the summation')
			error_overflow = True

	# Auxiliary parameters given by Eqs. (25 - 26) of Perczel17b
	i0 = c * np.pi * exponential / (lambda_value + 1e-16) * (-1j + erfi(aho * lambda_value / np.sqrt(2)))
	i2 = c * (-np.sqrt(2 * np.pi) / aho) + i0 * lambda_value ** 2

	# Non-vanishing elements for the Green dyadic given by Eq. (24) of J. Perczel, et al., PRA 96, 063801 (2017)
	gxx = (k0 ** 2 - px ** 2) * i0
	gyy = (k0 ** 2 - py ** 2) * i0
	gxy = -px * py * i0
	gzz = k0 ** 2 * i0 - i2

	# Realize the sum for all reciprocal-lattice vectors
	gxx = np.exp(k0 ** 2 * aho ** 2 / 2) / area * np.sum(gxx) - g0
	gyy = np.exp(k0 ** 2 * aho ** 2 / 2) / area * np.sum(gyy) - g0
	gzz = np.exp(k0 ** 2 * aho ** 2 / 2) / area * np.sum(gzz) - g0
	gxy = np.exp(k0 ** 2 * aho ** 2 / 2) / area * np.sum(gxy)

	if matrix_return:
		chi = np.zeros((3, 3), dtype=complex)

		# introduce the non-vanishing elements
		chi[0, 0] = gxx
		chi[1, 1] = gyy
		chi[2, 2] = gzz
		chi[1, 0] = gxy
		chi[0, 1] = gxy

		chi *= 3 * np.pi / k0

		return chi
	else:
		chi_elements = np.array([gxx, gyy, gzz, gxy]) * 3 * np.pi / k0
		return chi_elements


def lattice_sites(a1, a2, nmax, mmax):
	"""
	Generate the lattice sites with lattice vectors a1 and a2. The sites for each vector are
	(-nmax, -nmax +1, ..., 0, ..., nmax). The total number of elements is (2 * nmax + 1) * (2 * mmax + 1).

	Parameters
	----------
	a1: array_like
		Cartesian indices (x, y) for the first lattice vector.
	a2: array_like
		Cartesian indices (x, y) for the second lattice vector.
	nmax: int
		Number of sites in the direction of a1.
	mmax: int
		Number of sites in the direction of a1.

	Returns
	-------
	xij, yij: (array_like)
		Coordinates for each site in the lattice.

	"""
	nmat = np.arange(-nmax, nmax + 1)
	mmat = np.arange(-mmax, mmax + 1)
	nmat, mmat = np.meshgrid(nmat, mmat, indexing='ij')

	xij = nmat * a1[0] + mmat * a2[0]
	yij = nmat * a1[1] + mmat * a2[1]

	return xij, yij


def lambda_fun(px, py, k0):
	"""
	Generate the auxiliary parameter Lambda(px, py) needed in the lattice sum. This corresponds to Eq. (28)
	of J. Perczel, et al., PRA 96, 063801 (2017).

	Parameters
	----------
	px: array_like
		x-component of the momentum.
	py: array_like
		y-component of the momentum.
	k0: float
		Wave number of the dipoles in the array.

	Returns
	-------
	complex
		Lambda auxiliary parameter.
	"""
	# The positive value is automatically chosen by numpy. The 0j factor is needed to force a complex return
	return np.sqrt(k0 ** 2 - px ** 2 - py ** 2 + 0j)


def c_fun(px, py, k0, aho):
	"""
	Generate the auxiliary parameter C(px, py) needed in the lattice sum. This corresponds to Eq. (27) of J. Perczel,
	et al., PRA 96, 063801 (2017).

	Parameters
	----------
	px: array_like
		x-component of the momentum.
	py: array_like
		y-component of the momentum.
	k0: float
		Wave number of the dipoles in the array.
	aho: float
		Distance associated with the cut-off in momentum.

	Returns
	-------
	float
		C auxiliary parameter
	"""
	return 1 / (2 * np.pi * k0 ** 2) * np.exp(-aho ** 2 * (px ** 2 + py ** 2) / 2)


def dos_general(energy, weight_function=None, args=None, min_E=None, max_E=None, n=1000):
	"""
	Compute the density of stats (DoS) for a given energies previously computed by other functions. This method assumes
	an approximation for the Dirac delta. If non approximation is provided, then and a square Heaviside function is
	used to weight the energies.

	Parameters
	----------
	energy: array_like (n)
		Energies of the system.
	weight_function: function, optional, default=None
		User defined function to approximate a Dirac delta. If none, use a two-side Heaviside step function. This
		function needs the width of the step as a parameter.
	args: list, optional, default=None
		Arguments for weight_function. If None, and the weight_function is not specified, a default step width of 0.01
		is chosen.
	min_E: float, optional, default=min(energy)
		Minimal value for the energy at which compute the DoS.
	max_E: float, optional, default=max(energy)
		Maximum value for the energy at which compute the DoS.
	n: int, optional, default=1000
		Number of energies to compute the DoS.

	Returns
	-------
	energy_vector: array_like (n)
		Energies where the DoS is computed.
	DoS: array_like (n)
		Computed DoS.
	"""

	# If no function is provided
	if weight_function is None:
		if args is None:
			args = [0.01]

		# Define a two sides Heaviside step with total width of delta_x
		def weight_function(x_energy, delta_x):
			return np.sum((x_energy > - delta_x / 2) * (x_energy < delta_x / 2))

	# Ensure that args is a list, and not a single parameter
	if type(args) != list:
		args = [args]

	# If the extreme values for the energy are not given, it is set to the extreme values of the given energy multiplied
	# by certain factor to observe the decrease of the DoS
	range_E = np.max(energy) - np.min(energy)
	scale_factor = 10 / 100
	if min_E is None:
		min_E = np.min(energy) - range_E * scale_factor
	if max_E is None:
		max_E = np.max(energy) + range_E * scale_factor

	# Initialize the array to save the DoS and the energies at which evaluate it
	DoS = np.zeros(n)
	energy_vector = np.linspace(min_E, max_E, n, endpoint=True)

	# Iterate every energy
	for i in range(n):
		x = (energy_vector[i] - energy)
		# Count the energies weighted by a given approximated Dirac Delta
		DoS[i] = weight_function(x, *args)

	return energy_vector, DoS


def plot_band_structure(fig, ax, J, Gamma, symmetric_points, limit_light=None, limit_inf=None, limit_sup=None,
                        label_cbar=None, cbar_norm=25, cmap='jet', cbar=True, size=1, lines=False, dos=None,
                        energy_dos=None, ax_dos=None):
	"""
	Plot the band structure previously compute by other function. The decay rate is encoded in the colour of the bands.
	The bands are along the given directions in symmetric_points, with m / len(symmetric_points) points in each
	direction.
	If provided, the DoS for each band, and the total DoS, is also plotted.

	Parameters
	----------
	fig: matplotlib.fig
		Figure in which plot the band structure.
	ax: matplotlib.axes
		Axis in which plot the band structure.
	J: array_like (m, n)
		n bands denoting the energies.
	Gamma: array_like (m, n)
		n bands denoting the decay rates.
	symmetric_points: list
		Labels of the symmetric points.
	limit_light: list, optional, default=None
		Position with respect to x for the cone light. If None, no cone light is plotted.
	limit_inf: float, optional, default=None
		Minimum value for the y-axis.
	limit_sup: float, optional, default=None
		Maximum value for the y-axis.
	label_cbar: str, optional, default=None
		Label of the color bar.
	cbar_norm: float, optional, default=25
		Normalization of the decay rates.
	cmap: str, optional, default='jet'
		Colormap for the decay rate.
	cbar: bool, optional, default=True
		Draw the color bar.
	size: int, optional, default=1
		Size of the points.
	lines: bool, optional, default=True
		If true, plot lines joining the data points.
	dos: array_like (n, k)
		DoS to plot for each band.
	energy_dos: array_like (k), optional, default=None
		Energies at which the DoS are computed. If None, the DoS is not plotted.
	ax_dos: matplotlib.axes, optional, default=None
		Axis at which plot the DoS. If None, the DoS is not plotted.
	"""

	if label_cbar is None:
		label_cbar = r'$\gamma_\mathbf{k}/\Gamma_0$'

	m, n = np.shape(J)  # Total number of bands
	norm = plt.Normalize(0, cbar_norm)  # Normalize the decay rates

	x = np.linspace(-1, len(symmetric_points) - 2, m)

	sc = None  # Ensure that the variable exists
	for i in range(n):  # Iterate over all the bands
		if lines:
			ax.plot(x, J[:, i], 'k')
		else:
			sc = ax.scatter(x, J[:, i], c=Gamma[:, i], norm=norm, cmap=cmap,
			                s=size)  # Plot the bands with a scatter with de color encoding the decay rate

	if cbar:  # If the color-bar is plotted
		ax_divider = make_axes_locatable(ax)  # Locate the axis in which the band structure is plotted
		cax = ax_divider.append_axes("top", size="4%", pad="2%")  # The color-bar is located at the top of the axis
		fig.colorbar(sc, cax=cax, orientation="horizontal", extend='max', label=label_cbar)  # Plot the color-bar

		# Make the ticks and label of the color-bar at top
		cax.xaxis.set_ticks_position("top")
		cax.xaxis.set_label_position('top')

	# Compute the limits of the figure if these are not provided
	if limit_inf is None:
		limit_inf = np.min(J) * 1.1
	if limit_sup is None:
		limit_sup = np.max(J) * 1.1

	# Modify the limits of the figure
	ax.set_xlim(-1, len(symmetric_points) - 2)
	ax.set_ylim(limit_inf, limit_sup)

	pos = np.arange(-1, len(symmetric_points) - 1)  # Position of the symmetric points

	# Symmetric point names as x labels
	ax.set_xticks(pos)
	ax.set_xticklabels(symmetric_points)
	ax.xaxis.set_minor_locator(AutoMinorLocator(2))

	# Plot the light cone
	if limit_light is not None:
		ax.fill_between([limit_light[0], limit_light[1]], limit_inf - np.abs(limit_inf) * 10,
		                limit_sup + np.abs(limit_sup) * 10, color='grey', alpha=0.2)

	ax.set_ylabel(r'$(\omega_\mathbf{k}-\omega_0)/\Gamma_0$')
	ax.grid(zorder=-10)

	if dos is not None:
		for i in range(n):
			ax_dos.plot(dos[i], energy_dos, alpha=0.5)
		DoS_total = sum(dos)

		ax_dos.plot(DoS_total, energy_dos, 'k')
		ax_dos.set_xlabel('DoS')

		ax_dos.set_xlim((0, np.max(DoS_total)))
		ax_dos.set_ylim([limit_inf, limit_sup])

		ax_dos.grid(axis='y')

		ax2_divider = make_axes_locatable(ax_dos)
		ax2_2 = ax2_divider.append_axes("top", size="4%", pad="2%")
		ax2_2.axis('off')
		ax_dos.axes.yaxis.set_ticklabels([])
		ax_dos.axes.xaxis.set_ticklabels([])


def check_convergence(points_sites, b1, b2, A, nmax, k0, aho, G0=None, n=100, limit_error=1e-5, step=2,
                      limit_counter=50, polarizations=None, print_progress=False, bands=None):
	"""
	Check if the band structure converged for the given number of reciprocal-lattice vectors included in the summation.
	The convergence is up to a given limit_error, and with a maximum number of iterations.

	Parameters
	----------
	points_sites: list
		Coordinates at which compute the band structure.
	b1: array_like
		Cartesian coordinates (x and y) for the first lattice vector of the reciprocal space.
	b2: array_like
		Cartesian coordinates (x and y) for the second lattice vector of the reciprocal space.
	A: float
		Area of the unit cell.
	nmax: int
		 Number of elements in the summatory for each direction.
	k0: float
		Wave number of the dipoles in the array.
	aho: float
		Distance associated with the cut-off in momentum.
	G0: float, optional, default=None
		Rescaled Green tensor at the origin. If None, it is calculated here.
	n: int, optional, default=100
		Number of momentum's between adjacent sites.
	limit_error: float, optional, default=1e-5
		Minimum absolute error for the check.
	step: int, optional, default=2
		Incrementation on the number of the reciprocal-lattice vectors at each iteration.
	limit_counter: int, optional, default=50
		Maximum number of interactions, if the convergence has not been reached after that, the function return a
		warning.
	polarizations: list, optional, default=None
		Polarization of the dipoles.
	print_progress: bool, optional, default=False
		Print the step and the maximum error.
	bands: ndarray, optional, default=None
		Number of bans to be returned, if None all bands all returned.

	Returns
	-------
	nmax - step: int
		Minimum number of reciprocal-lattice vectors to reach convergence.
	J_final: array_like
		Converged band structure.
	Gamma_final: array_like
		Converged dissipation rates.
	"""

	# Compute G0 if it's not given
	if G0 is None:
		G0 = g0_fun(k0, aho)

	# Initialize the maximum error at infinity
	max_error = np.inf

	# Compute the first band structure with the initial guest of the number of reciprocal-lattice vectors
	J_1, Gamma_1 = band_structure(points_sites, b1, b2, A, nmax, k0, aho, n, G0=G0, polarizations=polarizations)
	counter = 0  # initialize the counter of iterations at 0

	# While the minimum error is not reached, and the maximum number of iterations is not reached
	while max_error > limit_error and counter < limit_counter:
		# Compute the band structure with a total of nmax + step reciprocal-lattice vectors
		J_2, Gamma_2 = band_structure(points_sites, b1, b2, A, nmax + step, k0, aho, n, G0=G0,
		                              polarizations=polarizations)

		# Compute the maximum error for all the band
		error_J = np.mean(np.abs(J_1 - J_2))
		error_Gamma = np.mean(np.abs(Gamma_1 - Gamma_2))
		max_error = np.mean((error_J, error_Gamma))

		# Update the number of reciprocal-lattice vectors, the counter and the band
		nmax += step
		counter += 1
		J_1 = J_2
		Gamma_1 = Gamma_2

		if print_progress:
			print('Step: {}, Avg error: {}'.format(counter, max_error))

	# If the maximum number of iterations is reached, show a warning
	if counter >= limit_counter:
		print('The calculation does not converge in the given number of steps')
		return 0

	if bands is None:
		bands = np.arange(0, J_1.shape[1])

	J_final = J_1[:, bands]
	Gamma_final = Gamma_1[:, bands]

	return nmax - step, J_final, Gamma_final


def band_structure(points_sites, b1, b2, A, nmax, k0, aho, n=100, G0=None, polarizations=None):
	"""
	Compute the band structure following straight line between the given points of the first Brillouin zone.

	Parameters
	----------
	points_sites: list
		Coordinates at which compute the band structure.
	b1: array_like
		Cartesian coordinates (x and y) for the first lattice vector of the reciprocal space.
	b2: array_like
		Cartesian coordinates (x and y) for the second lattice vector of the reciprocal space.
	A: float
		Area of the unit cell.
	nmax: int
		 Number of elements in the summatory for each direction.
	k0: float
		Wave number of the dipoles in the array.
	aho: float
		Distance associated with the cut-off in momentum.
	G0: float, optional, default=None
		Rescaled Green tensor at the origin. If None, it is calculated here.
	n: int, optional, default=100
		Number of momentum's between adjacent sites.
	polarizations: list, optional, default=None
		Polarization of the dipoles.

	Returns
	-------
	J: array_like
		Band structure.
	G: array_like
		Dissipation rates.
	"""

	# Compute G0 if it's not given
	if G0 is None:
		G0 = g0_fun(k0, aho)

	num = len(points_sites) - 1  # Number of intervals

	# Compute the number of bands
	if polarizations is None:
		n_bands = 3
	else:
		n_bands = len(polarizations)

	J = np.zeros([num * n, n_bands])  # Bands
	Gamma = np.zeros([num * n, n_bands])  # Decays
	k = np.zeros((n, 2))  # Momentum's for each interval

	# Iterate over each interval
	for i in range(num):
		# Compute the momentum's in the interval
		k[:, 0] = np.linspace(points_sites[i][0], points_sites[i + 1][0], n)
		k[:, 1] = np.linspace(points_sites[i][1], points_sites[i + 1][1], n)

		# Compute the hamiltonian in the interval
		J[i * n:(i + 1) * n, :], Gamma[i * n:(i + 1) * n, :] = compute_energies(b1, b2, A, nmax, k0, aho, k, G0=G0,
		                                                                        polarizations=polarizations)

	return J, Gamma


def compute_hamiltonian_real_space(r_lat, k0, pol_lat, pol_emi=None, r_emi=None, omega_emi=None, gamma_emi=None,
                                   plot=False):
	"""
	Compute the Hamiltonian in teh real space divided by Gamma_0 for a 2D finite array of dipoles. The origin of
	energies is set at omega_0/Gamma_0 by subtracting this quantity from the diagonal. If there is any array, they are
	included in the last indices.

	Parameters
	----------
	r_lat: list, (3, N_tot)
		3D coordinates in the cartesian basis for all the dipoles in the array.
	k0: float
		Wave number of the dipoles in the array.
	pol_lat: array_like (3)
		Polarization of all the dipoles in the array.
	pol_emi: List, optional, default=None
		Polarizations of the emitters, if any. Each dipole have and individual polarization given by one of the items
		in the list.
	r_emi: list (n_e, 3), optional, default=None
		3D coordinates in the cartesian basis for the total of n_e emitters in the array.
	omega_emi: list (n_e), optional, default=None
		Transitions for the emitters with respect to omega_0, divided by Gamma_0.
	gamma_emi: list (n_e), optional, default=None
		Individual decay rates for each emitter divided by Gamma_0.
	plot: bool, optional, default=False
		If True, plot the dipoles in 2D. The array is plotted in blue, the emitters in red.

	Returns
	-------
	Hamiltonian: array_like (N_tot + n_e, N_tot + n_e)
		Non-Hermitian Hamiltonian for the system of N_tot dipoles in the array, and n_e emitters.
	"""

	# Unpack the position of the dipoles in the array
	X_pos, Y_pos, Z_pos = r_lat
	del r_lat

	N_lat = len(X_pos)  # Number of dipoles in the array

	# Number of emitters
	if r_emi is None:
		r_emi = []

	n_e = len(r_emi)

	# Iterate over all the emitters
	for i in range(n_e):
		# Save the position of the emitter
		pos_emi = r_emi[i]
		X_pos = np.append(X_pos, pos_emi[0])
		Y_pos = np.append(Y_pos, pos_emi[1])
		Z_pos = np.append(Z_pos, pos_emi[2])

	# Plot the dipoles
	if plot:
		plt.figure(figsize=(5, 5))
		plt.scatter(X_pos[:N_lat], Y_pos[:N_lat], c='b')  # Array
		plt.scatter(X_pos[N_lat:], Y_pos[N_lat:], c='r')  # Emitters
		plt.axis('equal')
		plt.show()

	r_total = np.zeros((3, (N_lat + n_e) ** 2))  # Relative position between all the dipoles

	# Iterate over all dipoles
	for i in range(N_lat + n_e):
		r_total[:, i * (N_lat + n_e): (i + 1) * (N_lat + n_e)] = np.vstack(
			(X_pos[i] - X_pos, Y_pos[i] - Y_pos, Z_pos[i] - Z_pos))

	del X_pos, Y_pos, Z_pos

	# Compute the Green tensor
	Hamiltonian = np.zeros((N_lat + n_e, N_lat + n_e)) + 1j * 0  # Total Hamiltonian

	H_temp = green_dyadic_real_space(r_total, k0)
	H_temp *= -1
	H_temp = H_temp.reshape((3, 3, (N_lat + n_e), (N_lat + n_e)))

	del r_total

	H_temp = np.transpose(H_temp, (2, 3, 0, 1))  # Bring the dipoles indices to the first positions

	# Multiply by the polarization of the array and save the progress, matrix (N_lat x N_lat)
	Hamiltonian[:N_lat, :N_lat] = (pol_lat.T.conjugate() @ H_temp[:N_lat, :N_lat, :, :] @ pol_lat)[:, :, 0, 0]

	# Iterate over all the emitters
	for i in range(n_e):
		begin = N_lat + i  # Index denoting the emitter

		# Multiply by the polarization of the array and the emitter
		Hamiltonian[begin, :N_lat] = (pol_lat.T.conjugate() @ H_temp[begin, :N_lat, :, :] @ pol_emi[i])[:, 0,
		                             0] * np.sqrt(gamma_emi[i])  # Correct the individual decay

		# Same than before but instead for a row, now for the column, interchanging emitter <---> array
		Hamiltonian[:N_lat, begin] = (pol_emi[i].T.conjugate() @ H_temp[:N_lat, begin, :, :] @ pol_lat)[:, 0,
		                             0] * np.sqrt(gamma_emi[i])

		# Iterate over all pairs of emitters
		for j in range(n_e):
			if i != j:
				# Compute the element of the Hamiltonian corresponding to the interaction between emitters
				Hamiltonian[begin, begin - i + j] = \
					(pol_emi[j].T.conjugate() @ H_temp[begin, begin - i + j, :, :] @ pol_emi[i])[0, 0] * np.sqrt(
						gamma_emi[i] * gamma_emi[j])

	del H_temp
	Hamiltonian *= 3 * np.pi / k0

	# Identity in the array basis
	identity = -1j * np.eye(N_lat + n_e) / 2
	identity[N_lat:, N_lat:] = 0

	if n_e > 0:  # If there is some emitter
		for i in range(n_e):  # Iterate over all the emitters
			# Shift the diagonal elements for individual decays and transition
			identity[N_lat + i, N_lat + i] = omega_emi[i] - 1j * gamma_emi[i] / 2

	Hamiltonian += identity  # Shift the diagonal of the array dipoles for individual decays and detunning of emitters

	return Hamiltonian


def green_dyadic_real_space(r_vec, k0):
	"""
	Compute the Green tensor for a finite size array in the real space. This is NOT multiply be the factor 3 * pi / k0.
	The	function corresponds to Eq. (6) of A. Asenjo-García, et al., PRX 7, 031024 (2017).

	Parameters
	----------
	r_vec: array_like, (3, n)
		Vectors in 3D defining the distances between all the n dipoles (including emitter) in the array.
	k0: float
		Wave number of the dipoles.

	Returns
	-------
	g_tensor: numpy.ndarray (3, n, n)
		Green dyadic
	"""

	# If only one vector is given, reshape it for a matrix with 3 rows and only one column
	if np.shape(np.shape(r_vec))[0] == 1:
		r_vec = np.reshape(r_vec, (3, 1))

	N_tot = np.shape(r_vec)[1]

	# Compute the norm of the vectors
	r = np.sqrt(r_vec[0] ** 2 + r_vec[1] ** 2 + r_vec[2] ** 2)

	# Compute where the distance is 0, so we can eliminate these points from our calculations. The elimination is done
	# so no error raise
	index = np.where(r == 0)[0]

	if len(index) > 0:
		r[index] = 1

	factor = np.exp(1j * k0 * r) / (4 * np.pi * k0 ** 2 * r ** 3)  # Pre-factor in the Green tensor

	if len(index) > 0:
		factor[index] = 0

	# Compute each element of the equation
	sum1 = k0 ** 2 * r ** 2 + 1j * k0 * r - 1
	sum2 = (-k0 ** 2 * r ** 2 - 3j * k0 * r + 3) / r ** 2

	del r

	identity = np.zeros((3, 3, N_tot))  # Hypermatrix denoting the 3x3 identity matrix in the axis (0, 1)
	tensor = np.zeros((3, 3, N_tot))  # Tensor denoting r_vec x r_vec

	for i in range(3):  # Iterate all the cartesian coordinates
		identity[i, i, :] += 1  # Construct the identity matrix
		for j in range(3):  # Iterate all the cartesian coordinates
			tensor[i, j, :] = r_vec[i, :] * r_vec[j, :]  # Construct the tensor product

	g_tensor = factor * (sum1 * identity + sum2 * tensor)

	return g_tensor


def band_structure_real_space(energies, modes, full_output=False):
	"""
	Rearrange the energies of a finite size square array with total dipoles n^2 = N to plot the band structure in the
	momentum space. The energies and modes given are computed in real space.

	Parameters
	----------
	energies: array_like (N)
		Complex eigenenergies computed in real space.
	modes: array_like (n, n, N)
		Complex eigenmodes computed in real space. The mode associated to energies[i] corresponds to modes[:, :, i].
	full_output: bool, optional, default=False
		Return the energies and modes in the momentum space: if False, only the band following M -> Gamma -> X -> M.

	Returns
	-------
	energies_momentum: array_like (n, n)
		Matrix with the eigenenergies ordered in the momentum space, e.g. the energies energies_momentum[0, -1]
		corresponds to the momentum k = pi/a * (1, -1)
	"""

	n_y, n_x = modes.shape[:2]  # Number of dipoles

	modes_momentum = np.fft.fft2(modes, norm='ortho', axes=(0, 1))  # 2D Fourier Transform
	modes_momentum = np.fft.fftshift(modes_momentum,
	                                 axes=(0, 1))  # Shift axis so the momentum k = (0, 0) is in the center

	energies_momentum = np.zeros((n_y, n_x), dtype=complex)

	# Iterate over all sites
	for i in range(n_y):
		for j in range(n_x):
			temp = np.abs(modes_momentum[i, j, :])  # Modes with weight on k = pi/a * (-n + 2i, -n + 2j)/n
			index = np.where(np.max(temp) == temp)[0][0]  # Mode with higher weight
			energies_momentum[i, j] = energies[index]  # Save energy

	if full_output:
		return energies_momentum, modes_momentum

	else:
		band = np.zeros(3 * (n_x // 2), dtype=complex)

		for i in range(n_x // 2):  # Iterate over the momentum's between symmetry points
			band[i] = energies_momentum[-1 - i, i]  # M -> Gamma
			band[(n_x // 2) + i] = energies_momentum[n_x // 2 + i, n_x // 2]  # Gamma -> X
			band[2 * (n_x // 2) + i] = energies_momentum[-1, n_x // 2 - i]  # X -> M

		return band


def generate_alpha(r_j, polarization, k0):
	"""
	Compute the 2D version of the parameter defined in S. J. Masson, et al., PRR 2, 043213 (2020) Eq. (A9a) needed to
	compute the Purcell factor (optical depth).

	Parameters
	----------
	r_j: array_like (3, Ny, Nx)
		Three cartesian coordinates for r_emitter - r_i for the N_x * N_y total dipoles in the system.
	polarization: array_like (3, 1)
		Polarization of the lattice.
	k0: float
		Wave number associated to the transition of the dipoles

	Returns
	-------
	alpha: array_like (3, Ny, Nx)
		Final result.
	"""
	Ny, Nx = r_j.shape[1:]  # Number of dipoles in the axis Y and X

	# Green tensor in real space for the pairs dipole-emitter
	G_0 = green_dyadic_real_space(r_j.reshape((3, -1)), k0)  # (3, 3, Nx * Ny)
	G_0 = (G_0.transpose((2, 0, 1)) @ polarization).squeeze()  # (Nx * Ny, 3)
	G_0 = G_0.reshape((Ny, Nx, 3))

	# Compute the inverse FT in 2D, and eliminate the normalization
	alpha = np.fft.ifft2(G_0, axes=(0, 1)) * Nx * Ny
	alpha = np.fft.fftshift(alpha, axes=(0, 1))  # Sort the momentum's in ascending order

	return alpha.transpose((2, 0, 1))  # (3, Ny, Nx)


def generate_beta(r_j, polarization, k0):
	"""
	Compute the 2D version of the parameter defined in S. J. Masson, et al., PRR 2, 043213 (2020) Eq. (A9b) needed to
	compute the Purcell factor (optical depth).

	Parameters
	----------
	r_j: array_like, (3, Ny, Nx)
		Three cartesian coordinates for r_emitter - r_i for the N_x * N_y total dipoles in the system.
	polarization: array_like, (3, 1)
		Polarization of the lattice.
	k0: float
		Wave number associated to the transition of the dipoles

	Returns
	-------
	beta: array_like, (3, Ny, Nx)
		Final result.
	"""
	Ny, Nx = r_j.shape[1:]  # Number of dipoles in the axis Y and X

	# Green tensor in real space for the pairs dipole-emitter
	G_0 = green_dyadic_real_space(-r_j.reshape((3, -1)), k0)  # (3, 3, Nx * Ny)
	G_0 = (polarization.T.conj() @ G_0.transpose((2, 0, 1))).squeeze()  # (Nx * Ny, 3)
	G_0 = G_0.reshape((Ny, Nx, 3))

	# Compute the FT in 2D
	beta = np.fft.fft2(G_0, axes=(0, 1))
	beta = np.fft.fftshift(beta, axes=(0, 1))  # Sort the momentum's in ascending order

	return beta.transpose((2, 0, 1))  # (3, Ny, Nx)


def purcell_factor(r_lat, r_emi, k_xy, omega_k, omega_q, k0, pol_lat, pol_emi, a=1):
	"""
	Compute the decay rates into free space and into the 2D array for an emitter. This function follows a 2D version of
	the	Eq. (A8) given in S. J. Masson, et al., PRR 2, 043213 (2020).

	Parameters
	----------
	r_lat: list, (3)
		List with three indices, denoting the cartesian coordinates of the dipoles in the array. Each element of the
		list is a ndarray of shape (Ny, Nx) denoting each dipole. The integral is computed with the romb method,
		so Ny, Nx = 2^n + 1, where n is a natural number.
	r_emi: list (3)
		Three indices for the cartesian coordinates of the emitter.
	k_xy: array_like, (Ny, Nx)
		Bloch momentum's inside the first Brillouin zone.
	omega_k: ndarray, (Ny, Nx)
		Complex energies corresponding to the given in k_xy.
	omega_q: float
		Energy of the emitter divided by Gamma_0, with respect to omega_0.
	k0: float
		Wave number associated to the transition of the dipoles.
	pol_lat: array_like, (3, 1)
		Polarization of the lattice.
	pol_emi: array_like, (3, 1)
		Polarization of the emitter.
	a: float, optional, default=1
		Interatomic distance for the dipoles in the array.

	Returns
	-------
	Gamma_prime: float
		Decay rate of an emitter into free space, in units of Gamma_0^q.
	Gamma_2D: float
		Decay rate of an emitter into the 2D array, in units of Gamma_0^q.
	"""

	Ny, Nx = r_lat[0].shape  # Number of dipoles in the Y and X axis

	# Parameter needed to shift the pole in the integral to +i * epsilon. Close to the momentum discretization
	epsilon = 2 * np.pi / Nx

	# Array with the relative positions between all dipoles and the emitter: r_emitter - r_j
	r_j = np.zeros((3, Ny, Nx))
	for i in range(3):  # Iterate over the three cartesian coordinates
		r_j[i, :, :] = r_emi[i] - r_lat[i]

	# Compute the parameters alpha and beta
	alpha = generate_alpha(r_j, pol_lat, k0)
	beta = generate_beta(r_j, pol_lat, k0)

	# Compute the numerator of the integral
	numerator = np.zeros((Ny, Nx, 3, 3), dtype=complex)
	# Tensor product
	for i in range(3):
		for j in range(3):
			numerator[:, :, i, j] = alpha[i] * beta[j]
	numerator = (pol_emi.T.conj() @ numerator @ pol_emi)[:, :, 0, 0]  # Contract with the polarization of the emitter

	# Denominator of the integral, with the shifted poles in the complex plane
	denominator = omega_q - omega_k + 1j * epsilon

	integrand = numerator / denominator  # Function to integrate

	# Integrate over all the first Brillouin zone, first in Y and later in X
	integral = romb(integrand, 2 * np.pi / Ny, axis=0)
	integral = romb(integral, 2 * np.pi / Nx)

	Gamma_q = 1 - 9 * a ** 2 / (2 * k0 ** 2) * np.imag(
		integral)  # Total decay rate of the emitter due free space + the presence of the array

	# Now we compute the integral for momentum's inside the light cone, aka decay into free space
	k_abs = np.sqrt(k_xy[0] ** 2 + k_xy[1] ** 2)  # Compute the distance in the momentum space
	mask = np.abs(k_abs) > k0  # Momentum's outside the light cone

	numerator[mask] = 0  # The integral is zero outside the light cone
	integrand = numerator / denominator  # New function to integrate

	# Integrate over all the first Brillouin zone, first in Y and later in X
	integral = romb(integrand, 2 * np.pi / Ny, axis=0)
	integral = romb(integral, 2 * np.pi / Nx)

	Gamma_prime = 1 - 9 * a ** 2 / (2 * k0 ** 2) * np.imag(integral)  # Decay rate of the emitter into free space

	Gamma_2D = Gamma_q - Gamma_prime  # Decay rate of the emitter into the 2D array

	return Gamma_prime, Gamma_2D


def change_borders(hamiltonian, max_index, function, r_pos, return_indices=False, args=None):
	"""
	Change the original hamiltonian for a finite array so the individual decay rates of the boundary increase smoothly
	in order to imitate an infinite array so the excitons does not bounce on the borders.

	Parameters
	----------
	hamiltonian: array_like, (N_T + n_e, n_T + n_e)
		Original Hamiltonian for the total of N_T = n_x * n_y dipoles in the array plus the possible emitters.
	max_index: float
		Maximum penetration of the change.
	function: function
		Smooth function for the increase in the individual decay rate. The first index is the distance from the origin
		to the dipoles, the second one is the minimum distance, where the decay rate start to increase, and the third
		one is the maximum distance. Optional arguments are also allowed.
	r_pos: list, (3, N_tot)
		3D coordinates in the cartesian basis for all the dipoles in the array.
	return_indices: bool, optional, default=False
		If True, return the indices in the Hamiltonian of the modified dipoles.
	args: dic, optional, default=None
		Extra parameters of function
	Returns
	-------
	dipoles_indices: list
		Indices of the dipoles with modified decay rate
	"""

	if args is None:
		args = {}

	r = np.sqrt(r_pos[0] ** 2 + r_pos[1] ** 2)
	dipoles_indices = np.where(r > (np.max(np.abs(r_pos[0])) - max_index))

	# Compute the new decay rate
	new_decay = function(r[dipoles_indices], np.min(r[dipoles_indices]), np.max(r[dipoles_indices]), **args)

	# Change the diagonal with the new decay rate
	hamiltonian[dipoles_indices, dipoles_indices] = -1j * new_decay / 2

	if return_indices:
		return dipoles_indices


def quadratic(r, min_r, max_r, max_gamma=1.5):
	"""
	Quadratic function.

	Parameters
	----------
	r: array_like, (N_T)
		Distances from the original for all the N_T dipoles in the array.
	min_r: float
		Minimum distance.
	max_r: float
		Maximum distance
	max_gamma: float
		Maximum value for the function at r = r_max

	Returns
	-------
	array_like, (N_T)
		Quadratic function for r
	"""
	r = r - min_r
	A = (max_gamma - 1) / (max_r - min_r) ** 2
	B = 1
	return A * r ** 2 + B


def midpoint_circle(Nx, Ny, r):
	"""
	Compute the indices for the sites in a square array, close to a circle centered in the origin and with radius r.

	Parameters
	----------
	Nx: int
		Number of array dipoles in the x-axis.
	Ny: int
		Number of array dipoles in the y-axis.
	r: float
		Radius of the circle.

	Returns
	-------
	border_index: list
		Indices of the dipoles close to the circle.
	"""
	if Nx % 2 == 0:
		r += 0.5

	# Dipoles positions
	X_atom = np.arange(0, Nx) - Nx / 2 + 1 / 2
	Y_atom = np.arange(0, Ny) - Ny / 2 + 1 / 2

	X_atom, Y_atom = np.meshgrid(X_atom, Y_atom)
	X_atom = X_atom.flatten()
	Y_atom = Y_atom.flatten()

	y = 0
	x = r
	p = 1 - r

	x_circle = [x]
	y_circle = [y]

	while x > y:
		y += 1

		if p <= 0:
			p += (2 * y + 1)
		else:
			x -= 1
			p += 2 * (y - x) + 1

		x_circle.append(x)
		y_circle.append(y)

	x_circle = np.array(x_circle)
	y_circle = np.array(y_circle)

	x_total = np.append(x_circle, y_circle[::-1])
	y_total = np.append(y_circle, x_circle[::-1])

	x_total = np.append(x_total, -x_total[::-1])
	y_total = np.append(y_total, y_total[::-1])

	x_total = np.append(x_total, x_total[::-1])
	y_total = np.append(y_total, -y_total[::-1])

	coordinates = np.vstack([x_total, y_total])
	_, indices = np.unique(coordinates, axis=1, return_index=True)
	coordinates = (coordinates.T[np.sort(indices)]).T

	if Nx % 2 == 0:
		for coordinate in range(2):
			coordinates = coordinates.T[np.where(coordinates[coordinate] != 0)[0]].T
			for sign in range(2):
				coordinates[coordinate, np.where((-1) ** sign * coordinates[coordinate] > 0)] = coordinates[
					                                                                                coordinate, np.where(
						                                                                                (-1) ** sign *
						                                                                                coordinates[
							                                                                                coordinate] > 0)] - (
					                                                                                -1) ** sign * 0.5

	border_index = []
	for i in range(len(coordinates.T)):
		x, y = coordinates[:, i]
		try:
			border_index.append(np.where((X_atom == x) * (Y_atom == y))[0][0])
		except:
			pass

	return border_index


def complete_dynamics(r_pos, k0, pol_lat, dt, tf, psi0, N_x=None, N_y=None, emitter=None, pol_emi=None, gamma_emi=None,
                      omega_emi=None, border=None, decay_fun=quadratic, max_gamma=None, theta_max=None, r_circles=None,
                      type_border=midpoint_circle, progress_bar=False, plot=False, verify=False, limit_verify=1e-5,
                      window=100, factor_increase=10, counter_max=100):
	"""
	Compute the complete dynamics for the system of an array and some emitters, in a given initial state. After the
	dynamics is computed, the quality factor and the chirality are also computed.

	Parameters
	----------
	r_pos: list, (3, N_tot)
		3D coordinates in the cartesian basis for all the dipoles in the array.
	k0: float
		Wave number of the dipoles in the array.
	pol_lat: array_like (3)
		Polarization of all the dipoles in the array.
	dt: float
		Time step for the calculation.
	tf: float
		Total time of the dynamics.
	psi0: array_like, (N_T + n_e)
		Initial wave function.
	N_x: int, optional, default=None
		Number of array dipoles in the x-axis. If None, the chirality is not computed
	N_y: int, optional, default=None
		Number of array dipoles in the y-axis. If None, the chirality is not computed
	emitter: List, (n_e, 3), optional, default=None
		3D coordinates in the cartesian basis for the total of n_e emitters in the array.
	pol_emi: List, optional, default=None
		Polarizations of the emitters, if any. Each dipole have and individual polarization given by one of the items.
	gamma_emi: list (n_e), optional, default=None
		Individual decay rates for each emitter divided by Gamma_0.
	omega_emi: list (n_e), optional, default=None
		Transitions for the emitters with respect to omega_0, divided by Gamma_0.
	border: float, optional, default=None
		Maximum penetration of the change.
	decay_fun: function, optional, default=quadratic
		Smooth function for the increase in the individual decay rate.
	max_gamma: float, optional, default=None
		Maximum increase of the decay rate, divided by Gamma_0.
	theta_max: float, optional, default=None
		Angle of reference for the computation of the chirality. If None, the angle of maximum population is chosen as
		the reference.
	r_circles: list, optional, default=(0, 1, 2, 3, 4)
		Number of circles to average the chirality.
	type_border: str, optional, default=midpoint_circle
		Type of border to compute the chirality.
	progress_bar: bool, optional, default=False
		Print the progress bar for the dynamics.
	plot: bool, optional, default=False
		Plot the array and the emitters.
	verify: bool, optional, default=False
		If True, verify that the numerical Purcell factor reaches a constant value. The function increase the time step
		and the total time until the numerical final time derivative of the Purcell factor is lower than limit_verify.
	limit_verify: float, optional, default=1e-5
		Minimum value for the time derivative of the Purcell factor at the final time.
	window: int, optional, default=100
		All values of dQ_n[-window:] must be lower than limit_verify so the verification is passed.
	factor_increase: float, optional, default=10
		If the verification is not passed, then the time step and the final time are multiplied by the given factor.
	counter_max: int, optional, default=100
		Maximum iterations for the increase of the time step and total time. If the limit is not reached before, an
		exception is sent.

	Returns
	-------
	dict
		Dictionary with the following information:
			U: array_like, (N_T + n_e, N_T + n_e)
				Matrix exponential of exp(1j * dt * H).
			Hamiltonian: array_like, (N_T + n_e, N_T + n_e)
				Hamiltonian for the system.
			indices_border: list
				Indices of the dipoles with modified decay rate.
			psi: array_like, (n, N_T + n_e)
				Wave function for the system at all n times.
			time: array_like, (n)
				Array of times, from 0 to tf, with a step of dt.
			Q_n: array_like, (n - 1)
				Quality factor at all times except the initial one.
			chirality: array_like, (len(n_circles))
				Chirality for each circle.
			n_M: float
				Non_Markovianity of the dynamics.
			theta_max_value: array_like, (len(n_circles))
				Origin of theta for the calculation of the chirality, for each circle.
	"""
	global counter_verify
	if pol_emi is not None:
		n_e = len(pol_emi)
	else:
		n_e = 0

	Hamiltonian, U, indices_border = compute_U(r_pos, k0, pol_lat, dt, emitter=emitter, pol_emi=pol_emi,
	                                           gamma_emi=gamma_emi, omega_emi=omega_emi, border=border,
	                                           decay_fun=decay_fun, max_gamma=max_gamma, return_indices=True, plot=plot)

	psi, time = compute_evolution_U(U, tf, dt, psi0=psi0, progress_bar=progress_bar)

	if indices_border is not None:
		Q_n = quality_factor(Hamiltonian, indices_border[0], psi, time, n_e)
	else:
		Q_n = None

	if Q_n is not None and verify:
		counter_verify += 1
		dQ_n = np.gradient(Q_n, dt)
		if np.any(np.abs(dQ_n[-window:]) > limit_verify):
			if counter_verify > counter_max:
				counter_verify = 0  # Reset the counter
				raise Exception('The maximum number of iterations for the verification has been reached.')
			else:
				return complete_dynamics(r_pos, k0, pol_lat, dt * factor_increase, tf * factor_increase, psi0, N_x, N_y,
				                         emitter, pol_emi, gamma_emi, omega_emi, border, decay_fun, max_gamma,
				                         theta_max, r_circles, type_border, progress_bar, plot, verify, limit_verify,
				                         window, factor_increase, counter_max)
		counter_verify = 0  # Reset the counter

	if r_circles is None:
		r_circles = np.arange(0, 5)

	n_circles = len(r_circles)
	theta_max_value = np.zeros(n_circles)
	chirality = np.zeros(n_circles)

	if N_x is not None and N_y is not None:
		for i in range(n_circles):
			chirality[i], theta_max_value[i] = compute_chirality(psi, time, r_pos, r_circles[i], N_x, N_y,
			                                                     theta_max=theta_max, shape=type_border,
			                                                     return_theta_max=True)
	else:
		chirality = None
		theta_max_value = None

	if pol_emi is not None:
		n_M = non_markovianity(np.sum(psi[:, -n_e:], axis=1), time)
	else:
		n_M = None

	return {'U': U, 'Hamiltonian': Hamiltonian, 'indices_border': indices_border, 'psi': psi, 'time': time, 'Q_n': Q_n,
	        'chirality': chirality, 'n_M': n_M, 'theta_max_value': theta_max_value}


def quality_factor(Hamiltonian, indices_border, psi, time, n_e):
	"""
	Compute the quality factor (Purcell factor, optical depth, ...) in the numerical approach.

	Parameters
	----------
	Hamiltonian: array_like, (N_T + n_e, N_T + n_e)
		Hamiltonian for the system.
	indices_border: list
		Indices of the dipoles with modified decay rate.
	psi: array_like, (n, N_T + n_e)
		Wave function for the system at all n times.
	time: array_like, (n)
		Array of times, from 0 to tf, with a step of dt.
	n_e: int
		Number of emitters.

	Returns
	-------
	Q_n: array_like, (n - 1)
		Quality factor at all times except the initial one.
	"""
	k_prime = -2 * np.imag(np.diag(Hamiltonian))[indices_border] - 1
	c_border = psi[:, indices_border]
	integrand = np.sum(k_prime * np.abs(c_border) ** 2, axis=1)

	L_b = cumtrapz(integrand, time, initial=0)
	L_T = 1 - np.sum(np.abs(psi) ** 2, axis=1)
	p_L = np.sum(np.abs(psi[:, :-n_e]) ** 2, axis=1)

	Q_n = (p_L[1:] + L_b[1:]) / (L_T[1:] - L_b[1:])

	return Q_n


def compute_evolution(r_pos, k0, pol_lat, tf, dt, emitter=None, pol_emi=None, gamma_emi=None, omega_emi=None,
                      border=None, psi0=None, return_Hamiltonian=False, return_U=False):
	"""
	Compute the dynamics of the system.

	Parameters
	----------
	r_pos: list, (3, N_tot)
		3D coordinates in the cartesian basis for all the dipoles in the array.
	k0: float
		Wave number of the dipoles in the array.
	pol_lat: array_like (3)
		Polarization of all the dipoles in the array.
	dt: float
		Time step for the calculation.
	tf: float
		Total time of the dynamics.
	emitter: List, (n_e, 3), optional, default=None
		3D coordinates in the cartesian basis for the total of n_e emitters in the array.
	pol_emi: List, optional, default=None
		Polarizations of the emitters, if any. Each dipole have and individual polarization given by one of the items.
	gamma_emi: list (n_e), optional, default=None
		Individual decay rates for each emitter divided by Gamma_0.
	omega_emi: list (n_e), optional, default=None
		Transitions for the emitters with respect to omega_0, divided by Gamma_0.
	border: float, optional, default=None
		Maximum penetration of the change.
	psi0: array_like, (N_T + n_e), optional, default=None
		Initial wave function. If None, the system start with the excitation in the last emitter.
	return_Hamiltonian: bool, optional, default=False
		If True, return the Hamiltonian.
	return_U: bool, optional, default=False
		If True, return the evolution matrix.

	Returns
	-------
	Hamiltonian: array_like, (N_T + n_e, N_T + n_e)
		Hamiltonian for the system.
	U: array_like, (N_T + n_e, N_T + n_e)
		Matrix exponential of exp(1j * dt * H).
	psi: array_like, (n, N_T + n_e)
		Wave function for the system at all n times.
	time: array_like, (n)
		Array of times, from 0 to tf, with a step of dt.
	"""
	time = np.arange(0, tf, dt)

	Hamiltonian, U = compute_U(r_pos, k0, pol_lat, dt, emitter=emitter, pol_emi=pol_emi, gamma_emi=gamma_emi,
	                           omega_emi=omega_emi, border=border)

	psi = np.zeros([len(time), np.shape(U)[0]], dtype=complex)

	if psi0 is None:
		psi[0, -1] = 1
	else:
		psi[0] = psi0

	for i in range(1, len(time)):
		psi[i, :] = U @ psi[i - 1, :]

	output = []

	if return_Hamiltonian:
		output.append(Hamiltonian)

	if return_U:
		output.append(U)

	output.append(psi)
	output.append(time)

	return output


def W_1D(theta, theta_max):
	"""
	Compute the weight function for a unidirectional emission.

	Parameters
	----------
	theta: array_like, (n_B)
		Angle for the dipoles in the border, with respect to the x-axis.
	theta_max: float
		Angle of reference

	Returns
	-------
	array_like, (n_B)
		Weight function
	"""
	return np.cos(2 * (theta - theta_max))


def compute_chirality(psi, time, r_pos, border, N_x, N_y, theta_max=None, shape=midpoint_circle, return_data=False,
                      return_theta_max=False, W_fun=W_1D):
	"""
	Compute the chirality for the given border.

	Parameters
	----------
	psi: array_like, (n, N_T + n_e)
		Wave function for the system at all n times.
	time: array_like, (n)
		Array of times, from 0 to tf, with a step of dt.
	r_pos: list, (3, N_tot)
		3D coordinates in the cartesian basis for all the dipoles in the array.
	border: float
		Radius of the circle where the chirality is computed
	N_x: int
		Number of array dipoles in the x-axis.
	N_y: int
		Number of array dipoles in the y-axis.
	theta_max: float, optional, default=None
		Angle of reference for the computation of the chirality. If None, the angle of maximum population is chosen as
		the reference.
	shape: str, optional, default=midpoint_circle
		Type of border to compute the chirality.
	return_data: bool, optional, default=False
		If True, return the data for the cumulative population and the angles of the dipoles.
	return_theta_max: bool, optional, default=False
		If True, return the angle of reference.
	W_fun: function, optional, default=W_1D
		Weight function

	Returns
	-------
	chi: float
		Chirality at the given circle
	"""
	x_pos, y_pos, z_pos = r_pos

	data = np.abs(psi[:, :N_x * N_y]) ** 2
	data = data.reshape((len(time), N_y, N_x))

	T, indices_border = cumulative_population(time, data, border, shape=shape)

	theta = np.arccos(x_pos[indices_border] / np.sqrt(x_pos[indices_border] ** 2 + y_pos[indices_border] ** 2))
	theta[np.where(y_pos[indices_border] < 0)] *= -1
	theta = theta % (2 * np.pi)

	indices_sort = np.argsort(theta)
	T = T[indices_sort]
	theta = theta[indices_sort]

	theta, indices = np.unique(theta, return_index=True)
	T = T[indices]

	chi = chirality_fun(T, theta, W_fun=W_fun, theta_max=theta_max)

	if theta_max is not None and return_theta_max:
		temp = np.argmax(T)
		theta_max = theta[temp]

	data_return = [chi]

	if return_data:
		data_return.append(T)
		data_return.append(theta)
	if return_theta_max:
		data_return.append(theta_max)

	if len(data_return) == 1:
		data_return = data_return[0]

	return data_return


def chirality_fun(T, theta, W_fun=W_1D, theta_max=None):
	"""
	Compute the chirality defined as sum_i(T_i * W_i), where T_i is the cumulative population of the dipole (i)
	located in the border of a circle centered in (0, 0), and W_i is a weigh function.

	Parameters
	----------
	T: array_like, (N)
		Cumulative population of the N dipoles in the border of the circle
	theta: array_like, (N)
		Angle of each dipole
	W_fun: function, optional, default=W_1D
		Weight function
	theta_max: float, optional, default=None
		Angle of reference for the computation of the chirality. If None, the angle of maximum population is chosen as
		the reference.

	Returns
	-------
	chi: float
		Chirality at the given circle
	"""

	if theta_max is None:
		theta_max = theta[np.argmax(T)]

	W = W_fun(theta, theta_max)  # Weigh function
	chi = np.sum(T * W)  # Sum over the dipoles in the border

	return chi


def compute_U(r_pos, k0, pol_lat, dt, emitter=None, pol_emi=None, gamma_emi=None, omega_emi=None, border=None,
              max_gamma=2, decay_fun=quadratic, return_indices=False, plot=False):
	"""
	Compute the evolution matrix.

	Parameters
	----------
	r_pos: list, (3, N_tot)
		3D coordinates in the cartesian basis for all the dipoles in the array.
	k0: float
		Wave number of the dipoles in the array.
	pol_lat: array_like (3)
		Polarization of all the dipoles in the array.
	dt: float
		Time step for the calculation.
	emitter: List, (n_e, 3), optional, default=None
		3D coordinates in the cartesian basis for the total of n_e emitters in the array.
	pol_emi: List, optional, default=None
		Polarizations of the emitters, if any. Each dipole have and individual polarization given by one of the items.
	gamma_emi: list (n_e), optional, default=None
		Individual decay rates for each emitter divided by Gamma_0.
	omega_emi: list (n_e), optional, default=None
		Transitions for the emitters with respect to omega_0, divided by Gamma_0.
	border: float, optional, default=None
		Maximum penetration of the change.
	max_gamma: float, optional, default=None
		Maximum increase of the decay rate, divided by Gamma_0.
	decay_fun: function, optional, default=quadratic
		Smooth function for the increase in the individual decay rate.
	return_indices: bool, optional, default=False
		If True, return the indices in the Hamiltonian of the modified dipoles.
	plot: bool, optional, default=False
		Plot the array and the emitters.

	Returns
	-------
	tuple, (2 or 3)
		hamiltonian: array_like, (N_T + n_e, n_T + n_e)
			Hamiltonian for the total of N_T = n_x * n_y dipoles in the array plus the possible emitters.
		U: array_like, (N_T + n_e, N_T + n_e)
			Matrix exponential of exp(1j * dt * H).
		indices_border: list
			Indices of the dipoles with modified decay rate.
	"""
	if pol_emi is not None:
		n_e = len(pol_emi)

		Hamiltonian = compute_hamiltonian_real_space(r_pos, k0, pol_lat, r_emi=emitter, pol_emi=pol_emi,
		                                             gamma_emi=np.array([gamma_emi] * n_e),
		                                             omega_emi=np.array([omega_emi] * n_e), plot=plot)
	else:
		Hamiltonian = compute_hamiltonian_real_space(r_pos, k0, pol_lat)

	indices_border = None  # Ensure that the variable exists
	if border is not None:
		if not return_indices:
			change_borders(Hamiltonian, border, decay_fun, r_pos, args={'max_gamma': max_gamma})
		else:
			indices_border = change_borders(Hamiltonian, border, decay_fun, r_pos, args={'max_gamma': max_gamma},
			                                return_indices=return_indices)

	U = expm(-1j * Hamiltonian * dt)

	returns = [Hamiltonian, U]

	if return_indices:
		returns.append(indices_border)

	return returns


def compute_evolution_U(U, tf, dt, psi0=None, progress_bar=False, pbar_label=None):
	"""
	Compute the time evolution.

	Parameters
	----------
	U: array_like, (N_T + n_e, N_T + n_e)
		Matrix exponential of exp(1j * dt * H).
	tf: float
		Total time of the dynamics.
	dt: float
		Time step for the calculation.
	psi0: array_like, (N_T + n_e), optional, default=None
		Initial wave function. If None, the system start with the excitation in the last emitter.
	progress_bar: bool, optional, default=False
		Print the progress bar for the dynamics.
	pbar_label: str, optional, default='Time evolution'

	Returns
	-------
	psi: array_like, (n, N_T + n_e)
		Wave function for the system at all n times.
	time: array_like, (n)
		Array of times, from 0 to tf, with a step of dt.
	"""

	if pbar_label is None:
		pbar_label = '   Time evolution'

	time = np.arange(0, tf + dt, dt)
	psi = np.zeros([len(time), np.shape(U)[0]], dtype=complex)

	if psi0 is None:
		psi[0, -1] = 1
	else:
		psi[0] = psi0

	pbar = None  # Ensure that the variable exists
	if progress_bar:
		pbar = tqdm(total=len(time) - 1, desc=pbar_label)

	for i in range(1, len(time)):
		psi[i, :] = U @ psi[i - 1, :]

		if progress_bar:
			pbar.update()

	if progress_bar:
		pbar.close()

	return psi, time


def non_markovianity(psi_e, time):
	"""
	Compute the non-Markovianity following S. Lorenzo, et al., Sci Rep 7, 42729 (2017).

	Parameters
	----------
	psi_e: array_like, (n)
		Wave function of the emitter
	time: array_like, (n)
		Time of the dynamics

	Returns
	-------
	n_M: float
		Non-Markovianity
	"""
	c_e = np.abs(psi_e) ** 2
	c_e_prime = np.gradient(c_e, time)

	c_e_prime_temp = np.copy(c_e_prime)
	c_e_prime_temp[c_e_prime_temp < 0] = 0
	N_v = 1 / (time[-1] - time[0]) * simps(c_e_prime_temp, time)

	c_e_prime_temp = np.copy(c_e_prime)
	c_e_prime_temp[c_e_prime_temp > 0] = 0
	N_n = 1 / (time[-1] - time[0]) * simps(c_e_prime_temp, time)

	return N_v / np.abs(N_n)


def cumulative_population(time, population, border=0, shape=midpoint_circle):
	"""
	Compute the normalized cumulative population of the border of the bath, integrating the population over time.

	Parameters
	----------
	time: array_like, (n_t)
		Time of the evolution, in units of 1/Gamma_0^q.
	population: array_like, (n_t, Ny, Nx)
		|c_i(t)|^2 Population of the dipoles in the array for all times in real space.
	border: float, optional, default=0
		Distance from the border of the lattice
	shape: function, optional, default=midpoint_circle
		Function to detect the borders of the array. The first argument is the number of sites in the x-axis, the second
		one is the number of sites in the y-axis, and the last one is the radius of the circle.

	Returns
	-------
	T_i: array_like
		Cumulative population of the border of the array. If the shape is a square, the sequence follows the path:
		Upper left -> Lower left -> Lower right -> Upper right -> Upper left
	indices: array_like
		Indices of the border
	"""

	Ny, Nx = np.shape(population[0, :, :])  # Shape of the array

	indices = shape(Nx, Ny, (Nx - 1) / 2 - border)

	population_border = population.reshape((-1, Ny * Nx))[:, indices]  # Population of the border

	T_i = simps(population_border, time, axis=0)  # Cumulative population

	T_i = T_i / np.sum(T_i)  # Normalized population

	return T_i, indices


def compute_shift(r1, r2, pol1, pol2, k0, Gamma_q, psi0):
	"""
	Compute the shift in energy due to the hybridization between two emitters.

	Parameters
	----------
	r1: array_like (3)
		Cartesian coordinates for the first emitter.
	r2: array_like (3)
		Cartesian coordinates for the second emitter.
	pol1: array_like (3)
		Polarizations of the first emitters.
	pol2: array_like (3)
		Polarizations of the first emitters.
	k0: float
		Wave number of the dipoles in the array.
	Gamma_q: float
		Individual decay rates for the emitters divided by Gamma_0.
	psi0: array_like, (n_e)
		Initial wave function.

	Returns
	-------
	shift: complex
		Shift in energy. The correct energy is omega_q - shift.
	"""
	psi0 = psi0.reshape((2, 1))
	r = np.array(r1) - np.array(r2)
	G = green_dyadic_real_space(np.reshape(r, (3, 1)), k0)[:, :, 0]

	J12 = - Gamma_q * 3 * np.pi / k0 * (pol1.T.conjugate() @ G @ pol2)[0, 0]

	H = np.array([[-1j * Gamma_q / 2, J12], [J12, -1j * Gamma_q / 2]])
	shift = (psi0.T.conj() @ H @ psi0)[0, 0]

	return shift


def generate_hexagon(n, a=1):
	"""
	Generate the (x, y) coordinates of a filled hexagon for a triangular lattice with interatomic distance 'a', sheen in
	the momentum space.

	Parameters
	----------
	n: int
		Number of sites in each direction from the origin.
	a: float, optional, default=1
		Interatomic distance of the triangular lattice

	Returns
	-------
	(x, y): array_like
		Coordinates of thje sites in the hexagonal lattice.
	"""
	a1 = [a, 0]
	a2 = [a / 2, a * np.sqrt(3) / 2]
	x, y = [i.flatten() for i in lattice_sites(a1, a2, n, n)]
	x *= 4 / 3 * np.pi / a / n
	y *= 4 / 3 * np.pi / a / n

	r = lambda x, sign: -np.sqrt(3) * x + 4 * np.pi / (np.sqrt(3) * a) * sign

	mask_top = np.where(y < r(x, 1))
	x = x[mask_top]
	y = y[mask_top]

	mask_top = np.where(y > r(x, -1))
	x = x[mask_top]
	y = y[mask_top]

	return x, y
