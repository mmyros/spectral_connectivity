import numpy as np
from pytest import mark
from unittest.mock import PropertyMock

import xarray as xr
from spectral_connectivity.connectivity import (Connectivity, _bandpass,
                                   _complex_inner_product,
                                   _conjugate_transpose,
                                   _find_largest_independent_group,
                                   _find_largest_significant_group,
                                   _get_independent_frequencies,
                                   _get_independent_frequency_step,
                                   _inner_combination,
                                   _remove_instantaneous_causality,
                                   _reshape, _set_diagonal_to_zero,
                                   _squared_magnitude, _total_inflow,
                                   _total_outflow)


def fourier_coefficients_to_xarray(fourier_coefficients,n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals):

    fourier_coefficients = xr.DataArray(data=fourier_coefficients,
                                        # coords=[range(mea) for mea in
                                        #        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)],
                                        dims=('time_samples', 'trials', 'tapers', 'fft_samples', 'signals'),
                                        #name=['fourier_coefficients']
                                        )
    return fourier_coefficients


for axis in range(3):
    '''Test that the cross spectrum is correct for each dimension.'''
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (
        20, 3, 4, 3, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals),
        dtype=np.complex)

    signal_fourier_coefficient = [2 * np.exp(1j * np.pi / 2),
                                  3 * np.exp(1j * -np.pi / 2)]
    fourier_ind = [slice(0, 4)] * 5
    fourier_ind[-1] = slice(None)
    fourier_ind[axis] = slice(1, 2)
    fourier_coefficients[fourier_ind] = signal_fourier_coefficient


    expected_cross_spectral_matrix = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals,
         n_signals), dtype=np.complex)

    expected_slice = np.array([[4, -6], [-6, 9]], dtype=np.complex)
    expected_ind = [slice(0, 5)] * 6
    expected_ind[-1] = slice(None)
    expected_ind[-2] = slice(None)
    expected_ind[axis] = slice(1, 2)
    expected_cross_spectral_matrix[expected_ind] = expected_slice

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(expected_cross_spectral_matrix, this_Conn._cross_spectral_matrix)

    #%%

    xfourier_coefficients=fourier_coefficients_to_xarray(fourier_coefficients, n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
    this_Conn = Connectivity(fourier_coefficients=xfourier_coefficients)
    this_Conn._cross_spectral_matrix
    thismat=this_Conn._cross_spectral_matrix
    assert np.allclose(expected_cross_spectral_matrix, this_Conn._cross_spectral_matrix)
