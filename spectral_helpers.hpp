#pragma once
#include <vector>

namespace spec {
using dvec = std::vector<double>;

/*
 * Compute relative spectral power distribution of a black-body radiator
 * at temperature T (Kelvin) for the supplied wavelength grid (nm).
 * Result is normalised so the maximum value = 1.0 (matches colour-science normalisation).
 */
void blackbody_spd(double temperature, const dvec& wavelengths_nm, dvec& out);

/*
 * Compute relative CIE daylight spectral distribution (D-series) for a given
 * correlated colour temperature (CCT, Kelvin).
 * Implementation follows CIE 15:2004, identical to colour-science daylight_spd.
 * Output is normalised to a maximum of 1.0.
 */
void daylight_spd(double cct, const dvec& wavelengths_nm, dvec& out);

} // namespace spec 