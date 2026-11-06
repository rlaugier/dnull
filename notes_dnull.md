
# Strucutre:

* Sources 
	- DN_Source_BB
		+ DN__PointCollection
		+ DN_BB
	- 
* DN_ErrorPhasorPistonPointing
	- piston (array)
	- pointing (array)
	- fourDsquare (float) *Given by D at init*
	- *phasor*
	- *single_phasor*
	- *no_error* returns the neutral element of *phasor*

* DN_NOTT_LDC
	- piston
	- amplitude
	- co2_ppm
	- co2_length
	- glass_length
	- co2_eq_index
	- air_eq_index
	- glass_eq_index

* CalibSetup (NOTT_LDC_series, background)
	- DN_NOTT_LDC_series
	- DN_NOTT_LDC
		+ piston
		+ amplitude
		+ co2_ppm
		+ co2_length
		+ glass_length
		+ co2_eq_index
		+ air_eq_index
		+ glass_eq_index

* DN_Observation (dn_nifits, nuisance soucelist, interest_sourcelist, error_phasor)

* DN_Post
		