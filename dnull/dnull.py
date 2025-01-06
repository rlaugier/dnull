"""
    dnull offers a specialty backend ton NIFITS.
It does not facilitate the exportation back to NIFITS
nor the creation of NIFITS files from scratch.

Main classes:

* DN_NIFITS: regroups all the extensions and offers the
  computation methods.
* DN_xxxxxx: the different nifits extension objects
  mostly matching the objects of nifits.io objects
  should be mostly manipulated by the methods of DN_NIFITS
"""
from nifits import io
from nifits import backend

import jax
from jax import numpy as jp, scipy as jscipy
import zodiax as zdx

from astropy import units, constants as cst

import sys
def getclass(classname):
    return getattr(sys.modules[__name__], classname)


class DN_CPX(zdx.Base):
    real: jp.ndarray
    imag: jp.ndarray
    def __init__(self, complex_val: jp.complex64):
        self.real = jp.asarray(complex_val.real, dtype=float)
        self.imag = jp.asarray(complex_val.imag, dtype=float)

    @property
    def abs(self):
        return jp.hypot(self.real, self.imag)
    @property
    def angle(self):
        return jp.angle(self.real + 1j*self.imag)
    @property
    def cpx(self):
        return self.real + 1j*self.imag

    @classmethod
    def ones(cls, shape):
        complex_val = jp.ones(shape) + 1j*jp.zeros(shape)
        return cls(complex_val)

    @classmethod
    def ones_like(cls, example):
        complex_val = jp.ones_like(example) + 1j*jp.zeros_like(example)
        return cls(complex_val)
    
    @classmethod
    def zeros(cls, shape):
        complex_val = jp.zeros(shape) + 1j*jp.zeros(shape)
        return cls(complex_val)

    @classmethod
    def zeros_like(cls, example):
        complex_val = jp.zeros_like(example) + 1j*jp.zeros_like(example)
        return cls(complex_val)

class DN_WAVELENGTH(zdx.Base):
    """
    An object storing the OI_WAVELENGTH information, in compatibility with
    OIFITS practices.

    **Shorthands:**

    * ``self.lambs`` : ``jp.ndarray`` [m] returns an array containing the center
      of each spectral channel.
    * ``self.dlmabs`` : ``jp.ndarray`` [m] an array containing the spectral bin
      widths.
    * ``self.nus`` : ``jp.ndarray`` [Hz] an array containing central frequencies
      of the
      spectral channels.
    * ``self.dnus`` : ``jp.ndarray`` [Hz] an array containing the frequency bin
      widths.

    """
    lambs: jp.ndarray
    name: str

    def __init__(self, ni_wavelength: io.oifits.OI_WAVELENGTH):
        self.lambs = jp.asarray(ni_wavelength.lambs, dtype=float)
        self.name = ni_wavelength.name

    @property
    def dlambs(self):
        return jp.gradient(self.lambs)
    @property
    def nus(self):
        assert cst.c.unit == units.m/units.s, f"c has units {cst.c.unit}"
        return cst.c.value/self.lambs
    @property
    def dnus(self):
        assert cst.c.unit == units.m/units.s, f"c has units {cst.c.unit}"
        return cst.c.value/self.dlambs
    


DN_TARGET = io.oifits.OI_TARGET

class DN_IOUT(zdx.Base):
    iout: jp.ndarray
    unit: units.Unit

    def __init__(self, ni_iout: io.oifits.NI_IOUT):
        self.iout = jp.asarray(ni_iout.iout, dtype=float)
        self.unit = ni_iout.unit

class DN_CATM(zdx.Base):
    M: DN_CPX
    def __init__(self, ni_catm: io.oifits.NI_CATM):
        self.M = DN_CPX(ni_catm.M, dtype=jp.complex64)

class DN_KIOUT(zdx.Base):
    kiout: jp.ndarray
    unit: units.Unit
    def __init__(self, ni_kiout: io.oifits.NI_KIOUT):
        self.kiout = jp.asarray(ni_kiout.kiout, dtype=float)
        self.unit = ni_kiout.unit
    @property
    def shape(self):
        return self.kiout.shape

class DN_KCOV(zdx.Base):
    kcov: jp.ndarray
    unit: units.Unit
    def __init__(self, ni_kcov: io.oifits.NI_KCOV):
        self.kcov = jp.asarray(ni_kcov.kcov, dtype=float)
        self.unit = ni_kcov.unit

class DN_KMAT(zdx.Base):
    K: jp.ndarray
    def __init__(self, ni_kmat: io.oifits.NI_KMAT):
        self.K = jp.asarray(ni_kmat.K, dtype=float)




class DN_MOD(zdx.Base):
    time: jp.ndarray
    int_time: jp.ndarray
    mod_phas: DN_CPX
    app_xy: jp.ndarray
    arrcol: jp.ndarray
    fov_index: jp.ndarray

    def __init__(self, ni_mod: io.oifits.NI_MOD):
        self.time = jp.asarray(ni_mod.data_table["TIME"].data, dtype=float)
        self.int_time = jp.asarray(ni_mod.int_time, dtype=float)
        self.mod_phas = DN_CPX(ni_mod.all_phasors)
        self.app_xy = jp.asarray(ni_mod.appxy, dtype=float)
        self.arrcol = jp.asarray(ni_mod.arrcol, dtype=float)
        self.fov_index = jp.asarray(ni_mod.data_table["FOV_INDEX"].data,
                                                            dtype=int)

    @property
    def n_series(self):
        return len(self.time)

    @property
    def all_phasors(self):
        return self.mod_phas.cpx

    @property
    def dateobs(self):
        """
        Get the dateobs from the weighted mean of the observation time
        from each of the observation times given in the rows of ``NI_MOD``
        table.
        """
        raise NotImplementedError(self.dateobs)
        return None



class DN_FOV_diam_gau_rad(zdx.Base):
    D: jp.ndarray
    offset: jp.ndarray
    def __init__(self, ni_fov: io.oifits.NI_FOV):
        assert ni_fov.header["FOV_MODE"] == "diameter_gaussian_radial",\
                    NotImplementedError("Only diameter_gaussian_radial implemented")
        D = ni_fov.data_table["FOV_TELDIAM"]
        uD = units.Unit(ni_fov.data_table["FOV_TELDIAM_UNIT"])
        self.D = jp.asarray(D*uD.to(units.m), dtype=float)
        self.offset = jp.asarray(ni_fov.data_table["offsets"], dtype=float)

    def fov_function(self, x, y, lambs):
        """
        Returns the phasor corresponding to position in FOV

        Shape: [n_frames n_wavelengths n_points, ]
        """
        r_0 = (1/2*self.dn_wavelength.lambs/self.dn_fov_D)# *units.rad.to(units.mas)
        r = jp.hypot(x[None, None, :] - self.fov.offset[:,:, 0, None],
                        y[None,None, :] - self.fov.offset[:,:, 1, None])
        phasor = jp.exp(-(r[:,:]/r_0[:,None])**2)
        return phasor.astype(jp.complex64)



from typing import Union, List, Callable

DN_FOV_TYPE = Union[DN_FOV_diam_gau_rad,]

rad2mas = units.rad.to(units.mas)
mas2rad = units.mas.to(units.rad)


class DN_PointCollection(object):
    """
        A class to hold arrays of coordinates. Handy to compute
    the transmission map on a large number of points.

    **Units default to mas.**

    Args:
        aa    : [unit (mas)] (ArrayLike) first coordinate flat array, 
              typically RA.
        bb    : [unit (mas)] (ArrayLike) second coordinate flat array, 
              typically Dec.

    Constructors:
        * ``from_uniform_disk``   : 
        * ``from_grid``           :
        * ``from_centered_square``:
        * ``from_segment``        :

    Modificators:
        * ``__add__``         : basically a concatenation
        * ``transform``       : Linear transformation in 3D by a matrix

    Handles:
        * ``coords``         : The array values as first provided
        * ``coords_rad``     : The array values, converted from 
          ``self.unit`` into radians.
        * ``coords_quantity``: Return the values as a quantity.
        * ``coords_radial``  : Returns the radial coordinates (rho,theta)
        * ``extent``         : The [left, right, top, bottom] extent
          (used for some plots).
    """
    aa: jp.ndarray = None
    bb: jp.ndarray = None
    ds_mas2: jp.ndarray = None
    shape: tuple = None
    orig_shape: tuple = None
    unit: unit.Unit = None
    def __init__(self, aa, bb, ds_mas2, unit=units.mas,
                    shape=None, orig_shape=None):
        self.aa = aa
        self.bb = bb
        self.unit = unit
        self.shape = self.aa.shape
        self.ds_mas2 = ds_mas2
        if not hasattr(self, "orig_shape"):
            self.orig_shape = self.shape

    @classmethod
    def from_uniform_disk(cls, radius=None,
                        n: int = 10,
                        phi_0: float = 0.,
                        offset: jp.ndarray = jp.array((0.,0.)),
                        unit: units.Unit = units.mas):
        """
            Create a point collection as a uniformly sampled disk.

        Args:
            a_coords : The array of samples along the first axis
                  (typically alpha)
            b_coords : The array of samples along the second axis
                  (typically beta, the second dimension)

        **Handles:**
        """
        alpha = jp.pi * (3 - jp.sqrt(5))    # the "golden angle"
        points = []
        for k in jp.arange(n):
          theta = k * alpha + phi_0
          r = radius * jp.sqrt(float(k)/n)
          points.append((r * jp.cos(theta), r * jp.sin(theta)))
        points = jp.array(points).T + offset[:,None]
        total_s = jp.pi * radius**2
        ds_mas2 = total_s / n * jp.ones(n)
        myobj = cls(*points, ds_mas2, unit=unit)
        return myobj

    @classmethod
    def from_grid(cls, a_coords: jp.ndarray, b_coords: jp.ndarray,
                        unit: units.Unit = units.mas):
        """
            Create a point collection as a cartesian grid.

        Args:
            a_coords : The array of samples along the first axis
                  (typically alpha)
            b_coords : The array of samples along the second axis
                  (typically beta, the second dimension)
            unit : Units for ``a_coords`` and ``b_coords``

        **Handles:**
        """
        aa, bb = jp.meshgrid(a_coords, b_coords)
        original_shape = aa.shape
        aa = aa.flatten()
        bb = bb.flatten()
        das = jp.gradient(a_coords)
        dbs = jp.gradient(b_coords)
        ds_mas2 = (das[:,None] * dbs[None,:]).flatten()
        myobj = cls(aa=aa, bb=bb, ds_mas2=ds_mas2, unit=unit)
        myobj.orig_shape = original_shape
        return myobj

    @property
    def extent(self):
        extent = jp.array([jp.min(self.aa), jp.max(self.aa),
                    jp.min(self.bb), jp.max(self.bb)])
        return extent
        

    @classmethod
    def from_segment(cls, start_coords: jp.ndarray,
                        end_coords: jp.ndarray,
                        n_samples: int,
                        width: float,
                        unit: units.Unit = units.mas):
        """
            Create a point collection as a cartesian grid.

        Args:
            start_coords : The (a,b) array of the starting point.
                  (typically alpha, beta)
            end_coords : The (a,b) array of the ending point.
                  (typically alpha, beta)
            n_sameples   : The number of samples along the line.
            width        : The transverse angular size to consider 

        **Handles:**
        """
        aa = jp.linspace(start_coords[0], end_coords[0], n_samples, dtype=float)
        bb = jp.linspace(start_coords[1], end_coords[1], n_samples, dtype=float)
        ds_mas2 = jp.asarray(aa * width, dtype=float)
        original_shape = aa.shape
        aa = aa.flatten()
        bb = bb.flatten()
        myobj = cls(aa=aa, bb=bb, ds_mas2=ds_mas2)
        return myobj

    @classmethod
    def from_centered_square_grid(cls,
                        radius,
                        resolution,
                        ):
        """
            Create a centered square cartesian grid object

        Args:
            radius      : The radial extent of the grid.
            resolution  : The number of pixels across the width.
        """
        a_coords = jp.linspace(-radius, radius, resolution)
        b_coords = jp.linspace(-radius, radius, resolution)
        myobj = cls.from_grid(a_coords=a_coords,
                            b_coords=b_coords,
                            )
        return myobj
        

    @property
    def coords(self):
        """
        Returns a tuple with the ``alpha`` and ``beta`` coordinates in 
        flat arrays.
        """
        return (self.aa, self.bb)
    
    @property
    def coords_rad(self):
        return (mas2rad*self.aa, mas2rad*self.bb)

    @property
    def coords_radial(self):
        """
        Returns the radial coordinates of points. (rho, theta) ([unit], [rad]).
        """
        cpx = self.aa + 1j*self.bb
        return (jp.abs(cpx), jp.angle(cpx))

    @property
    def coords_shaped(self):
        if hasattr(self, "orig_shape"):
            if self.orig_shape:
                return (self.aa.reshape(self.orig_shape), self.bb.reshape(self.orig_shape))
            else:
                raise AttributeError("Original shape was None")
        else:
            raise AttributeError("Did not have an original shape")

    def transform(self, matrix):
        """
        Produce a linear transform of the coordinates.

        Args:
            matrix: A transformation matrix (3D)
        """
        if not hasattr(self, "cc"):
            self.cc = jp.zeros_like(self.aa)
        vectors = jp.vstack((self.aa, self.bb, self.cc))
        transformed = jp.dot(matrix, vectors)
        aa = transformed[0,:]
        bb = transformed[1,:]
        cc = transformed[2,:]
        ds_mas2 = self.ds_mas2*(aa/self.aa)*(bb/self.bb)
        return DN_PointCollection(aa, bb, self.ds_mas2, unit=self.units,
                    shape=self.shape, orig_shape=self.orig_shape)

    def __add__(self, other):
        return 


def test_attr(obj, name):
    if hasattr(obj, name):
        return obj.name is not None
    else:
        return False

class DN_NIFITS(zdx.Base):
    dn_catm: DN_CATM = None
    dn_fov: DN_FOV_TYPE = None
    dn_kmat: DN_KMAT = None
    dn_wavelength: DN_WAVELENGTH = None
    dn_target: DN_TARGET = None
    dn_mod: DN_MOD = None
    dn_iout: DN_IOUT = None
    dn_kiout: DN_KIOUT = None
    dn_kcov: DN_KCOV = None
    def __init__(self,
                dn_catm: DN_CATM = None,
                dn_fov: DN_FOV_TYPE = None,
                dn_kmat: DN_KMAT = None,
                dn_wavelength: DN_WAVELENGTH = None,
                dn_target: DN_TARGET = None,
                dn_mod: DN_MOD = None,
                dn_iout: DN_IOUT = None,
                dn_kiout: DN_KIOUT = None,
                dn_kcov: DN_KCOV = None):
        
        self.fov_function = self.fov.fov_function
        pass
    @classmethod
    def from_nifits(cls, anifits):
        names_dn = [
                    "dn_catm",
                    "dn_fov",
                    "dn_kmat",
                    "dn_wavelength",
                    "dn_target",
                    "dn_mod",
                    "dn_iout",
                    "dn_kiout",
                    "dn_kcov"]
        names_ni = [
                    "ni_catm",
                    "ni_fov",
                    "ni_kmat",
                    "ni_wavelength",
                    "ni_target",
                    "ni_mod",
                    "ni_iout",
                    "ni_kiout",
                    "ni_kcov"]
        extensions = {}
        for niname, dnname in zip(names_ni, names_dn):
            if test_attr(anifits, niname):
                myclass = getclass(dnname.upper())
                myobj = myclass(getattr(anifits, niname))
                mydn = myclass(myobj)
                extensions[dnname] = mydn
        return cls(**extensions)
            
        

    def fov_function(self):
        """
        This method is meant to be replaced by the specific FOV function
        during __init__

        """
        pass

    
        

    @property
    def n_collectors(self):
        self.dn_catm.M.shape[2]
    @property
    def n_outputs(self):
        self.dn_catm.M.shape[1]
    @property
    def n_wl(self):
        self.dn_catm.M.shape[0]
    @property
    def n_frames(self):
        self.dn_mod.all_phasors.shape[0]



class DN_BB(zdx.Base):
    temperature : jp.ndarray
    # wavelengths : jp.ndarray
    cross_section : jp.ndarray
    col_dens : jp.ndarray
    def __init__(self, temperature,
                    cross_section=None, col_dens=None):
        """
        Args:
            temperature (float): Temperature [K]
            cross_section (array): Molecular cross-section [m^2]
            opacity (float): column density of the element [m^-2]
            # wavelengths (array): Light wavelengths [m]

        """
        self.temperature = jp.asarray(temperature, dtype=float)
        # self.wavelengths = jp.asarray(wavelengths, dtype=float)
        if col_dens is None:
            self.col_dens = jp.asarray(1., dtype=float)
        else:
            self.col_dens = jp.asarray(col_dens, dype=float)
        if cross_section is None:
            self.cross_section = jp.asarray(1., dtype=float)
        else:
            self.cross_section = jp.asarray(cross_section, dtype=float)
            
        h = cst.h
        assert h.unit == units.J*units.s, f"Unexpected unit for Planck's constant {h.unit}"
        # self.h = h.value
        c = cst.c
        assert c.unit == units.m/units.s, f"Unexpected unit for c {c.unit}"
        # self.c = c
        # self.outunits = units.J /units.s /units.m**2 /units.Hz
        kb = cst.k_B
        assert kb.unit == units.J/units.K, f"Unexpected unit for c {kb.unit}"
    @property
    def outunits(self):
        return units.J /units.s /units.m**2 /units.Hz/units.sr
            
    def model(self, wavelengths):
        """
        Computes B_nu(lambda) at T x opacity
        """
        nu = cst.c.value / wavelengths 
        # term1 = 2 * cst.h * nu**3 / cst.c**2
        term1 = 2 * jp.exp(jp.log(cst.h.value) + 3*jp.log(nu) - 2*jp.log(cst.c.value))
        term2 = 1/jp.expm1( (cst.h.value*nu) / (cst.k_B.value * self.temperature) )
        return  self.cross_section * self.col_dens * term1 * term2

class DN_Source_BB(zdx.Base):
    locs: DN_PointCollection
    blackbody: DN_BB
    def __init__(self, locs: DN_PointCollection,
                    blackbody: DN_BB):
        pass
    @property
    def irradiance(self):
        pass
    @classmethod
    def radial_from_bond_albedo(cls, locs, transformation,
                             cross_section: jp.ndarray = None,
                             radial_exp_density: float = 2.3,
                             radial_k: float = 2.0,
                             A_b: float = 0.5,
                             R_star: float = 1.,
                             T_star: float = 5300,
                             sub_rad: float = 0.5, 
                             distance: float = 10):
        r_mas = locs.coord_radial()[0]
        r_au = units.pc.to(units.au) * distance * mas2rad * r_mas
        Teq = T_star*jp.sqrt(R_star/r_au)*(1/4*(1-A_b))**(1/4)
        density = radial_k * r_au**radial_exp_density
        density = jp.where(r_au<=0, 0., density)
        blackbody = DN_BB(Teq, cross_section=cross_section, col_dens=density)
        cls(locs, blackbody)

class DN_Source_Base(zdx.Base):
    locs: DN_PointCollection
    irradiance: jp.ndarray

    def __init__(self, lap)

class DN_Source_Layered(zdx.Base):
    continuum: DN_BB
    layers: list
    def __init__(self, continuum: DN_Source_BB, layers: list):
        self.continuum = continuum
        self.layers = layers
    def get_spectrum(self):
        pass



# class DN_Source_radial_BB(zdx.Base):
#     source_bb: DN_Source_BB
#     radial_coord: jp.ndarray
#     transform: jp.ndarray
#     radial_exp_density: jp.ndarray
#     def __init__(self, locs: DN_PointCollection, transform, radial_exp_density):
#         self.source_bb = DN_Source_BB(locs, blackbody=)
#         pass
#     @property
#     def irradiance(self):
#         irradiance = self.radial_exp_density * 
#         return irradiance

from typing import Dict

class DN_Source_Spectrum(zdx.Base):
    locs: DN_PointCollection
    irradiance: jp.ndarray
    def __init__(self, locs: DN_PointCollection,
                    irradiance: jp.ndarray):
        self.locs = locs
        self.irradiance = jp.asarray(irradiance, dtype=float)

"""
    * DN_Source is defines the typical sources. I has a PointCollection
as `.locs` and a `.irradiance` as a property.
    * DN_Observation typically has a List of DN_Sources for `.nuisance`
and a list of DN_Sources for `.interest`
"""
DN_Source = Union[DN_Source_Spectrum]

class SourceList(zdx.Base):
    """Basic class for modelling a set of normal distributions"""
    sources : dict
    name : str

    def __init__(self, name, **kwargs):
        self.name = str(name)
        self.sources = kwargs

    def __len__(self):
        return len(sources)

    def __getattr__(self, key):
        """Allows us to access the individual normals by their dictionary key"""
        if key in self.sources.keys():
            return self.sources[key]
        else:
            raise AttributeError(f"{key} not in {self.sources.keys()}")


class DN_ErrorPhasorPistonPointing(zdx.Base):
    piston : jp.ndarray
    pointing : jp.ndarray

    def __init__(self, piston, pointing):
        self.piston = jp.asarray(piston, dtype=float)
        self.pointing = jp.asarray(pointing, dtype=float)

    def phasor(self, lambs):
        amp = (1 - self.pointing**2 / 2)
        phase = self.piston * 2 * jp.pi / lambs
        return amp * jp.exp(1j * phase)
    

DN_ErrorPhasorType = Union[DN_ErrorPhasorPistonPointing, ]

class DN_Observation(zdx.Base):
    dn_nifits: DN_NIFITS
    nuisance: SourceList
    interest: SourceList
    error_phasor : DN_ErrorPhasorType
    def __init__(self, dn_nifits, dn_nuisance, dn_interest):
        self.dn_nifits = dn_nifits
        self.nuisance = dn_nuisance
        self.interest = dn_interest

    def geometric_phasor(self, sourcelist):
        """
        Returns the complex phasor corresponding to the locations
        of the family of sources
        
        **Parameters:**
        
        * ``alpha``         : (n_frames, n_points) The coordinate matched to X in the array geometry
        * ``beta``          : (n_frames, n_points) The coordinate matched to Y in the array geometry
        * ``anarray``       : The array geometry (n_input, 2)
        * ``include_mod``   : Include the modulation phasor
        
        **Returns** : A vector of complex phasors

        """
        allbs = []
        for asource in sourcelist:
            alpha, beta = asource.locs.coords_rad
            ds_mas2 = asource.locs.ds_mas2
            xy_array = jp.array(self.nifits.ni_mod.appxy)
            lambs = jp.array(self.nifits.oi_wavelength.lambs)
            k = 2*jp.pi/lambs
            a = jp.array((alphas, betas), dtype=jp.float64)
            phi = k[:,None,None,None] * jp.einsum("t a x, x m -> t a m", xy_array[:,:,:], a[:,:])
            b = jp.exp(1j*phi)
            allbs.append(b)
        bs = jp.concatenateate(allbs, axis=-1)
        return b.transpose((1,0,2,3))

    def get_modulation_phasor(self):
        """
        Shape: [n_frames n_wavelengths n_inputs]
        
        """
        raw_mods = jp.array(self.dn_nifits.ni_mod.all_phasors)
        col_area = jp.array(self.dn_nifits.ni_mod.arrcol)
        mods =  raw_mods*jp.sqrt(col_area)[:,None,:]
        return mods

    def get_fov_function(self, sourcelist):
        """
        Shape: [n_frames, n_wavelengths, n_locs]
        """
        wavelengths = self.dn_nifits.dn_wavelengths.lambs
        allgs = []
        for asource in sourcelist:
            alpha, beta = asource.locs.coords_rad
            allgs.append(self.dn_nifits.dn_fov.fov_function(alpha,beta))
        gs = jp.concatenate(allgs, axis=-1)
        return gs

    def get_Is(self, xs):
        """
        Get intensity from an array of sources.

        """
        E = jp.einsum("w o i , t w i m -> t w o m", self.dn_nifits.dn_catm.M, xs)
        I = jp.abs(E)**2
        return I

    def get_KIs(self,
                    Iarray:jp.ndarray):
        r"""
        Get the prost-processed observable from an array of output intensities. The
        post-processing matrix K is taken from ``self.nifits.ni_kmat.K``

        Args:
            I     : (n_frames, n_wl, n_outputs, n_batch)

        Returns:
            The vector :math:`\boldsymbol{\kappa} = \mathbf{K}\cdot\mathbf{I}`

        """
        KI = jp.einsum("k o, t w o m -> t w k m", self.nifits.ni_kmat.K[:,:], Iarray)
        return KI


    def get_all_outs(self,sourcelist,
                        kernels=False):
        """
        Compute the transmission map for an array of coordinates. The map can be seen
        as equivalent collecting power expressed in [m^2] for each point sampled so as
        to facilitate comparison with models in Jansky multiplied by the exposure time
        of each frame (available in `nifits.ni_mod.int_time`).

        Args:
            alphas  : ArrayLike [rad] 1D array of coordinates in right ascension
            betas   : ArrayLike [rad] 1D array of coordinates in declination
            kernels : (bool) if True, then computes the post-processed
                  observables as per the KMAT matrix.

        Returns:
            if ``kernels`` is False: the *raw transmission output*.
            if ``kernels`` is True: the *differential observable*.

        .. hint:: **Shape:** (n_frames, n_wl, n_outputs, n_points)

        """
        # The phasor from the incidence on the array:
        xs = self.geometric_phasor(sourcelist)
        # print("xs", xs)
        
        # The phasor from the spatial filtering:
        x_inj_nuisance = self.dn_nifits.dn_fov.get_fov_function(sourcelist)
        # print("x_inj", x_inj)
        
        # The phasor from the internal modulation
        # x_mod = self.nifits.ni_mod.all_phasors
        x_mod = self.get_modulation_phasor()
        # print("x_mod", x_mod)
        
        # this is actually a collecting area
        Is = self.get_Is(xs * x_inj[:,:,None,:] * x_mod[:,:,:,None])
        if kernels:
            KIs = self.get_KIs(Is)
            return KIs
        else:
            return Is


class DN_Error_Estimation(DN_Observation):
    error_phasors: DN_CPX
    def __init__(self, *args, **kwargs):
        super().__init__nit__(*args, **kwargs)
        n_collectors = self.dn_nifits.n_collectors
        self.error_phasors = DN_CPX.zeros(n_collectors)

    def aberrated_model(self):
        pass


"""
TODO:
* The blackbody object can work with arrays: use that
* The wavelength is passed as extra parameter from the trunk object


TODO: propagation of light
## Objectives:
### Covariance matrix estimation:
Needs
* Extended nuisance star
* No planet
* Additional input noises! TODO
* Single frame propagation? TODO

### Blackbody retrieval
Needs
* Extended nuisance star
* One planet
* No additional noises
* Whitening TODO

### Full spectrum fit
* Extended nuisance star
* One planet with spectrum !
* Signal whitening

### Multi-planet fit


DN_Observation
* dn_nifits
* nuisance
    - DN_Source
* interest

* DN_Source
    - locs
"""









