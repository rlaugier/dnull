"""
    dnull offers a specialty backend ton NIFITS.
It does not facilitate the exportation back to NIFITS
nor the creation of NIFITS files from scratch.
Import an nifits file through the DN_NIFITS constructor.

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

# Numpy is used to handle arays of strings.
import numpy as np

from astropy import units, constants as cst
from einops import rearrange

from scipy.stats import ncx2

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
    @property
    def shape(self):
        return self.real.shape

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
        return jp.gradient(cst.c.value/self.dlambs)

class DN_IOTAGS(zdx.Base):
    outbright: jp.ndarray
    outdark: jp.ndarray
    outphot: jp.ndarray
    inpola: np.ndarray
    outpola: np.ndarray

    def __init__(self, ni_iotags: io.oifits.NI_IOTAGS):
        self.outbright = jp.asarray(ni_iotags.outbright, dtype=bool)
        self.outdark = jp.asarray(ni_iotags.outdark, dtype=bool)
        self.outphot = jp.asarray(ni_iotags.outphot, dtype=bool)
        self.inpola = np.asarray(ni_iotags.inpola, dtype=str)
        self.outpola = np.asarray(ni_iotags.outpola, dtype=str)

    @property
    def outbright(self):
        """
        The flags of bright outputs
        """
        return self.outbright
    @property
    def outdark(self):
        """
        The flags of dark outputs
        """
        return self.outdark
    @property
    def outphot(self):
        """
        The flags of photometric outputs
        """
        return self.outphot
    @property
    def outpola(self):
        """
        The polarization of outputs.
        """
        return self.outpola
    @property
    def inpola(self):
        """
        The polarization of inputs.
        """
        return self.inpola

class DN_OSWAVELENGTH(zdx.Base):
    """
    An object storing the wavelength before a downsampling. This must have the
    wavelength for each of the slice of the CATM matrix, each of the ``NI_MOD``
    phasors and each column of the ``NI_DSAMP`` matrix.

    If ``DN_OSWAVELENGTH`` is absent, assume that there is no over or down-
    sampling and take the values directly from ``DN_WAVELENGTH``.

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
        return jp.abs(jp.gradient(cst.c.value/self.lambs))

DN_TARGET = io.oifits.OI_TARGET

class DN_IOUT(zdx.Base):
    iout: jp.ndarray
    unit: units.Unit

    def __init__(self, ni_iout: io.oifits.NI_IOUT,
                    iout=None,
                    unit=None):
        if iout is None:
            self.iout = jp.asarray(ni_iout.iout, dtype=float)
        else:
            self.iout = iout
        if units is None:
            self.unit = ni_iout.unit
        else:
            self.unit = unit
from astropy.io.fits import Header
class DN_CATM(zdx.Base):
    M: DN_CPX
    def __init__(self, ni_catm: io.oifits.NI_CATM):
        self.M = DN_CPX(ni_catm.M)

class DN_KIOUT(zdx.Base):
    kiout: jp.ndarray
    unit: units.Unit
    def __init__(self, ni_kiout: io.oifits.NI_KIOUT):
        self.kiout = jp.asarray(ni_kiout.kiout, dtype=float)
        self.unit = ni_kiout.unit
    @property
    def shape(self):
        return self.kiout.shape

class DN_WKIOUT(DN_KIOUT):
    Ws : jp.ndarray
    def __init__(self, dn_kiout, Ws):
        self.Ws = Ws
        self.kiout = dn_kiout.kiout
        self.unit = dn_kiout.unit
        
    def whitened_kiout(self):
        data = self.data_table["value"].data
        full_shape = data.shape
        flat_full_shape = (full_shape[0], full_shape[1]*full_shape[2])
        flat_out = rearrange(data, "frame wavelength output -> frame (wavelength output)")
        wout = jp.einsum("f o i , f i -> f o", self.Ws, flat_out)
        wout_full = wout.reshape((full_shape))
        return wout_full


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

class DN_DSAMP(zdx.Base):
    D: jp.ndarray
    def __init__(self, ni_dsamp: io.oifits.NI_DSAMP):
        self.D = jp.asarray(ni_dsamp.D, dtype=float)

class DN_ARRAY(zdx.Base):
    """
        Temporary, not use
    """
    D: jp.ndarray
    def __init__(self, ni_dsamp: io.oifits.OI_ARRAY):
        self.D = jp.asarray(ni_dsamp.D, dtype=float)


class DN_MOD(zdx.Base):
    time: jp.ndarray
    int_time: jp.ndarray
    mod_phas: DN_CPX
    app_xy: jp.ndarray
    arrcol: jp.ndarray
    fov_index: jp.ndarray

    def __init__(self, ni_mod: io.oifits.NI_MOD=None,
                        time=None,
                        int_time=None,
                        mod_phas=None,
                        app_xy=None,
                        arrcol=None,
                        fov_index=None,
                        clip_length=None):            
        if time is None:
            self.time = jp.asarray(ni_mod.data_table["TIME"].data,
                                            dtype=float)
        else:
            self.time = time
        if int_time is None:
            self.int_time = jp.asarray(ni_mod.int_time, dtype=float)
        else:
            self.int_time = int_time
        if mod_phas is None:
            self.mod_phas = DN_CPX(ni_mod.all_phasors)
        else:
            self.mod_phas = mod_phas
        if app_xy is None:
            self.app_xy = jp.asarray(ni_mod.appxy, dtype=float)
        else:
            self.app_xy = app_xy
        if arrcol is None:
            self.arrcol = jp.asarray(ni_mod.arrcol, dtype=float)
        else:
            self.arrcol = arrcol
        if fov_index is None:
            self.fov_index = jp.asarray(ni_mod.data_table["FOV_INDEX"].data,
                                            dtype=int)
        else:
            self.fov_index = fov_index

        

    # @classmethodssmethod
    # def from_same(cls, dn_mod: DN_MOD=None):
    #     cls(time=,
    #         int_time=,
    #         mod_phase=,
    #         app_xy=,
    #         )
        
    
    @property
    def appxy(self):
        return self.app_xy

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


    

class DN_FOV(object):
    def __init__(self, ni_fov: io.oifits.NI_FOV):
        """
            This class is a generic class that is identifies and creates the
        corredt DN_FOV class. It will return an object from the correct class instead.

        At this point, it can only work with the `diameter_gaussian_radial` mode.
        """
        if isinstance(ni_fov, io.oifits.NI_FOV):
            assert ni_fov.header["FOV_MODE"] == "diameter_gaussian_radial",\
                    NotImplementedError("Only diameter_gaussian_radial implemented")
        self = DN_FOV_diam_gau_rad(ni_fov)

    def correct_fov_class(self, ni_fov: io.oifits.NI_FOV):
        """
            This method identifies the correct type of DN_FOV object
        to create, initializes and returns it.

        At this point, it can only work with the `diameter_gaussian_radial` mode.
        """
        if isinstance(ni_fov, DN_FOV_diam_gau_rad):
            return DN_FOV_diam_gau_rad
        assert ni_fov.header["FOV_MODE"] == "diameter_gaussian_radial",\
                    NotImplementedError("Only diameter_gaussian_radial implemented")
        if ni_fov.header["FOV_MODE"] == "diameter_gaussian_radial":
            return DN_FOV_diam_gau_rad


class DN_FOV_diam_gau_rad(zdx.Base):
    D: jp.ndarray
    offset: jp.ndarray
    def __init__(self, ni_fov: io.oifits.NI_FOV):
        assert ni_fov.header["FOV_MODE"] == "diameter_gaussian_radial",\
                    NotImplementedError("Only diameter_gaussian_radial implemented")
        D = ni_fov.header["FOV_TELDIAM"]
        uD = units.Unit(ni_fov.header["FOV_TELDIAM_UNIT"])
        self.D = jp.asarray(D*uD.to(units.m), dtype=float)
        self.offset = jp.asarray(ni_fov.data_table["offsets"], dtype=float)

    def fov_function(self, x, y, lambs):
        """
        Returns the phasor corresponding to position in FOV

        Shape: [n_frames n_wavelengths n_points, ]
        """
        r_0 = (1/2*lambs/self.D)# *units.rad.to(units.mas)
        r = jp.hypot(x[None, None, :] - self.offset[:,:, 0, None],
                        y[None,None, :] - self.offset[:,:, 1, None])
        phasor = jp.exp(-(r[:,:]/r_0[:,None])**2)
        return phasor.astype(jp.complex64)



from typing import Union, List, Callable

DN_FOV_TYPE = Union[DN_FOV, DN_FOV_diam_gau_rad,]

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
    unit: units.Unit = None
    def __init__(self, aa, bb, ds_mas2, unit=units.mas,
                    shape=None, orig_shape=None):
        self.aa = aa
        self.bb = bb
        self.unit = unit
        self.shape = self.aa.shape
        self.ds_mas2 = ds_mas2
        if not hasattr(self, "orig_shape"):
                self.orig_shape = self.shape
        elif self.orig_shape is None:
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

    def coords_rad_single(self, index):
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

    def plot_frame(self, z=None, frame_index=0, wl_index=0,
                        out_index=0, mycmap=None, marksize_increase=1.0,
                        colorbar=True, xlabel=True, title=True,
                        whitened=False):
        """
            A convenience method to plot values of the point collection
        spatially.

        Args:
            z : (float) The value to plot for color
            frame_index : (int) The frame index to plot
            wl_index : (int)
            out_index :
            mycmap : 
            marksize_increase :
            colorbar : 
            xlabel : 
            title :
            whitened : if True, then out_index is used and wl_index is moot.
            
        """
        import matplotlib.pyplot as plt
        marksize = marksize_increase * 50000/self.shape[0]
        if whitened:
            zframe = z[frame_index, out_index, :]
        else:
            zframe = z[frame_index,wl_index,out_index,:]
        if len(self.orig_shape) == 1:
            plt.scatter(*self.coords, c=zframe,
                    cmap=mycmap, s=marksize)
            plt.gca().set_aspect("equal")
        else:
            plt.imshow(zframe.reshape((self.orig_shape)),
                cmap=mycmap, extent=self.extent)
            plt.gca().set_aspect("equal")
            
        if colorbar:
            plt.colorbar()
        if xlabel is True:
            plt.xlabel("Relative position [mas]")
        elif xlabel is not False:
            plt.xlabel(xlabel)
        if title is True:
            plt.title(f"Output {out_index} for frame {frame_index}")
        elif title is not False:
            plt.title(title)



def test_attr(obj, name):
    if hasattr(obj, name):
        return getattr(obj, name) is not None
    else:
        return False

names_dn = [
            "dn_catm",
            "dn_fov",
            "dn_kmat",
            "dn_wavelength",
            "dn_target",
            "dn_mod",
            "dn_iout",
            "dn_kiout",
            "dn_kcov",
            "dn_array"]
names_ni = [
            "ni_catm",
            "ni_fov",
            "ni_kmat",
            "oi_wavelength",
            "oi_target",
            "ni_mod",
            "ni_iout",
            "ni_kiout",
            "ni_kcov",
            "oi_array"]


class DN_NIFITS(zdx.Base):
    dn_catm: DN_CATM
    dn_fov: DN_FOV_TYPE
    dn_kmat: DN_KMAT
    dn_wavelength: DN_WAVELENGTH
    dn_target: DN_TARGET
    dn_mod: DN_MOD
    dn_iout: DN_IOUT
    dn_kiout: DN_KIOUT
    dn_kcov: DN_KCOV
    dn_dsamp: DN_DSAMP
    dn_iotags: DN_IOTAGS
    dn_oswavelength: DN_OSWAVELENGTH
    def __init__(self,
                dn_catm: DN_CATM = None,
                dn_fov: DN_FOV_TYPE = None,
                dn_kmat: DN_KMAT = None,
                dn_wavelength: DN_WAVELENGTH = None,
                dn_oswavelength: DN_OSWAVELENGTH = None,
                dn_target: DN_TARGET = None,
                dn_mod: DN_MOD = None,
                dn_iout: DN_IOUT = None,
                dn_kiout: DN_KIOUT = None,
                dn_kcov: DN_KCOV = None,
                dn_dsamp: DN_DSAMP = None,
                dn_iotags: DN_IOTAGS = None):
        
        self.dn_catm = dn_catm
        self.dn_fov = dn_fov
        self.dn_kmat = dn_kmat
        self.dn_wavelength = dn_wavelength
        self.dn_target = dn_target
        self.dn_mod = dn_mod
        self.dn_iout = dn_iout
        self.dn_kiout = dn_kiout
        self.dn_kcov = dn_kcov
        self.dn_dsamp = dn_dsamp
        self.dn_iotags = dn_iotags
        self.dn_oswavelength = dn_oswavelength
        # self.fov_function = self.dn_fov.fov_function

    @classmethod
    def from_nifits(cls, anifits):
        extensions = {}
        for niname, dnname in zip(names_ni, names_dn):
            if test_attr(anifits, niname):
                myclass = getclass(dnname.upper())
                myobj = myclass(getattr(anifits, niname))
                print(niname, dnname)
                # Exception for FOV: overwrite the object with
                # the specific class instead of the generic one
                if dnname == "dn_fov":
                    print("Triggering dn_fov")
                    myobj = myobj.correct_fov_class(getattr(anifits, niname))
                    myobj = myobj(getattr(anifits, niname))
                extensions[dnname] = myobj
                    
                    
        print(extensions)
        return cls(**extensions)

    @classmethod
    def update(cls, adnull,
                dn_catm: DN_CATM = None,
                dn_fov: DN_FOV_TYPE = None,
                dn_kmat: DN_KMAT = None,
                dn_wavelength: DN_WAVELENGTH = None,
                dn_oswavelength: DN_OSWAVELENGTH = None,
                dn_target: DN_TARGET = None,
                dn_mod: DN_MOD = None,
                dn_iout: DN_IOUT = None,
                dn_kiout: DN_KIOUT = None,
                dn_kcov: DN_KCOV = None,
                dn_dsamp: DN_DSAMP = None,
                dn_iotags: DN_IOTAGS = None,
                dn_array: DN_ARRAY = None):

        extensions = {}
        for niname, dnname in zip(names_ni, names_dn):
            theinput = locals()[dnname]
            if test_attr(adnull, dnname):
                if theinput is None:
                    print("Reusing ", dnname)
                    # Exception for FOV: overwrite the object with
                    # the specific class instead of the generic one
                    locals()[dnname]
                    if hasattr(adnull, dnname):
                        myobj = getattr(adnull, dnname)
                        extensions[dnname] = myobj
                else:
                    print("Updating ", dnname)
                    extensions[dnname] = theinput
            
        print(extensions)
        return cls(**extensions)
        


    def fov_function(self, x, y):
        """
        This method is a wrapper for the method of the fov
        object. 

        """
        lambs = self.dn_wavelength.lambs
        fov_object = self.dn_fov
        return self.dn_fov.fov_function(x, y, lambs)
        

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

ext_corresp = {
        "dn_catm": "ni_catm" ,
        "dn_fov": "ni_fov" ,
        "dn_kmat": "ni_kmat" ,
        "dn_wavelength": "oi_wavelength" ,
        "dn_oswavelength": "ni_oswavelength" ,
        "dn_target": "oi_target" ,
        "dn_mod": "ni_mod" ,
        "dn_iout": "ni_iout" ,
        "dn_kiout": "ni_kiout" ,
        "dn_kcov": "ni_kcov" ,
        "dn_dsamp": "ni_dsamp" ,
        "dn_iotags": "ni_iotags"
        }


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
        self.locs = locs
        self.blackbody = DN_BB

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

    @classmethod
    def uniform_disk(cls, locs, transformation,
                     cross_section: jp.ndarray = None,
                     T_eff: float = 5300,
                     distance: float = 10):
        mybb = DN_BB(T_eff, cross_section=cross_section,
                            col_dens=None)
        return cls(locs, mybb)

class DN_Source_Base(zdx.Base):
    locs: DN_PointCollection
    irradiance: jp.ndarray

    def __init__(self, locs, irradiance):
        self.locs = locs
        self.irradiance = irradiance

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
    """
    (unsused)
    Basic class for modelling a set of light
    sources.
    """
    sources : dict
    name : str

    def __init__(self, name, **kwargs):
        self.name = str(name)
        self.sources = kwargs

    def __len__(self):
        return len(self.sources)

    def __getattr__(self, key):
        """Allows us to access the individual normals by their dictionary key"""
        if key in self.sources.keys():
            return self.sources[key]
        else:
            raise AttributeError(f"{key} not in {self.sources.keys()}")


        

class DN_ErrorPhasorPistonPointing(zdx.Base):
    """
        A class to define error realisations in the form
    of a piston and pointing tuple.
    D, the collector diameter is given to inform the link between
    pointing error and the associated chromatic injection amplitude.
    """
    piston : jp.ndarray
    pointing : jp.ndarray
    fourDsquare : jp.float32

    def __init__(self, piston, pointing, D):
        self.piston = jp.asarray(piston, dtype=float)
        self.pointing = jp.asarray(pointing, dtype=float)
        self.fourDsquare = 4*D**2

    def phasor(self, lambs):
        """
        The effect of pointing can be modeled as such:
        ```python
            import sympy as sp
            r, r_0, lamb_s, D = sp.symbols("r,r_0, lambda, D", real=True, positive=True)
            e = sp.exp(-(r/r_0)**2)
            e_series = sp.series(e.subs([(r_0, 1/2*lamb_s/D)]), r, n=4)
            sp.print_latex(e_series)
            e_series
        ```
        $ A(\\lambda) = 1 - \\frac{4.0 D^{2} r^{2}}{\\lambda^{2}} + O\\left(r^{4}\\right) $
        """
        amp = (1 - self.fourDsquare / lambs[None,:,None]**2
                        * self.pointing[:,None,:]**2)
        phase = self.piston[:,None,:] * 2 * jp.pi / lambs[None,:,None]
        return amp[:,:,:] * jp.exp(1j * phase)

    def single_phasor(self, lambs, index):
        """
        """
        amp = (1 - self.fourDsquare / lambs[:,None]**2
                        * self.pointing[None,:]**2)
        phase = self.piston[None,:] * 2 * jp.pi / lambs[:,None]
        return amp[:,:] * jp.exp(1j * phase)

    @classmethod
    def no_error(cls, mydn):
        """
        Creates a neutral error vector

        Args:
            mydn : a DN_NIFITS object to use for the shape of the arrays.
            D    : The main collector diameter
        """
        nwl = mydn.dn_wavelength.lambs.shape
        n_frams = mydn.dn_mod.n_series
        n_inputs = mydn.dn_mod.all_phasors.shape[-1]
        piston = jp.zeros((n_frams, n_inputs))
        pointing = jp.zeros((n_frams, n_inputs))
        D = mydn.dn_fov.D
        myobj = cls(piston=piston, pointing=pointing, D=D)
        return myobj

class DN_NOTT_LDC(zdx.Base):
    """
        Holds the *controlable* parameters of the NOTT LDC
    `xx_eq_index` holds the equivalent index of the materials for which lengths are stated.
    * `co2` : air-displacing length (1-n)
    * `glass` : air-displacing length
    * `air` : ambient air length
    """
    piston : jp.array
    co2_ppm : jp.float32
    co2_length : jp.array
    glass_length : jp.array
    co2_eq_index : jp.array
    air_eq_index : jp.array
    glass_eq_index : jp.array

    def __init__(self, piston, co2_ppm,
                    co2_length, glass_length,
                    temp, pres, rhum, co2, filet_path, lambs):
        self.piston = piston
        self.co2_ppm = co2_ppm
        self.co2_length = co2_length
        self.glass_length = glass_length
        from scifysim.n_air import wet_atmo
        from scifysim.correctors import corrector
        lab_air = wet_atmo(temp=temp, pres=pres, rhum=rhum, co2=co2)
        lab_co2 = wet_atmo(temp=temp, pres=pres, rhum=rhum, co2=co2)
        lab_air = corrector(temp=temp, pres=pres, rhum=rhum, co2=co2)
        self.co2_eq_index = lab_co2.get_Nair(lambs, add=0.)
        self.air_eq_index = lab_air.get_Nair(lambs, add=1.)
        self.glass_eq_index = load_txt_index(lambs,add=0)

    def phase(self, lambs):
        phase = 2*jp.pi/lambs[:,None] * (self.piston[None,:] * self.air_eq_index[:,None] \
                                    + self.co2_length[None,:] * self.co2_eq_index[:,None] \
                                    + self.gla_length[None,:] * self.glass_eq_index[:,None])
        return phase

    def amplitude(self, lambs):
        return jp.ones_like(lambs)

    def phasor(self, lambs):
        return self.amplitude(lambs) * jp.exp(1j*self.phase(lambs))
        
def load_txt_index(lambs, file_path, order=3):
    import scipy.interpolate as interp
    nplate_file = np.loadtxt(file_path, delimiter=";")
    nplate_int = interp.interp1d(nplate_file[:,0]*1e-6, nplate_file[:,1],
                                 kind=order, bounds_error=False )
    nplate = nplate_int(lambs)
    return nplate
    
    
class DN_NOTT_LDC_series(zdx.Base):
    locseries : list[DN_NOTT_LDC]

    def __init__(self, locseries, offset):
        self.locseries = locseries
        self.offset = offset

    @property
    def offset(self):
        return self.locseries[0]
    
    def phases(self, lambs):
        phases = np.array([astep.phase(lambs) for astep in self.locseries])
        return phases

    def amplitude(self, lambs):
        amplitudes = np.array([astep.amplitude(lambs) for astep in self.locseries])
        return amplitudes

    def phasor(self, lambs):
        amps = self.amplitude(lambs)
        phases = self.phases(lambs)
        return amps*jp.exp(1j*phases)


class DN_ErrorBankDouble(zdx.Base):
    """
        Use this for differentiable samples of realizations of
    instrumental errors.

    Applications:
        * Numerical/empirical evaluation (differentiable) of the covariance
          within one frame using single_variation_phasor, which produces
          many scaled realizations of the error phasor *instead* of a time
          series.

    For second order approximation of the covariance, use the Jac. and Hes.
    over a single ``DN_ErrorPhasorPistonPointing``
    """
    pistons : jp.ndarray
    pointings : jp.ndarray
    pistonscale : jp.ndarray
    pointingscale : jp.ndarray
    fourDsquare : jp.float32
    
    def __init__(self, pistons, pointings, pistonscale, pointingscale, fourDsquare):
        self.pistons = jp.asarray(pistons)
        self.pointings = jp.asarray(pointings)
        self.pistonscale = jp.asarray(pistonscale)
        self.pointingscale = jp.asarray(pointingscale)
        self.fourDsquare = jp.asarray(fourDsquare)
        

    @classmethod
    def gaussian_from_seed(cls, D=None,
                        shape=None, seed=None, rng=None,
                        verbose=True, pistonscale=1., pointingscale=1.):
        if rng is None:
            print("No rng, using seed.")
            if seed is None:
                print("No seed, using default value")
                seed = 10
            else:
                rng = np.random.default_rng(np.random.SeedSequence(seed))
        else :
            print("Using inherited rng")
        if not (isinstance(pistonscale, jp.ndarray) or isinstance(pistonscale, np.ndarray)):
            pistonscale = pistonscale * jp.ones(shape[1])
        if not (isinstance(pointingscale, jp.ndarray) or isinstance(pointingscale, np.ndarray)):
            pointingscale = pointingscale * jp.ones(shape[1])
        vals_piston = np.random.normal(size=shape)
        vals_pointings = np.random.normal(size=shape)

        obj = cls(vals_piston, vals_pointings, pistonscale, pointingscale, 4*D**2)
        return obj


    @property
    def scaled_pistons(self):
        return (self.pistons[:,:] * self.pistonscale[None,:])
    @property
    def scaled_pointings(self):
        return (self.pointings[:,:] * self.pointingscale[None,:])
    
    def variation_phasor(self, lambs, index):
        pistons = self.scaled_pistons
        pointings = self.scaled_pointings
        amp = (1 - self.fourDsquare / lambs[None,:,None]**2
                        * self.pointings[:,None,:]**2)
        phase = self.pistons[:,None,:] * 2 * jp.pi / lambs[None,:,None]
        phasor_variants = amp[:,:,:] * jp.exp(1j * phase)
        return phasor_variants
    
    def single_variation_phasor(self, lambs, index):
        pistons = self.scaled_pistons
        pointings = self.scaled_pointings
        amp = (1 - self.fourDsquare / lambs[:,None]**2
                        * pointings[index,None,:]**2)
        phase = pistons[index,None,:] * 2 * jp.pi / lambs[:,None]
        phasor_variants = amp * jp.exp(1j * phase)
        return phasor_variants[None,:,:]


DN_ErrorPhasorType = Union[DN_ErrorPhasorPistonPointing, ]

class DN_Observation(zdx.Base):
    dn_nifits: DN_NIFITS
    nuisance: SourceList
    interest: SourceList
    error_phasor : DN_ErrorPhasorType
    def __init__(self, dn_nifits, dn_nuisance, dn_interest,
                        error_phasor):
        self.dn_nifits = dn_nifits
        self.nuisance = dn_nuisance
        self.interest = dn_interest
        # TODO This is named a phasor but it is actually a piston
        self.error_phasor = error_phasor


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
            alphas, betas = asource.locs.coords_rad
            ds_mas2 = asource.locs.ds_mas2
            xy_array = jp.array(self.dn_nifits.dn_mod.appxy)
            lambs = jp.array(self.dn_nifits.dn_wavelength.lambs)
            k = 2*jp.pi/lambs
            a = jp.array((alphas, betas), dtype=jp.float32)
            phi = k[:,None,None,None] * jp.einsum("t a x, x m -> t a m", xy_array[:,:,:], a[:,:])
            b = jp.exp(1j*phi)
            allbs.append(b)
        bs = jp.concatenate(allbs, axis=-1)
        return bs.transpose((1,0,2,3))

    def single_geometric_phasor(self, sourcelist, index):
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
            alphas, betas = asource.locs.coords_rad_single(index)
            ds_mas2 = asource.locs.ds_mas2
            xy_array = jp.array(self.dn_nifits.dn_mod.appxy[index])
            lambs = jp.array(self.dn_nifits.dn_wavelength.lambs)
            k = 2*jp.pi/lambs
            a = jp.array((alphas, betas), dtype=jp.float32)
            phi = k[:,None,None,None] * jp.einsum("a x, x m -> a m", xy_array[:,:], a[:,:])
            b = jp.exp(1j*phi)
            allbs.append(b)
        bs = jp.concatenate(allbs, axis=-1)
        return bs.transpose((1,0,2,3))


    def get_modulation_phasor(self):
        """
        Shape: [n_frames n_wavelengths n_inputs]
        
        """
        raw_mods = jp.array(self.dn_nifits.dn_mod.all_phasors)
        col_area = jp.array(self.dn_nifits.dn_mod.arrcol)
        err_phas = self.error_phasor.phasor(self.dn_nifits.dn_wavelength.lambs)
        mods = err_phas * raw_mods * jp.sqrt(col_area)[:,None,:]
        return mods

    def get_single_clean_modulation_phasor(self, index):
        """
        Shape: [1, n_wavelengths n_inputs]
        
        """
        raw_mods = jp.array(self.dn_nifits.dn_mod.all_phasors[index])
        col_area = jp.array(self.dn_nifits.dn_mod.arrcol[index])
        mods = raw_mods * jp.sqrt(col_area)[None,:]
        return mods[None,:,:]

    def fov_function_wrapper(self, sourcelist):
        """
        Shape: [n_frames, n_wavelengths, n_locs]
        """
        wavelengths = self.dn_nifits.dn_wavelength.lambs
        allgs = []
        for asource in sourcelist:
            alpha, beta = asource.locs.coords_rad
            allgs.append(self.dn_nifits.dn_fov.fov_function(alpha,beta, wavelengths))
        gs = jp.concatenate(allgs, axis=-1)
        return gs

    def single_fov_function_wrapper(self, sourcelist, index):
        """
        Shape: [1, n_wavelengths, n_locs]
        """
        wavelengths = self.dn_nifits.dn_wavelength.lambs
        allgs = []
        for asource in sourcelist:
            alpha, beta = asource.locs.coords_rad_single(index)
            allgs.append(self.dn_nifits.dn_fov.fov_function(alpha,beta, wavelengths))
        gs = jp.concatenate(allgs, axis=-1)
        # TODO: inefficient: ideally compute only the needed one.
        return gs[index,:,:][None, :,:]

    def get_Is(self, xs):
        """
        Get intensity from an array of sources.

        """
        E = jp.einsum("w o i , t w i m -> t w o m", self.dn_nifits.dn_catm.M.cpx, xs)
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
        KI = jp.einsum("k o, t w o m -> t w k m", self.dn_nifits.dn_kmat.K[:,:], Iarray)
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
        x_inj = self.fov_function_wrapper(sourcelist)
        # print("x_inj", x_inj)
        
        # The phasor from the internal modulation
        # x_mod = self.nifits.ni_mod.all_phasors
        x_mod = self.get_modulation_phasor()
        # print("x_mod", x_mod)
        
        # this is actually a collecting area
        Es = xs * x_inj[:,:,None,:] * x_mod[:,:,:,None]
        Is = self.get_Is(Es)
        if kernels:
            KIs = self.get_KIs(Is)
            return self.downsample(KIs)
        else:
            return self.downsample(Is)
        
    def get_total_outs(self, sourcelist, kernels=False):
        """
        Same as `get_all_outs` but sums the contribution of all sources.
        """
        z = self.get_all_outs(sourcelist, kernels=kernels)
        return jp.sum(z, axis=-1)

    def get_total_single_variations(self, sourcelist,
                                index, 
                                errorbank: DN_ErrorBankDouble,
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

        .. hint:: **Shape:** (n_reals, n_wl, n_outputs, n_points)

        """
        # The phasor from the incidence on the array:
        xi = self.single_geometric_phasor(sourcelist, index)
        # print("xs", xs)
        
        # The phasor from the spatial filtering:
        x_inj = self.single_fov_function_wrapper(sourcelist, index)

        # print("x_inj", x_inj)
        
        # The phasor from the internal modulation
        # x_mod = self.nifits.ni_mod.all_phasors
        x_mod = self.get_single_clean_modulation_phasor(index)
        # print("x_mod", x_mod)

        err_phas = errorbank.single_variation_phasor(self.dn_nifits.dn_wavelength.lambs, index)
        
        # this is actually a collecting area
        print("xi", xi.shape,"x_inj",(x_inj.shape),"x_mod",x_mod.shape,"err_phas",err_phas.shape)
        Is = self.get_Is(xi * x_inj[:,:,None,:] * x_mod[:,:,:,None] * err_phas[:,:,:,None])
        if kernels:
            KIs = self.get_KIs(Is)
            return jp.sum(self.downsample(KIs), axis=-1)
        else:
            return jp.sum(self.downsample(Is), axis=-1)


    def downsample(self, Is):
        """
        Downsample flux from the NI_OSWAVELENGTH bins to
        OI_WAVELENGTH bins.
        Expected shape is : (n_frames, n_wl, n_outputs, n_points), the method
        simply applies the ``NI_DSAMP`` matrix along the second axis (1).

        Args:
            Is     : ArrayLike [flux] input the result computed with the
                     oversampled wavelength channels.
                     (n_frames, n_wlold, n_outputs, n_points)

        Returns:
            Ids    : ArrayLike [flux] (n_frames, n_wlnew, n_outputs, n_points)

        Returns
        """
        if (self.dn_nifits.dn_dsamp is None) or (self.dn_nifits.dn_oswavelength is None):
            return Is
        Ids = jp.einsum("l w, t w o m -> t l o m", self.dn_nifits.dn_dsamp.D, Is )
        return Ids



class DN_Error_Estimation(DN_Observation):
    """
        Use this to evaluate the uncertainty on the measurements
    """
    error_phasors: DN_CPX
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_collectors = self.dn_nifits.n_collectors
        self.error_phasors = DN_CPX.zeros(n_collectors)

    def aberrated_model(self):
        pass

import types
from types import ModuleType
from jax.scipy.linalg import sqrtm


class DN_Post(DN_Observation):
    datashape: tuple
    flatshape: tuple
    fullyflatshape: tuple
    Ws: jp.ndarray
    W_unit : units.Unit
    
    
    """
    This variant of the backend class offers a statistically whitened alternate
    forward model with directly whitened observables by calling ``w_`` prefixed
    methods. (Ceau et al. 2019, Laugier et al. 2023)

    After normal construction, use ``create_whitening_matrix()`` to update the
    whitening matrix based on the ``NI_KCOV`` data.
    
    Use ``w_get_all_outs`` and ``get_moving_outs`` in the same way, but they
    return whitened observables. Compare it to ``self.dn_nifits.ni_kiout.w_kiout``
    instead of ``self.dn_nifits.ni_kiout.kiout``.

    """

    def __init__(self, dn_nifits, dn_nuisance, dn_interest,
                        error_phasor):
        self.dn_nifits = dn_nifits
        self.nuisance = dn_nuisance
        self.interest = dn_interest
        # TODO This is named a phasor but it is actually a piston
        self.error_phasor = error_phasor
        self.create_whitening_matrix()
        

    def create_whitening_matrix(self,
                            ):
        """
            Updates the whitening matrix:

        Args:
            md :  A numpy-like backend module.

        The pile of whitening matrices is stored as ``self.dn_nifits.dn_kiout.Ws`` (one for
        each frame).
        """
        # Assertion: assert
        # assert hasattr(self.dn_nifits, "dn_kcov")
        # if self.dn_nifits.dn_kcov.header["NIFITS SHAPE"] != "frame (wavelength output)":
        #     raise NotImplementedError("Covariance shape expected: frame (wavelength output)")

        Ws = []
        for amat in self.dn_nifits.dn_kcov.kcov:
            Ws.append(np.linalg.inv(sqrtm(amat)))
        self.Ws = jp.array(Ws)
        self.W_unit = self.dn_nifits.dn_kcov.unit**(-1/2)
        if hasattr(self.dn_nifits, "dn_kiout"):
            self.datashape = (self.dn_nifits.dn_kiout.shape) 
        self.flatshape = (self.dn_nifits.dn_mod.mod_phas.shape[0], self.dn_nifits.dn_kcov.kcov.shape[1])
        self.fullyflatshape = np.prod(self.flatshape)

    def whiten_signal(self, signal):
        """
        Whitens a signal so that error covariance
        is identity in the new axis.

        Args:
            signal: The direct signal to whiten (differential observable
                a.k.a kernel-null)

        Returns:
            wout_full: the whitened signal ($\\mathbf{W}\\cdot \\mathbf{s}$)
                in the new basis.

        """
        full_shape = signal.shape
        # Flatten the spectral and output dimension
        flat_full_shape = (*self.flatshape, full_shape[-1])
        flat_out = rearrange(signal, "frame wavelength output source -> frame (wavelength output) source")
        wout = jp.einsum("f o i , f i m -> f o m", self.Ws, flat_out)
        return wout


    def w_get_all_outs(self, *args, **kwargs):
        """
        """
        output = self.get_all_outs(*args, **kwargs)
        wout_full = self.whiten_signal(output)
        return wout_full

    def w_get_moving_outs(self, *args, **kwargs):
        """
        """
        output = self.get_moving_outs(*args, **kwargs)
        wout_full = self.whiten_signal(output)
        return wout_full

    def add_blackbody(self, temperature):
        """
            Initializes the blackbody for a given temperature
        Args:
            temperature: units.Quantity
        """
        self.bb = BB(temperature)

    def get_pfa_Te(self, signal=None,
                        md=np):
        """
            Compute the Pfa for the energy detector test.
        Args:
            signal: The raw signal (non-whitened)

        Returns:
            pfa the false alarm probability (p-value) associated
                to the given signal.
        """
        w_signal = self.whiten_signal(signal)
        wf_signal = md.flatten(w_signal)
        pfa = 1 - ncx2.cdf(wf_signal, df=self.fullyflatshape, nc=0.)
        return pfa

    def get_pfa_Tnp(self, alphas, betas,
                    signal=None,
                    model_signal=None,
                        md=np):
        """
            Compute the Pfa for the Neyman-Pearson test.
        """
        pass


    def get_blackbody_native(self, ):
        """
        Returns:
            blackbody_spectrum: in units consistent with the native
                units of the file $[a.sr^{-1}.m^{-2}]$ (where [a] is typically [ph/s]).
        """
        # Typically given there in erg/Hz/s/sr/cm^2
        myspectral_density = self.bb(self.dn_nifits.oi_wavelength.lambs * units.m)
        # Photon_energies in J/ph
        photon_energies = (self.dn_nifits.oi_wavelength.lambs*units.m).to(
                            units.J, equivalencies=units.equivalencies.spectral())\
                                / units.photon
        dnus = self.dn_nifits.oi_wavelength.dnus * (units.Hz)
        print((myspectral_density * dnus / photon_energies).unit)
        blackbody_spectrum = (myspectral_density * dnus / photon_energies).to(
                                 self.dn_nifits.ni_iout.unit / units.sr / (units.m**2))
        return blackbody_spectrum


    def get_blackbody_collected(self, alphas, betas,
                        kernels=True, whiten=True,
                            to_si=True):
        """
            Obtain the output spectra of a blackbody at the given blackbody temperature
        Args:
            alphas: ArrayLike: Relative position in rad
            betas:  ArrayLike: Relative position in rad
            kernels: Bool (True) Whether to work in the kernel postprocessing space
                (False is not implemented yet)
            whiten: Bool (True) whether to use whitening post-processing (False
                is not implemented yet)
            to_si: Bool (True) convert to SI units
        """
        collecting_map_q = units.m**2 * self.get_all_outs(alphas, betas, kernels=kernels)
        blackbody_spectrum = self.get_blackbody_native()
        collected_flux = blackbody_spectrum[None,:,None,None] \
                                * collecting_map_q[:,:,:,:]
        if whiten:
            blackbody_signal = self.W_unit * self.whiten_signal(collected_flux)
        else:
            blackbody_signal = collected_flux
        # collected_flux is in equivalent W / rad^2 
        # That is a power per solid angle of source
        
        if to_si:
            return blackbody_signal.to(blackbody_signal.unit.to_system(units.si)[0])
        else:
            return blackbody_signal

    def get_Te(self):
        """
            Computes the Te test statistic of the current file. This test statistic
        is supposed to be distributed as a chi^2 under H_0.
        Returns:
            Te : x.T.dot(x) where x is the whitened signal.
        """
        if hasattr(self.dn_nifits, "ni_kiout"):
            kappa = self.whiten_signal(self.dn_nifits.ni_kiout.kiout)
        else:
            raise NotImplementedError("Needs a NI_KIOUT extension")
        x = kappa.flatten()
        return x.T.dot(x)

    def get_pdet_te(self, alphas, betas,
                    solid_angle,
                    kernels=True, pfa=0.046,
                    whiten=True,
                    temperature=None):
        """
        pfa:
        * 1 sigma: 0.32
        * 2 sigma: 0.046
        * 3 sigma: 0.0027
        """
        if temperature is not None:
            self.add_blackbody(temperature)
        ref_spectrum = solid_angle * self.get_blackbody_collected(alphas, betas,
                                                    kernels=kernels,
                                                    whiten=True,
                                                    to_si=True)
        print(ref_spectrum.unit)
        threshold = ncx2.ppf(1-pfa, df=self.fullyflatshape, nc=0.)
        x = ref_spectrum.reshape(-1, 1000)
        xTx = np.einsum("o m , o m -> m", x, x)
        pdet_Pfa = 1 - ncx2.cdf(threshold, self.fullyflatshape, xTx)
        return pdet_Pfa

    def get_sensitivity_te(self, alphas, betas,
                    kernels=True,
                    temperature=None, pfa=0.046, pdet=0.90,
                    distance=None, radius_unit=units.Rjup,
                    md=np):
        """
        .. code-block:: python

            from scipy.stats import ncx2
            xs = np.linspace(-10, 10, 100)
            ys = np.linspace(1e-6, 0.999, 100)
            u = 1 - ncx2.cdf(xs, df=10, nc=0)
            v = ncx2.ppf(1 - ys, df=10, nc=0)

            plt.figure()
            plt.plot(xs, u)
            plt.show()

            plt.figure()
            plt.plot(ys, v)
            plt.plot(u, xs)
            plt.show()
        """
        from scipy.optimize import leastsq
        if temperature is not None:
            self.add_blackbody(temperature)
        ref_spectrum = self.get_blackbody_collected(alphas=alphas,betas=betas,
                                                    kernels=kernels, whiten=True,
                                                    to_si=True)
        print("Ref spectrum unit: ", ref_spectrum.unit)
        threshold = ncx2.ppf(1-pfa, df=self.fullyflatshape, nc=0.)
        x = (ref_spectrum).reshape((-1, ref_spectrum.shape[-1]))
        print("Ref signal (x) unit: ", x.unit)
        print("Ref signal (x) shape: ", x.shape)
        xtx = md.einsum("m i, i m -> m", x.T, x)
        lambda0 = 1.0e-3 * self.fullyflatshape
        # The solution lambda is the x^T.x value satisfying Pdet and Pfa
        sol = leastsq(residual_pdet_Te, lambda0, 
                        args=(threshold, self.fullyflatshape, pdet))# AKA lambda
        lamb = sol[0][0]
        # Concatenate the wavelengths
        lim_solid_angle = np.sqrt(lamb) / np.sqrt(xtx)
        if distance is None:
            return lim_solid_angle
        elif isinstance(distance, units.Quantity):
            dist_converted = distance.to(radius_unit)
            lim_radius = dist_converted*md.sqrt(lim_solid_angle/md.pi)
            return lim_radius.to(radius_unit, equivalencies=units.equivalencies.dimensionless_angles())

    def get_Te(self):
        """
            Computes the Te test statistic of the current file. This test statistic
        is supposed to be distributed as a chi^2 under H_0.
        Returns:
            Te : x.T.dot(x) where x is the whitened signal.
        """
        if hasattr(self.dn_nifits, "ni_kiout"):
            kappa = self.whiten_signal(self.dn_nifits.ni_kiout.kiout)
        else:
            raise NotImplementedError("Needs a NI_KIOUT extension")
        x = kappa.flatten()
        return x.T.dot(x)

    def get_pdet_te(self, alphas, betas,
                    solid_angle,
                    kernels=True, pfa=0.046,
                    whiten=True,
                    temperature=None):
        """
        pfa:
        * 1 sigma: 0.32
        * 2 sigma: 0.046
        * 3 sigma: 0.0027
        """
        if temperature is not None:
            self.add_blackbody(temperature)
        ref_spectrum = solid_angle * self.get_blackbody_collected(alphas, betas,
                                                    kernels=kernels,
                                                    whiten=True,
                                                    to_si=True)
        print(ref_spectrum.unit)
        threshold = ncx2.ppf(1-pfa, df=self.fullyflatshape, nc=0.)
        x = ref_spectrum.reshape(-1, 1000)
        xTx = np.einsum("o m , o m -> m", x, x)
        pdet_Pfa = 1 - ncx2.cdf(threshold, self.fullyflatshape, xTx)
        return pdet_Pfa

    def get_sensitivity_te(self, alphas, betas,
                    kernels=True,
                    temperature=None, pfa=0.046, pdet=0.90,
                    distance=None, radius_unit=units.Rjup,
                    md=np):
        """
        .. code-block:: python

            from scipy.stats import ncx2
            xs = np.linspace(-10, 10, 100)
            ys = np.linspace(1e-6, 0.999, 100)
            u = 1 - ncx2.cdf(xs, df=10, nc=0)
            v = ncx2.ppf(1 - ys, df=10, nc=0)

            plt.figure()
            plt.plot(xs, u)
            plt.show()

            plt.figure()
            plt.plot(ys, v)
            plt.plot(u, xs)
            plt.show()
        """
        from scipy.optimize import leastsq
        if temperature is not None:
            self.add_blackbody(temperature)
        ref_spectrum = self.get_blackbody_collected(alphas=alphas,betas=betas,
                                                    kernels=kernels, whiten=True,
                                                    to_si=True)
        print("Ref spectrum unit: ", ref_spectrum.unit)
        threshold = ncx2.ppf(1-pfa, df=self.fullyflatshape, nc=0.)
        x = (ref_spectrum).reshape((-1, ref_spectrum.shape[-1]))
        print("Ref signal (x) unit: ", x.unit)
        print("Ref signal (x) shape: ", x.shape)
        xtx = md.einsum("m i, i m -> m", x.T, x)
        lambda0 = 1.0e-3 * self.fullyflatshape
        # The solution lambda is the x^T.x value satisfying Pdet and Pfa
        sol = leastsq(residual_pdet_Te, lambda0, 
                        args=(threshold, self.fullyflatshape, pdet))# AKA lambda
        lamb = sol[0][0]
        # Concatenate the wavelengths
        lim_solid_angle = np.sqrt(lamb) / np.sqrt(xtx)
        if distance is None:
            return lim_solid_angle
        elif isinstance(distance, units.Quantity):
            dist_converted = distance.to(radius_unit)
            lim_radius = dist_converted*md.sqrt(lim_solid_angle/md.pi)
            return lim_radius.to(radius_unit, equivalencies=units.equivalencies.dimensionless_angles())


    def evalutate_single_covariance(self, sourcelist, index, kernels=True):
        errorbank = self.error_phasor
        if not isinstance(self.error_phasor, DN_ErrorBankDouble):
            raise NotImplementedError
        outs = self.get_total_single_variations(sourcelist, index,
                                            errorbank=self.error_phasor,
                                            kernels=kernels)

from kernuller import VLTI

nott_arg_dict = {
    "teldiam":8.2,
    "statlocs_":VLTI,
}

class calib_setup(DN_Observation):
    var_vec: jp.ndarray
    background: jp.ndarray
    def __init__(self, dn_nifits, dn_nuisance, dn_interest,
                        error_phasor, var_vec, background):
        self.dn_nifits = dn_nifits
        self.nuisance = dn_nuisance
        self.interest = dn_interest
        # TODO This is named a phasor but it is actually a piston
        self.error_phasor = error_phasor
        self.var_vec=var_vec
        self.background = background


    @classmethod
    def from_probe(cls, series, arg_dict, dn_nifits):
        # Create NIFITS
        ## Create the oi_wavelength
        ## Create the NI_MOD
        ## Create the NI_CATM
        ## Create basic NI_FOV
        ##
        return

    def residual_outs_nuisance(self):
        """
        Residual of observation with all outputs.
        """
        outs_model = self.get_total_outs([self.nuisance], kernels=False)
        res = jp.sum((outs_model + self.background[None,:,:] - self.dn_nifits.dn_iout.iout)**2/ self.var_vec)
        return res

    def residual_outs_nuisance_outputs(self, output):
        """
        Residual of observation with all outputs.
        """
        outs_model = self.get_total_outs([self.nuisance], kernels=False)
        res = jp.sum((outs_model + self.background[None,:,:] - output)**2/ self.var_vec)
        return res
        
        

    @classmethod
    def create_nifits(cls, ):
        pass

def full_hadamard_probe(ntel, amp, steps=5):
    mod_shutters = shutter_probe(ntel)
    base_probe = hadamard_modulation(ntel, amp)
    grad_probe = graduify(base_probe, steps)
    

def hadamard_modulation(ntel, amp):
    """
    Returns a hadamard matrix of given 
    """
    import scipy as scp
    mat = scp.linalg.hadamard(4)
    return mat*amp

def shutter_probe(ntel):
    mod_shutters = jp.eye(ntel+1, ntel)
    mod_shutters = jp.roll(mod_shutters, 1, axis=0)
    return mod_shutters


def graduify(matrix, steps):
    newrows = []
    for arow in matrix:
        for i in range(steps):
            newrow = i/steps * arow
            newrows.append(newrow)
    return np.array(newrows)

def randomized_probe(n, ntel=4, scale=1.0e-6, func=np.random.normal):
    mat = func(size=(n,ntel), scale=scale)
    return mat
    
    

"""
TODO:
* The blackbody object can work with arrays: use that
* The wavelength is passed as extra parameter from the trunk object

TODO: Null optimization
* Calibration of the LDC -> projection onto instrument modes (Vacuum, Air, CO2, Glass)
* On-sky tracking -> 1 DoF
* On-sky transverse dispersion tracker
* On-sky recalibration
* Instrument matrix evaluation
    - With uncalibrated LDC
    - With calibrated LDC



TODO: propagation of light
## Objectives:
### Covariance matrix estimation using second order:
Needs
* Extended nuisance star DONE
* No planet
* Additional input noises! WIP
* Single frame propagation? TODO

### Continuous MC covariance estimation
Use: ``DN_ErrorBankDouble``
* Create a new class to host the calculations
* 


### Blackbody retrieval
Needs
* Extended nuisance star DONE
* One planet DONE
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


