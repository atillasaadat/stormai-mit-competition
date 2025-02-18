# | default_exp atm
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from pymsis import msis
#import dill

# Orekit imports
import orekit
from org.orekit.time import AbsoluteDate
from org.orekit.utils import PVCoordinates, Constants
from org.orekit.frames import Frame
from org.orekit.models.earth.atmosphere import PythonAtmosphere
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.frames import FramesFactory, Frame
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.utils import IERSConventions
from org.orekit.time import TimeScalesFactory

import orekit
from orekit.pyhelpers import setup_orekit_curdir, absolutedate_to_datetime

from org.orekit.orbits import KeplerianOrbit, EquinoctialOrbit, PositionAngleType, OrbitType, CartesianOrbit
from org.orekit.frames import FramesFactory, LOFType, Frame
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants, PVCoordinates
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.propagation import SpacecraftState
from org.orekit.bodies import OneAxisEllipsoid, CelestialBodyFactory, CelestialBody
from org.orekit.utils import IERSConventions
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction, OceanTides, SolidTides
from orekit import JArray_double, JArray
from java.util import ArrayList

from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient, RadiationSensitive
from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData, JB2008SpaceEnvironmentData
from org.orekit.forces.drag import IsotropicDrag, DragForce
from org.orekit.models.earth.atmosphere import DTM2000, HarrisPriester, JB2008, NRLMSISE00, Atmosphere, SimpleExponentialAtmosphere, PythonAtmosphere
from orekit.pyhelpers import datetime_to_absolutedate

from org.hipparchus.geometry.euclidean.threed import Vector3D

from math import radians, degrees, pi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import time
import matplotlib.ticker as ticker

from IPython import embed


# Initialize Orekit JVM
vm = orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir()

def prop_orbit(initial_orbit, duration, step, CustomAtmosphereClass):
    """
    Propagates the orbit of a satellite over a given duration using a high-fidelity numerical propagator.
    Parameters:
    initial_orbit (Orbit): The initial orbit of the satellite.
    duration (float): The duration for which to propagate the orbit, in seconds.
    CustomAtmosphereClass (class): A custom atmosphere model class to be used for drag force calculations.
    Returns:
    tuple: A tuple containing:
        - states (list of SpacecraftState): The list of spacecraft states at each propagation step.
        - densities (list of float): The list of atmospheric densities at each spacecraft state.
    The function performs the following steps:
    1. Initializes the orbit parameters and spacecraft properties.
    2. Sets up the time span for propagation.
    3. Configures the numerical integrator and propagator.
    4. Adds force models to the propagator.
    5. Propagates the orbit over the specified duration.
    7. Plots the satellite trajectory in 3D.
    8. Computes atmospheric densities at each propagated state.
    """

    dur = duration # propagation duration time [s]
    satellite_mass = 260.0
    crossSection = 3.2 * 1.6 # m^2
    srpArea = 30.0 # m^2

    degree = 70
    torder = 70
    dragCoeff = 2.2
    cr = 1.0

    initialDate = initial_orbit.getDate()
    #tspan = [initialDate.shiftedBy(float(dt)) for dt in np.linspace(0, duration, int(duration / step))]
    # Assuming omni2_data is your DataFrame
    # Assuming omni2_data is your DataFrame
    timestamp_series = pd.to_datetime(omni2_data["Timestamp"])  # Convert column to pandas datetime

    # Convert each timestamp to an Orekit AbsoluteDate
    tspan = [datetime_to_absolutedate(ts.to_pydatetime()) for ts in timestamp_series]

    minStep = 1e-6
    maxstep = 100.0
    initStep = 1.0
    positionTolerance = 1e-4

    sun = CelestialBodyFactory.getSun()
    moon = CelestialBodyFactory.getMoon()

    satmodel = IsotropicDrag(crossSection, dragCoeff) # Cross sectional area and the drag coefficient

    initialOrbit = initial_orbit
    orbitType = initialOrbit.getType()
    initialState = SpacecraftState(initialOrbit, satellite_mass)
    tol = NumericalPropagator.tolerances(positionTolerance, initialOrbit, orbitType)

    integrator = DormandPrince853Integrator(minStep, maxstep, JArray_double.cast_(tol[0]), JArray_double.cast_(tol[1]))
    integrator.setInitialStepSize(initStep)

    propagator_num = NumericalPropagator(integrator)
    propagator_num.setOrbitType(orbitType)
    propagator_num.setInitialState(initialState)

    # Add Solar Radiation Pressure
    spacecraft = IsotropicRadiationSingleCoefficient(srpArea, cr)
    srpProvider = SolarRadiationPressure(sun, earth, spacecraft)
    propagator_num.addForceModel(srpProvider)

    # Add Gravity Force
    gravityProvider = GravityFieldFactory.getConstantNormalizedProvider(degree, torder, initialDate)
    gravityForce = HolmesFeatherstoneAttractionModel(earth.getBodyFrame(), gravityProvider)
    propagator_num.addForceModel(gravityForce)

    # Add Solid Tides
    solidTidesBodies = ArrayList().of_(CelestialBody)
    solidTidesBodies.add(sun)
    solidTidesBodies.add(moon)
    solidTidesBodies = solidTidesBodies.toArray()
    solidTides = SolidTides(earth.getBodyFrame(), 
                            gravityProvider.getAe(), gravityProvider.getMu(),
                            gravityProvider.getTideSystem(), 
                            IERSConventions.IERS_2010,
                            TimeScalesFactory.getUT1(IERSConventions.IERS_2010, True), 
                            solidTidesBodies)
    propagator_num.addForceModel(solidTides)

    # Add Third Body Attractions
    propagator_num.addForceModel(ThirdBodyAttraction(sun))
    propagator_num.addForceModel(ThirdBodyAttraction(moon)) 

    # Add Custom Drag Force
    atmosphere = CustomAtmosphereClass
    dragForce = DragForce(CustomAtmosphereClass, satmodel)
    propagator_num.addForceModel(dragForce)

    print(f'WOWOWO, {CustomAtmosphereClass}')
    #embed();quit();
    states = [initialState]
    tic = time.time()
    states = [propagator_num.propagate(tt)  for tt in tspan]
    toc = time.time()
    

    posvel = [state.getPVCoordinates() for state in states]
    poss = [state.getPosition() for state in posvel]
    vels = [state.getVelocity() for state in posvel]
    px = [pos.getX() * 1e-3 for pos in poss]
    py = [pos.getY() * 1e-3 for pos in poss]
    pz = [pos.getZ() * 1e-3 for pos in poss]
    vx = [vel.getX() * 1e-3 for vel in vels]
    vy = [vel.getY() * 1e-3 for vel in vels]
    vz = [vel.getZ() * 1e-3 for vel in vels]
    stat_list = [dur, toc - tic, px[-1], py[-1], pz[-1], vx[-1], vy[-1], vz[-1], step]
    print("Time interval [s]:", stat_list[0])
    print("Time step [s]:", stat_list[8])
    print("CPU time [s]:", stat_list[1])
    print("Final Pos [km]:", np.linalg.norm([px[-1], py[-1], pz[-1]]))
    print("Final Vel [km]:", np.linalg.norm([vx[-1], vy[-1], vz[-1]]))

    # Plot the satellite trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(px, py, pz, label='Trajectory')
    
    # plot earth
    phi = np.linspace(-np.pi, np.pi, 100)
    theta = np.linspace(-np.pi/2, np.pi/2, 50)
    X_Earth = r_Earth*1e-3 * np.outer(np.cos(phi), np.cos(theta)).T
    Y_Earth = r_Earth*1e-3 * np.outer(np.sin(phi), np.cos(theta)).T
    Z_Earth = r_Earth*1e-3 * np.outer(np.ones(np.size(phi)), np.sin(theta)).T
    ax.plot_surface(X_Earth, Y_Earth, Z_Earth, cmap='binary', alpha=0.35, antialiased=False, zorder = 1)
    
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Satellite Trajectory')
    ax.legend()
    plt.show()

    print('prop_done!!')
    embed();quit()

    densities = []
    for state in states:
        density = atmosphere.getDensity(state.getDate(), state.getPVCoordinates().getPosition(), state.getFrame())
        densities.append(density)
    
    return states, densities

class PersistenceMSIS():
    def __init__(self, omni2_data):
        """
        Initialize the Persistence Model.

        Args:
          - omni2_data (pd.DataFrame): OMNI2 data containing Ap, F10.7, and other parameters.

        """
        self.initial_date = None
        self.omni2_data = omni2_data

    def run(self, dt, lon, lat, alt):
        """
        Runs the MSIS model for the initial date using OMNI2 data to avoid online calls.

        Parameters:
            datetime_input (datetime): Datetime for the simulation.
            lon (float): Longitude in degrees.
            lat (float): Latitude in degrees.
            alt (float): Altitude in km.

        Returns:
            np.ndarray: Output from the MSIS model.
        """

        # Find the closest row in OMNI2 data
        #print("HEHRE")
        #embed();quit()

        # We want to replicate the initial state through the output so we only
        # keep the initial date
        if self.initial_date is None:
            self.initial_date = dt
            row = self.omni2_data.loc[0] #HACK: replace iwth last value in OMNI dataset
        else:
            row = self.omni2_data.iloc[(self.omni2_data['Timestamp'] - dt).abs().argmin()]

        f107_daily = row['f10.7_index_pred']
        ap_current = row['ap_index_nT_pred']

        # Prepare Ap indices using the helper function
        #print(row, self.initial_date, ap_current)

        # TODO: generate AP for past data points
        #aps = self._prepare_ap_indices(self.initial_date, ap_current) 
        aps = ap_current

        # Run the MSIS model
        try:
            result = msis.run(
                dates=[[dt]],
                lons=[[lon]],
                lats=[[lat]],
                alts=[[alt]],
                f107s=[[f107_daily]],
                aps=[[aps]],
                geomagnetic_activity=1 #TODO: Change this when AP past points is used. Account for geomagnetic activity (1 = Daily Ap mode, -1 = Storm-time Ap mode)
            )
        except:
            embed();quit()
        total_mass_density = result[0,0]  # Return the density for the specific point
        #print(f"Total mass density: {total_mass_density} kg/m^3, ap: {aps}, f107: {f107_daily}, lon: {lon}, lat: {lat}, alt: {alt}, dt: {dt}")
        return total_mass_density

    def _prepare_ap_indices(self, datetime_input, ap_current):
        """
        Private helper function to compute Ap indices and averages required for MSIS.

        Parameters:
            datetime_input (datetime): Datetime for the simulation.
            ap_current (float): Current daily Ap value.

        Returns:
            list: Prepared Ap array for MSIS input.
        """
        index = self.omni2_data.index[self.omni2_data['Timestamp'] == datetime_input][0]

        # Compute 3-hourly Ap indices
        ap_3hr_indices = [
            self.omni2_data.iloc[index - i]['ap_index_nT'] if (index - i) >= 0 else ap_current
            for i in range(0, 4)
        ]

        # Compute averages for specific periods
        ap_12_33_avg = np.mean([
            self.omni2_data.iloc[index - i]['ap_index_nT'] if (index - i) >= 0 else ap_current
            for i in range(12, 34, 3)
        ])
        ap_36_57_avg = np.mean([
            self.omni2_data.iloc[index - i]['ap_index_nT'] if (index - i) >= 0 else ap_current
            for i in range(36, 58, 3)
        ])

        # Prepare Ap array
        aps = [
            ap_current,  # Daily Ap
            ap_3hr_indices[0],  # Current 3-hour Ap
            ap_3hr_indices[1],  # 3 hours before
            ap_3hr_indices[2],  # 6 hours before
            ap_3hr_indices[3],  # 9 hours before
            ap_12_33_avg,       # Average of 12-33 hours prior
            ap_36_57_avg        # Average of 36-57 hours prior
        ]

        return aps

class MSISPersistenceAtmosphere(PythonAtmosphere):
    """
    CustomAtmosphere is a custom implementation of the PythonAtmosphere class
    that uses the PersistenceModel to compute atmospheric density and velocity.

    Attributes:
        atm (PersistenceModel): An instance of the PersistenceModel.
        earth (Body): The central body (Earth) for the atmospheric model.

    Methods:
        getMSISPersistence(input_df: pd.DataFrame) -> pd.DataFrame:
            Generates persistent MSIS data using the PersistenceModel.

        getDensity(date: AbsoluteDate, position: Vector3D, frame: Frame) -> float:
            Computes the atmospheric density at a given date, position, and frame
            using the PersistenceModel output.

        _position_to_geo(position: Vector3D) -> Tuple[float, float, float]:
            Helper method to convert position to latitude, longitude, and altitude.
    """
    def __init__(self, omni2, **kwargs):
        super().__init__()
        self.atm = PersistenceMSIS(omni2)

        r_Earth = Constants.IERS2010_EARTH_EQUATORIAL_RADIUS #m
        self.itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True) # International Terrestrial Reference Frame, earth fixed
        self.earth = OneAxisEllipsoid(
                         r_Earth,
                         Constants.IERS2010_EARTH_FLATTENING,
                         self.itrf
                    )

    def getDensity(self, date: AbsoluteDate, position: Vector3D, frame: Frame) -> float:
        """
        Compute the atmospheric density at a given date, position, and frame using the PersistenceModel output.

        Args:
            date (AbsoluteDate): The date for which to compute density.
            position (Vector3D): The position in the given frame.
            frame (Frame): The reference frame.

        Returns:
            float: The computed atmospheric density.
        """
        try:
            lat, lon, alt = self._position_to_geo(position, date)

            # Convert date
            time_str = date.toString(0)
            dt = pd.to_datetime(time_str).tz_localize(None)

            # Get persistence model output
            density = self.atm.run(dt, lon, lat, alt)
            #print(f"Date: {dt}, Density: {float(density)} kg/m^3")
            return float(density)
        except Exception as e:
            print(f"Error in getDensity: {e}")
            print(f"Date: {date}, Position: {position}, Frame: {frame}")
            raise


    def getVelocity(self, date: AbsoluteDate, position: Vector3D, frame: Frame):
        '''
        Get the inertial velocity of atmosphere molecules.
        By default, atmosphere is supposed to have a null
        velocity in the central body frame.</p>
        '''
        # get the transform from body frame to the inertial frame
        bodyToFrame = self.earth.getBodyFrame().getKinematicTransformTo(frame, date)
        # Inverse transform the position to the body frame
        posInBody = bodyToFrame.getStaticInverse().transformPosition(position)
        # Create PVCoordinates object assuming zero velocity in body frame
        pv_body = PVCoordinates(posInBody, Vector3D.ZERO)
        # Transform the position/velocity (PV) coordinates to the given frame
        pvFrame = bodyToFrame.transformOnlyPV(pv_body)
        # Return the velocity in the current frame
        return pvFrame.getVelocity()

    def _position_to_geo(self, positionICRF, date):
            """
            Converts a position vector (in ICRF frame) to geodetic coordinates (lat, lon, alt).
    
            Parameters:
            positionICRF: Vector3D, position vector in ICRF frame.
            date: AbsoluteDate, the date of the position.
    
            Returns:
            tuple: (latitude, longitude, altitude) in degrees and meters.
            """
            # Create a PVCoordinates object (assuming zero velocity)
            pvICRF = PVCoordinates(positionICRF, Vector3D.ZERO)
    
            # Transform position from ICRF to ECEF (ITRF)
            transform = self.earth.getBodyFrame().getTransformTo(self.itrf, date)
            pvECEF = transform.transformPVCoordinates(pvICRF)
            positionECEF = pvECEF.getPosition()
    
            # Convert the ECEF position to geodetic coordinates
            geodeticPoint = self.earth.transform(positionECEF, self.itrf, date)
    
            # Extract latitude, longitude, and altitude
            latitude = geodeticPoint.getLatitude()  # radians
            longitude = geodeticPoint.getLongitude()  # radians
            altitude = geodeticPoint.getAltitude()  # meters
    
            # Convert radians to degrees for latitude and longitude
            latitudeDeg = math.degrees(latitude)
            longitudeDeg = math.degrees(longitude)
    
            return latitudeDeg, longitudeDeg, altitude

class PersistenceModel(nn.Module):
    def __init__(self, plot_trajectory=False):
        super().__init__()
        self.plot = plot_trajectory
    
    def forward(self, omni2_data, initial_state={}):        
        states, densities = prop_orbit(
                                initial_state, 
                                MSISPersistenceAtmosphere,
                                atm_model_data=omni2_data, 
                                plot_trajectory=self.plot
                            )

        return self._convert_to_df(states, densities)


    def _convert_to_df(self, states, densities):
        """
        Generates a DataFrame with timestamps and atmospheric densities.

        Parameters:
        - states (list of SpacecraftState): List of spacecraft states from the propagator.
        - densities (list of float): List of atmospheric densities corresponding to each state.

        Returns:
        - pd.DataFrame: A DataFrame with columns ['timestamp', 'density'].
        """
        # Initialize a list to hold data
        density_data = []

        # Iterate through states and densities
        for state, density in zip(states, densities):
            # Extract timestamp from the state
            timestamp = pd.to_datetime(state.getDate().toString(0))  # Convert to pandas datetime

            # Append data to the list
            density_data.append({'Timestamp': timestamp, 'Density (kg/m3)': density})

        # Convert the list to a DataFrame
        df = pd.DataFrame(density_data)

        return df



#==============================================================================
# Load OMNI2 data directly
file_path = './data/sat_density_omni_forcasted/sat_density_00000.csv'
#omni2_data = pd.read_csv(file_path, usecols=['Timestamp', 'f10.7_index', 'ap_index_nT'])
omni2_data = pd.read_csv(file_path, usecols=['Timestamp','Orbit Mean Density (kg/m^3)','ap_index_nT_pred','f10.7_index_pred'])

# Process OMNI2 data immediately after loading
omni2_data['Timestamp'] = pd.to_datetime(omni2_data['Timestamp'])
omni2_data = omni2_data.ffill()
print(omni2_data)

atm = MSISPersistenceAtmosphere(omni2_data)
date = AbsoluteDate(2013, 11, 29, 0, 0, 0.000, TimeScalesFactory.getUTC()
)

mu = Constants.IERS2010_EARTH_MU #m^3/s^2
degree = 70
torder = 70
cr = 1.0
utc = TimeScalesFactory.getUTC()
sun = CelestialBodyFactory.getSun()

# Initialize the Vector3D for position
position = Vector3D(5_870_038.485921082, 2_396_433.1768343644, 2_396_433.176834364)

# Initialize the Frame (EME2000)
frame = FramesFactory.getEME2000() 

r_Earth = Constants.IERS2010_EARTH_EQUATORIAL_RADIUS #m
itrf    = FramesFactory.getITRF(IERSConventions.IERS_2010, True) # International Terrestrial Reference Frame, earth fixed
inertialFrame = FramesFactory.getEME2000()
earth = OneAxisEllipsoid(r_Earth,
                         Constants.IERS2010_EARTH_FLATTENING,
                         itrf)
mu = Constants.IERS2010_EARTH_MU #m^3/s^2
utc = TimeScalesFactory.getUTC()

#File ID,Timestamp,Semi-major Axis (km),Eccentricity,Inclination (deg),RAAN (deg),Argument of Perigee (deg),True Anomaly (deg),Latitude (deg),Longitude (deg),Altitude (km)
#0,2000-08-02 04:50:33,6826.387246918713,0.0038817999884486,87.27530555555555,144.13511111111112,257.3143888888889,102.383269692981,43.63781476773328,-62.54312803592875,466.44889019083695

rp0 = r_Earth + 400 * 1e3 # perigee radius (m)
ra0 = r_Earth + 600 * 1e3 # apogee radius (m)
deg = np.pi / 180
a0 = 6826.387246918713 * 1e3 # semi-major axis (m)
e0 = 0.0038817999884486 # eccentricity 
w0 = radians(257.3143888888889) # perigee argument (rad)
i0 = radians(87.27530555555555) # inclination (rad)
ra0 = radians(144.13511111111112) # right ascension of ascending node (rad)
M0 = radians(102.383269692981) # anomaly

initialDate = AbsoluteDate(2000, 8, 2, 4, 50, 33.000, TimeScalesFactory.getUTC()) # date of orbit parameters

initialOrbit = KeplerianOrbit(a0, e0, i0, w0, ra0, M0, PositionAngleType.TRUE, inertialFrame, initialDate, mu)

duration = 1 * 86400.0 # 1 day in seconds

secs = 60.0
mins = 10
step = secs * mins # propagation step size [s]

states, densities = prop_orbit(initialOrbit, duration, step, atm)
#atm.getDensity(date, position, frame)

# model = PersistenceModel(plot_trajectory=True)
# predictions = model(omni2_data)