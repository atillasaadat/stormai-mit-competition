"""
Refactored orbit propagation and MSIS persistence simulation.
This code uses Orekit for high-fidelity orbit propagation and a custom
MSIS persistence atmosphere model. It is organized in an object-oriented
manner so that multiple file IDs can be processed.
"""

import math
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datahandler import DataHandler
from pymsis import msis

# Orekit imports
import orekit
from orekit.pyhelpers import (
    setup_orekit_curdir,
    datetime_to_absolutedate,
)
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.frames import FramesFactory, Frame
from org.orekit.models.earth.atmosphere import PythonAtmosphere
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid, CelestialBody
from org.orekit.utils import Constants, PVCoordinates, IERSConventions
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.propagation import SpacecraftState
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction
from org.orekit.forces.gravity import SolidTides
from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient
from org.orekit.forces.drag import IsotropicDrag, DragForce
from orekit import JArray_double
from java.util import ArrayList
from org.hipparchus.geometry.euclidean.threed import Vector3D
from math import radians

# Initialize the Orekit JVM
orekit.initVM()
setup_orekit_curdir(from_pip_library=True)

# =============================================================================
# Orbit Propagation Class
# =============================================================================
class OrbitPropagator:
    """
    Propagates a given orbit using Orekit's numerical propagator.
    It uses a custom atmospheric model (e.g. MSISPersistenceAtmosphere)
    for drag force calculations.
    """

    def __init__(self, initial_orbit, custom_atmosphere, sat_config, sim_config):
        """
        Initialize the orbit propagator.

        Args:
            initial_orbit (KeplerianOrbit): The initial orbit.
            custom_atmosphere (PythonAtmosphere): Custom atmosphere model.
        """
        self.initial_orbit = initial_orbit
        self.sat_config = sat_config
        self.sim_config = sim_config
        self.atmosphere = custom_atmosphere

    def propagate(self, data):
        """
        Propagates the orbit and computes atmospheric density along the trajectory.

        Args:
            forecasted_omni2_data (pd.DataFrame): DataFrame used for the MSIS model.

        Returns:
            tuple: (timestamps, states, densities)
        """
        logger.info("Initializing orbit propagation.")

        # Time span: use forecasted timestamps from the forecasted OMNI2 DataFrame
        # (timestamps must be in pandas datetime format)
        timestamp_series = pd.to_datetime(data["Timestamp"])
        tspan = [
            datetime_to_absolutedate(ts.to_pydatetime()) for ts in timestamp_series
        ]

        # Get central bodies
        sun = CelestialBodyFactory.getSun()
        moon = CelestialBodyFactory.getMoon()
        # Earth for gravity and SRP
        itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        r_Earth = Constants.IERS2010_EARTH_EQUATORIAL_RADIUS
        earth = OneAxisEllipsoid(r_Earth, Constants.IERS2010_EARTH_FLATTENING, itrf)
        mu = Constants.IERS2010_EARTH_MU

        # Initial spacecraft state
        initial_date = self.initial_orbit.getDate()
        initial_state = SpacecraftState(self.initial_orbit, self.sat_config["satellite_mass_kg"])
        orbit_type = self.initial_orbit.getType()
        tol = NumericalPropagator.tolerances(self.sim_config["pos_tol"], self.initial_orbit, orbit_type)

        # Set up numerical integrator and propagator
        integrator = DormandPrince853Integrator(
            self.sim_config["min_step"], self.sim_config["max_step"], JArray_double.cast_(tol[0]), JArray_double.cast_(tol[1])
        )
        integrator.setInitialStepSize(self.sim_config["init_step"])
        propagator = NumericalPropagator(integrator)
        propagator.setOrbitType(orbit_type)
        propagator.setInitialState(initial_state)

        # --- Add force models ---
        # 1. Solar Radiation Pressure
        srp_model = IsotropicRadiationSingleCoefficient(self.sat_config["srp_area_m2"], self.sat_config["cr"])
        srp_provider = SolarRadiationPressure(sun, earth, srp_model)
        propagator.addForceModel(srp_provider)

        # 2. Gravity Force using Holmes-Featherstone model
        gravity_provider = GravityFieldFactory.getConstantNormalizedProvider(self.sim_config['spherical_harmonics'][0], 
                                                                             self.sim_config['spherical_harmonics'][1], 
                                                                             initial_date)
        gravity_force = HolmesFeatherstoneAttractionModel(earth.getBodyFrame(), gravity_provider)
        propagator.addForceModel(gravity_force)

        # 3. Solid Tides
        solid_tides_bodies = ArrayList().of_(CelestialBody)
        solid_tides_bodies.add(sun)
        solid_tides_bodies.add(moon)
        solid_tides = SolidTides(
            earth.getBodyFrame(),
            gravity_provider.getAe(),
            gravity_provider.getMu(),
            gravity_provider.getTideSystem(),
            IERSConventions.IERS_2010,
            TimeScalesFactory.getUT1(IERSConventions.IERS_2010, True),
            solid_tides_bodies.toArray(),
        )
        propagator.addForceModel(solid_tides)

        # 4. Third Body Attractions (Sun and Moon)
        propagator.addForceModel(ThirdBodyAttraction(sun))
        propagator.addForceModel(ThirdBodyAttraction(moon))

        # 5. Drag Force using custom atmospheric model
        drag_model = IsotropicDrag(self.sat_config["cross_section_m2"], self.sat_config["drag_coeff"])
        drag_force = DragForce(self.atmosphere, drag_model)
        propagator.addForceModel(drag_force)

        # Propagate over the entire timespan
        states = []
        logger.info(f"Beginning orbit propagation over {len(tspan)} time steps.")
        tic = time.time()
        progress_interval = max(1, len(tspan) // 20)  # Calculate interval for 5% progress
        for idx, current_date in enumerate(tspan):
            if idx % progress_interval == 0:
                progress = (idx / len(tspan)) * 100
                logger.info(f"Propagation progress: {progress:.1f}%")
            logger.debug(f"Propagating to {current_date} ({idx + 1}/{len(tspan)})")
            state = propagator.propagate(current_date)
            states.append(state)
        toc = time.time()
        logger.info(f"Propagation completed in {toc - tic:.3f} seconds.")
        logger.info("Propagation progress: 100.0%")

        # Compute atmospheric densities for each propagated state
        densities = []
        for state in states:
            density = self.atmosphere.getDensity(
                state.getDate(), state.getPVCoordinates().getPosition(), state.getFrame()
            )
            densities.append(float(density))

        # (Optional: trajectory plotting code could be added here)
        return tspan, states, densities

# =============================================================================
# Persistence and MSIS Atmosphere Classes
# =============================================================================
class PersistenceMSIS:
    """
    Persistence model for MSIS. Combines forecasted and historical OMNI2 data
    to drive the MSIS atmospheric model.
    """

    def __init__(self, logger, all_space_weather_data):
        """
        Initialize the persistence model by combining historical and forecasted data.
        """
        self.logger = logger
        self.all_space_weather_data = all_space_weather_data.copy()

    def run(self, dt, lon, lat, alt):
        """
        Runs the MSIS model for the given datetime and geodetic coordinates.
        """
        index = (self.all_space_weather_data["Timestamp"] - dt).abs().idxmin()
        f107_daily = float(self.all_space_weather_data.iloc[index - 1]["f10.7_index"])
        ap_current = float(self.all_space_weather_data.iloc[index - 1]["ap_index_nT"])

        # Prepare Ap indices
        aps = self.prepare_ap_indices(ap_current, index)
        alt = alt / 1e3  # Convert to km
        try:
            result = msis.run(
                dates=[[dt]],
                lons=[[lon]],
                lats=[[lat]],
                alts=[[alt]],
                f107s=[[f107_daily]],
                aps=[aps],
            )
        except Exception as e:
            logger.error(f"Error running MSIS: {e}")
            raise
        density = result[0, 0]
        self.logger.debug(f"Running MSIS for {dt} at lon={lon}, lat={lat}, alt={alt}, Ap : {aps}, f107={f107_daily}, density: {density}")
        return density

    def prepare_ap_indices(self, ap_current, index):
        """
        Helper method to compute the Ap index array needed by MSIS.
        """
        ap_3hr = [
            self.all_space_weather_data.iloc[index - i]["ap_index_nT"]
            if (index - i) >= 0
            else ap_current
            for i in range(4)
        ]
        ap_12_33_avg = np.mean(
            [
                self.all_space_weather_data.iloc[index - i]["ap_index_nT"]
                if (index - i) >= 0
                else ap_current
                for i in range(12, 34, 3)
            ]
        )
        ap_36_57_avg = np.mean(
            [
                self.all_space_weather_data.iloc[index - i]["ap_index_nT"]
                if (index - i) >= 0
                else ap_current
                for i in range(36, 58, 3)
            ]
        )
        aps = [
            float(ap_current),
            float(ap_3hr[0]),
            float(ap_3hr[1]),
            float(ap_3hr[2]),
            float(ap_3hr[3]),
            float(ap_12_33_avg),
            float(ap_36_57_avg),
        ]
        return aps


class MSISPersistenceAtmosphere(PythonAtmosphere):
    """
    Custom atmospheric model that uses PersistenceMSIS to compute density.
    """

    def __init__(self, logger, omni_data):
        super().__init__()
        self.logger = logger
        self.atm = PersistenceMSIS(logger, omni_data)

        r_Earth = Constants.IERS2010_EARTH_EQUATORIAL_RADIUS  # m
        self.itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        self.earth = OneAxisEllipsoid(r_Earth, Constants.IERS2010_EARTH_FLATTENING, self.itrf)

    def getDensity(self, date: AbsoluteDate, position: Vector3D, frame: Frame) -> float:
        """
        Compute atmospheric density at the given date, position, and frame.
        """
        try:
            lat, lon, alt = self.__position_to_geo(position, date)
            dt = pd.to_datetime(date.toString(0)).tz_localize(None)
            density = self.atm.run(dt, lon, lat, alt)
            return float(density)
        except Exception as e:
            logger.error(f"Error in getDensity: {e}")
            raise

    def getVelocity(self, date: AbsoluteDate, position: Vector3D, frame: Frame):
        """
        Compute atmospheric velocity (assumed zero in Earth-fixed frame).
        """
        bodyToFrame = self.earth.getBodyFrame().getKinematicTransformTo(frame, date)
        posInBody = bodyToFrame.getStaticInverse().transformPosition(position)
        pv_body = PVCoordinates(posInBody, Vector3D.ZERO)
        pvFrame = bodyToFrame.transformOnlyPV(pv_body)
        return pvFrame.getVelocity()

    def __position_to_geo(self, position, date):
        """
        Convert a position vector in ICRF to geodetic coordinates.
        Returns (latitude [deg], longitude [deg], altitude [m]).
        """
        pvICRF = PVCoordinates(position, Vector3D.ZERO)
        transform = self.earth.getBodyFrame().getTransformTo(self.itrf, date)
        pvECEF = transform.transformPVCoordinates(pvICRF)
        positionECEF = pvECEF.getPosition()
        geodeticPoint = self.earth.transform(positionECEF, self.itrf, date)
        lat = math.degrees(geodeticPoint.getLatitude())
        lon = math.degrees(geodeticPoint.getLongitude())
        alt = geodeticPoint.getAltitude()
        return lat, lon, alt


# =============================================================================
# SimulationRunner Class
# =============================================================================
class SimulationRunner:
    """
    Runs the complete simulation pipeline for a given file ID.
    """

    def __init__(self, logger, sat_config, sim_config):
        """
        Initialize the runner.

        Args:
            file_id (int): The file identifier.
            data_handler (DataHandler): Data handler instance.
            propagation_minutes (int): Number of minutes to propagate.
            propagation_secs (int): Seconds per propagation step.
        """
        self.logger = logger
        self.sat_config = sat_config
        self.sim_config = sim_config

    def run_simulation(self, file_id, initial_state, space_weather_data, sat_density_truth):
        """
        Runs the simulation:
            1. Loads required data.
            2. Sets up the MSIS persistence atmosphere.
            3. Builds the initial orbit.
            4. Propagates the orbit.
            5. Matches computed densities with forecasted timestamps.
            6. Saves the output.
        """
        logger.info(f"Starting simulation for File ID {file_id:05d}.")

        # Build the custom atmosphere model
        atmosphere = MSISPersistenceAtmosphere(self.logger, space_weather_data)

        # Build the initial orbit using the initial state values
        a0 = float(initial_state["Semi-major Axis (km)"]) * 1e3  # meters
        e0 = float(initial_state["Eccentricity"])
        w0 = radians(initial_state["Argument of Perigee (deg)"])
        i0 = radians(initial_state["Inclination (deg)"])
        ra0 = radians(initial_state["RAAN (deg)"])
        M0 = radians(initial_state["True Anomaly (deg)"])

        initial_date = pd.to_datetime(initial_state["Timestamp"])
        abs_date = AbsoluteDate(
            initial_date.year,
            initial_date.month,
            initial_date.day,
            initial_date.hour,
            initial_date.minute,
            initial_date.second + initial_date.microsecond * 1e-6,
            TimeScalesFactory.getUTC(),
        )
        inertial_frame = FramesFactory.getEME2000()
        mu = Constants.IERS2010_EARTH_MU

        initial_orbit = KeplerianOrbit(
            a0,
            e0,
            i0,
            w0,
            ra0,
            M0,
            PositionAngleType.TRUE,
            inertial_frame,
            abs_date,
            mu,
        )

        # Propagate the orbit
        propagator = OrbitPropagator(initial_orbit, atmosphere, self.sat_config, self.sim_config)
        timestamps, states, densities = propagator.propagate(sat_density_truth)

        # Convert Orekit AbsoluteDate timestamps to pandas datetime
        timestamps_pd = [pd.to_datetime(ts.toString(0)).tz_localize(None) for ts in timestamps]

        # Update forecasted DataFrame with computed densities
        new_df = sat_density_truth.copy()
        new_df["MSIS Density (kg/m^3)"] = np.nan
        for ts, density in zip(timestamps_pd, densities):
            new_df.loc[new_df["Timestamp"] == ts, "MSIS Density (kg/m^3)"] = density

        #self.data_handler.save_propagated_results(self.file_id, new_df)
        return new_df


if __name__ == "__main__":
    # Satellite and force model parameters
    sat_config = {
        "satellite_mass_kg": 522.0,  # kg
        "cross_section_m2": 0.9,  # m^2
        "srp_area_m2": 0.9,  # m^2
        "drag_coeff": 2.2,
        "cr": 1.0,
    }
    # Integrator tolerances and step sizes
    sim_config = {
        "min_step": 1e-6,
        "max_step": 100.0,
        "init_step": 1.0,
        "pos_tol": 1e-4,
        "spherical_harmonics": (70, 70),
    }

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    dh = DataHandler(logger,
        omni2_folder = Path("./data/omni2"),
        initial_state_folder = Path("./data/initial_state"),
        sat_density_folder = Path("./data/sat_density"),
        forcasted_omni2_folder = Path("./data/forcasted_omni2"),
        sat_density_omni_forcasted_folder = Path("./data/sat_density_omni_forcasted"),
        sat_density_omni_propagated_folder = Path("./data/sat_density_omni_propagated"),
    )
    sim = SimulationRunner(logger, sat_config, sim_config)
    density_results = sim.run_simulation(1, dh.get_initial_state(1), dh.read_csv_data(1, dh.omni2_folder), dh.read_csv_data(1, dh.sat_density_folder))
    from IPython import embed; embed(); quit()