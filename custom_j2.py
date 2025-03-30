import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pymsis import msis
from pyproj import Transformer
from numba import njit

# Earth's gravitational parameter and WGS84 equatorial radius.
MU = 3.986004418e14       # [m^3/s^2]
R_E = 6378137.0           # [m]

# Create a transformer to convert from ECEF (EPSG:4978) to WGS84 geodetic (EPSG:4326)
ecef_to_geo = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)

# ----------------------------------------------------------------------
# 1. Orbital Elements to State Vector Conversion (ECI)
# ----------------------------------------------------------------------
def oe2rv(a_km, e, i_deg, raan_deg, argp_deg, nu_deg):
    """
    Convert orbital elements (in km, deg) to ECI state vectors (m, m/s).
    """
    i = math.radians(i_deg)
    raan = math.radians(raan_deg)
    argp = math.radians(argp_deg)
    nu = math.radians(nu_deg)
    
    a = a_km * 1e3
    p = a * (1 - e**2)
    r_val = p / (1 + e * math.cos(nu))
    
    r_pf = np.array([r_val * math.cos(nu), r_val * math.sin(nu), 0.0])
    v_pf = np.array([
        -math.sqrt(MU / p) * math.sin(nu),
         math.sqrt(MU / p) * (e + math.cos(nu)),
         0.0
    ])
    
    cos_raan = math.cos(raan)
    sin_raan = math.sin(raan)
    cos_i = math.cos(i)
    sin_i = math.sin(i)
    cos_argp = math.cos(argp)
    sin_argp = math.sin(argp)
    
    R = np.array([
        [cos_raan*cos_argp - sin_raan*sin_argp*cos_i,
         -cos_raan*sin_argp - sin_raan*cos_argp*cos_i,
         sin_raan*sin_i],
        [sin_raan*cos_argp + cos_raan*sin_argp*cos_i,
         -sin_raan*sin_argp + cos_raan*cos_argp*cos_i,
         -cos_raan*sin_i],
        [sin_argp*sin_i,
         cos_argp*sin_i,
         cos_i]
    ])
    
    r_eci = R @ r_pf
    v_eci = R @ v_pf
    return r_eci, v_eci

# ----------------------------------------------------------------------
# Helper: Compute GMST in radians (approximate)
# ----------------------------------------------------------------------
def gmst(dt):
    """
    Compute an approximate Greenwich Mean Sidereal Time (radians)
    for a given datetime (UTC).
    """
    JD = (dt - datetime(2000, 1, 1, 12)).total_seconds()/86400.0 + 2451545.0
    GMST_hours = 18.697374558 + 24.06570982441908 * (JD - 2451545.0)
    GMST_hours %= 24.0
    return (GMST_hours / 24.0) * 2 * math.pi

# ----------------------------------------------------------------------
# Helper: Convert ECI to ECEF using GMST rotation.
# ----------------------------------------------------------------------
def eci_to_ecef(r_eci, current_time):
    """
    Convert an ECI vector (m) to ECEF coordinates using a simple GMST rotation.
    """
    theta = gmst(current_time)
    cos_theta = math.cos(-theta)
    sin_theta = math.sin(-theta)
    R = np.array([[cos_theta, -sin_theta, 0],
                  [sin_theta,  cos_theta, 0],
                  [0,          0,         1]])
    return R @ r_eci

# ----------------------------------------------------------------------
# 2. Dynamic MSIS Call for Atmospheric Density using WGS84 altitude
# ----------------------------------------------------------------------
def get_density(current_time, r_eci):
    """
    Convert the current ECI position to ECEF and then to WGS84 geodetic coordinates,
    then call pymsis dynamically to obtain the atmospheric density.
    MSIS expects altitude in km.
    """
    r_ecef = eci_to_ecef(r_eci, current_time)
    x, y, z = r_ecef
    lon, lat, alt_m = ecef_to_geo.transform(x, y, z)
    alt_km = alt_m / 1000.0
    f107_daily = 155.1
    aps = [15] * 7
    try:
        result = msis.run(
            dates=[current_time],
            lons=[lon],
            lats=[lat],
            alts=[alt_km],
            f107s=[f107_daily],
            aps=[aps]
        )
        density = result[0, 0]
    except Exception as e:
        raise Exception(f"Error running MSIS at alt {alt_km:.2f} km: {e}")
    return density

# ----------------------------------------------------------------------
# 3. Acceleration Models (Numerical parts jitted)
# ----------------------------------------------------------------------
@njit
def gravitational_acceleration(r):
    # Pure numerical function
    x, y, z = r[0], r[1], r[2]
    r_norm = math.sqrt(x*x + y*y + z*z)
    r_norm3 = r_norm**3
    acc = np.array([-MU*x / r_norm3,
                    -MU*y / r_norm3,
                    -MU*z / r_norm3])
    J2 = 1.08263e-3
    factor = 1.5 * J2 * MU * (R_E**2) / (r_norm**5)
    acc[0] += factor * x * (5*(z**2)/(r_norm**2) - 1)
    acc[1] += factor * y * (5*(z**2)/(r_norm**2) - 1)
    acc[2] += factor * z * (5*(z**2)/(r_norm**2) - 3)
    return acc

@njit
def drag_from_density(density, v, mass, Cd, A):
    v_norm = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if v_norm < 1e-8:
        return np.zeros(3)
    return -0.5 * Cd * A * density * v_norm * v / mass

@njit
def rk4_step(r, v, density, dt, mass, Cd, A):
    # Compute derivatives assuming density is constant over dt.
    k1_r = v
    k1_v = gravitational_acceleration(r) + drag_from_density(density, v, mass, Cd, A)
    
    r_temp = r + 0.5 * dt * k1_r
    v_temp = v + 0.5 * dt * k1_v
    k2_r = v_temp
    k2_v = gravitational_acceleration(r_temp) + drag_from_density(density, v_temp, mass, Cd, A)
    
    r_temp = r + 0.5 * dt * k2_r
    v_temp = v + 0.5 * dt * k2_v
    k3_r = v_temp
    k3_v = gravitational_acceleration(r_temp) + drag_from_density(density, v_temp, mass, Cd, A)
    
    r_temp = r + dt * k3_r
    v_temp = v + dt * k3_v
    k4_r = v_temp
    k4_v = gravitational_acceleration(r_temp) + drag_from_density(density, v_temp, mass, Cd, A)
    
    r_new = r + (dt/6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    v_new = v + (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    return r_new, v_new

# ----------------------------------------------------------------------
# 4. RK4 Orbit Propagation with Dynamic MSIS Density
# ----------------------------------------------------------------------
def propagate_orbit(r0, v0, t0, dt, steps, mass, Cd, A):
    """
    Propagate the orbit from initial state (r0, v0) starting at time t0
    using RK4 integration with time step dt (seconds) for a total of
    'steps' steps.
    For each step, we dynamically compute the density (non-jitted) and then
    use a jitted RK4 step with that density assumed constant over dt.
    
    Returns lists of timestamps, state vectors, and densities.
    """
    times = [t0]
    states = [np.hstack((r0, v0))]
    densities = [get_density(t0, r0)]
    
    r = r0.copy()
    v = v0.copy()
    current_time = t0
    for i in range(steps):
        # Get dynamic density at current state (this call is not jitted).
        density = get_density(current_time, r)
        # Advance one RK4 step with the jitted function.
        r, v = rk4_step(r, v, density, dt, mass, Cd, A)
        current_time = current_time + timedelta(seconds=dt)
        
        times.append(current_time)
        states.append(np.hstack((r, v)))
        densities.append(get_density(current_time, r))
        
    return times, np.array(states), densities

# ----------------------------------------------------------------------
# 5. Helper: Convert ECI state vector to geodetic LLA.
# ----------------------------------------------------------------------
def eci_to_lla(r_eci, current_time):
    r_ecef = eci_to_ecef(r_eci, current_time)
    x, y, z = r_ecef
    lon, lat, alt_m = ecef_to_geo.transform(x, y, z)
    return lat, lon, alt_m/1000.0

# ----------------------------------------------------------------------
# 6. Main Example: Propagate and Plot Results
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Input (from CSV):
    import time
    start_time = time.time()
    timestamp = "2000-08-02 04:50:33"
    a_km = 6826.387246918713
    e = 0.0038817999884486
    i_deg = 87.27530555555555
    raan_deg = 144.13511111111112
    argp_deg = 257.3143888888889
    nu_deg = 102.383269692981
    
    # Convert orbital elements to initial state vectors.
    r0, v0 = oe2rv(a_km, e, i_deg, raan_deg, argp_deg, nu_deg)
    
    # Propagation parameters.
    t0 = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    dt_integ = 10.0                # integration step: 10 seconds
    total_seconds = 3 * 86400      # 3 days
    steps_integ = int(total_seconds / dt_integ)
    mass = 1000.0   # kg
    Cd = 2.2        # Drag coefficient
    A = 20.0        # Cross-sectional area (m^2)
    
    # Propagate with the hybrid (jitted inner RK4) integrator.
    times_fine, states_fine, densities_fine = propagate_orbit(r0, v0, t0, dt_integ, steps_integ, mass, Cd, A)
    
    # Downsample to 10-minute intervals (every 60 steps).
    sample_rate = 60
    times = times_fine[::sample_rate]
    states = states_fine[::sample_rate]
    densities = [densities_fine[i] for i in range(0, len(densities_fine), sample_rate)]
    
    # Build a DataFrame.
    df = pd.DataFrame({
        "timestamp": times,
        "x": states[:, 0],
        "y": states[:, 1],
        "z": states[:, 2],
        "vx": states[:, 3],
        "vy": states[:, 4],
        "vz": states[:, 5],
        "density": densities
    })
    
    # Convert ECI states to geodetic LLA for plotting.
    lats, lons, alts = [], [], []
    for idx, row in df.iterrows():
        r_eci = np.array([row["x"], row["y"], row["z"]])
        lat_val, lon_val, alt_val = eci_to_lla(r_eci, row["timestamp"])
        lats.append(lat_val)
        lons.append(lon_val)
        alts.append(alt_val)
    end_time = time.time()
    print(f'time: {end_time - start_time}')
    
    # Plot LLA and density.
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    axs[0].plot(df["timestamp"], lats, label="Latitude (deg)")
    axs[0].plot(df["timestamp"], lons, label="Longitude (deg)")
    axs[0].plot(df["timestamp"], alts, label="Altitude (km)")
    axs[0].set_ylabel("Geodetic Coordinates")
    axs[0].legend(loc="upper left")
    axs[0].set_title("Geodetic LLA vs Time")
    
    axs[1].plot(df["timestamp"], df["density"], label="Density (kg/m^3)", color="tab:red")
    axs[1].set_ylabel("Density (kg/m^3)")
    axs[1].set_xlabel("Time")
    axs[1].legend(loc="upper left")
    axs[1].set_title("Atmospheric Density vs Time")
    
    plt.tight_layout()
    plt.show()
