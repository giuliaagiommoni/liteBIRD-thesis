import numpy as np
import ducc0
import litebird_sim as lbs

# -------------------------------
# Funzione add_OofaNoise
# -------------------------------
def add_OofaNoise(tod, det, block_duration_s):
    """
    Genera rumore 1/f + bianco usando OofaNoise (ducc0) per un TOD.
    """
    n_block = int(block_duration_s * det.sampling_rate_hz)
    n_samp = tod.shape[1]

    # Crea un generatore OofaNoise per ogni detector
    noise_gens = [
        ducc0.misc.OofaNoise(
            sigmawhite=det.net_ukrts * 1e-6 * np.sqrt(det.sampling_rate_hz),
            f_knee=det.fknee_mhz * 1e-3,
            f_min=det.fmin_hz,
            f_samp=det.sampling_rate_hz,
            slope=-det.alpha,
        )
        for _ in range(tod.shape[0])
    ]

    # Genera rumore a blocchi
    for start in range(0, n_samp, n_block):
        end = min(start + n_block, n_samp)
        block_len = end - start

        for d in range(tod.shape[0]):
            white_chunk = np.random.normal(0., 1., block_len)               # Rumore bianco
            tod[d, start:end] += noise_gens[d].filterGaussian(white_chunk)  # Aggiunge rumore 1/f in-place
            del white_chunk  # Libera memoria esplicitamente

    return tod

# -------------------------------
# Setup simulazione
# -------------------------------
start_time = 0
time_span_s = 1000.0

sim = lbs.Simulation(
    start_time=start_time,
    duration_s=time_span_s,
    random_seed=12345,
    imo=lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)
)

sim.set_scanning_strategy(
    lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=np.deg2rad(0),
        precession_rate_hz=0,
        spin_rate_hz=1/60,
        start_time=start_time
    ),
    delta_time_s=5.0
)

sim.set_instrument(
    lbs.InstrumentInfo(
        boresight_rotangle_rad=0.0,
        spin_boresight_angle_rad=np.deg2rad(90),
        spin_rotangle_rad=np.deg2rad(75)
    )
)

# -------------------------------
# Mock Detector
# -------------------------------
det = lbs.DetectorInfo(
    name="Boresight_detector",
    sampling_rate_hz=20.,
    bandcenter_ghz=100.0,
    net_ukrts=50.0,
    fknee_mhz=500.,
    fmin_hz=1e-5,
    alpha=1
)

sim.create_observations(detectors=det)

# -------------------------------
# Aggiunta rumore
# -------------------------------
tod = sim.observations[0].tod
tod = add_OofaNoise(tod, det, block_duration_s=100)

print("Funzione add_OofaNoise eseguita correttamente")
