import numpy as np
import litebird_sim as lbs

# -------------------------------
# Funzione add_lbsNoise
# -------------------------------
def add_lbsNoise(tod, det, block_duration_s):
    """
    Genera rumore strumentale tipo LBS (1/f + bianco) per un TOD.
    """
    n_block = int(block_duration_s * det.sampling_rate_hz)
    n_samp = tod.shape[1]

    # RNG per ogni detector
    dets_random = [np.random.default_rng() for _ in range(tod.shape[0])]

    for start in range(0, n_samp, n_block):
        end = min(start + n_block, n_samp)

        lbs.add_noise(
            tod=tod[:, start:end],
            noise_type="one_over_f",
            sampling_rate_hz=det.sampling_rate_hz,
            net_ukrts=det.net_ukrts,
            fknee_mhz=det.fknee_mhz,
            fmin_hz=det.fmin_hz,
            alpha=det.alpha,
            dets_random=dets_random,
            scale=1.0,
        )

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
tod = add_lbsNoise(tod, det, block_duration_s=100)

print("Funzione add_lbsNoise eseguita correttamente")
