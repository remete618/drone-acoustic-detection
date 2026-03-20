import numpy as np

DRONE_PROFILES = {
    "fpv_5inch": {
        "fundamental_hz": 280,
        "harmonics": [560, 840, 1120],
        "harmonic_amplitudes": [0.6, 0.3, 0.15],
        "broadband_amplitude": 0.05,
        "n_motors": 4,
        "n_blades": 3,
        "motor_kv": 2450,
        "hover_rpm_spread": 0.15,   # 15% spread between motors
        "pid_jitter_pct": 0.18,     # 18% PID-driven RPM variation
        "session_rpm_var_pct": 0.08, # 8% session-to-session RPM shift (battery, temp, prop wear)
    },
    "micro_whoop": {
        "fundamental_hz": 600,
        "harmonics": [1200, 1800],
        "harmonic_amplitudes": [0.4, 0.2],
        "broadband_amplitude": 0.03,
        "n_motors": 4,
        "n_blades": 2,
        "motor_kv": 19000,
        "hover_rpm_spread": 0.20,
        "pid_jitter_pct": 0.22,
        "session_rpm_var_pct": 0.10,
    },
    "dji_mini": {
        "fundamental_hz": 320,
        "harmonics": [640, 960, 1280],
        "harmonic_amplitudes": [0.5, 0.25, 0.1],
        "broadband_amplitude": 0.04,
        "n_motors": 4,
        "n_blades": 2,
        "motor_kv": 3400,
        "hover_rpm_spread": 0.08,
        "pid_jitter_pct": 0.10,
        "session_rpm_var_pct": 0.05,
    },
}

ENVIRONMENT_PROFILES = {
    "open_field": {
        "ambient_db": 35,
        "wind_noise_hz": 20,
        "wind_amplitude": 0.01,
        "gust_rate_per_min": 3,      # avg gusts per minute
        "gust_amplitude_mult": 8.0,  # gust peak = base * mult
        "noise_event_rate_per_min": 2,  # birds, insects
        "noise_event_db_above": 8,
    },
    "suburban": {
        "ambient_db": 55,
        "wind_noise_hz": 30,
        "wind_amplitude": 0.03,
        "gust_rate_per_min": 4,
        "gust_amplitude_mult": 5.0,
        "noise_event_rate_per_min": 8,  # traffic, dogs, people
        "noise_event_db_above": 12,
    },
    "warehouse": {
        "ambient_db": 45,
        "wind_noise_hz": 10,
        "wind_amplitude": 0.005,
        "gust_rate_per_min": 0.5,     # indoor, rare drafts
        "gust_amplitude_mult": 3.0,
        "noise_event_rate_per_min": 1,  # echoes, machinery hum
        "noise_event_db_above": 6,
    },
}


def _bandlimited_noise(n_samples, cutoff_hz, sample_rate, rng, order=2):
    from scipy.signal import butter, sosfilt
    if n_samples < 2 or cutoff_hz <= 0:
        return np.zeros(n_samples)
    nyq = sample_rate / 2.0
    if cutoff_hz >= nyq:
        return rng.standard_normal(n_samples)
    sos = butter(order, cutoff_hz, btype='low', fs=sample_rate, output='sos')
    raw = rng.standard_normal(n_samples)
    filtered = sosfilt(sos, raw)
    std = np.std(filtered)
    if std > 1e-12:
        filtered /= std
    return filtered


def _generate_single_motor_signal(
    fundamental_hz: float,
    harmonics: list[float],
    harmonic_amplitudes: list[float],
    broadband_amplitude: float,
    motor_rpm_offset: float,
    pid_jitter_pct: float,
    n_samples: int,
    sample_rate: int,
    attenuation_curve: np.ndarray,
    battery_drift: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    # PID jitter: bursty, 1-5 Hz, models attitude controller corrections
    pid_noise = _bandlimited_noise(n_samples, 5.0, sample_rate, rng, order=2)
    # Add slower wobble (0.5-2 Hz) from attitude hold
    attitude_wobble = _bandlimited_noise(n_samples, 2.0, sample_rate, rng, order=1)
    jitter = 1.0 + pid_jitter_pct * (0.6 * pid_noise + 0.4 * attitude_wobble)

    # Motor-specific RPM = base * offset * jitter * battery_drift
    motor_fund = fundamental_hz * (1.0 + motor_rpm_offset) * jitter * battery_drift

    # Cumulative phase for fundamental
    fund_phase = np.cumsum(2 * np.pi * motor_fund / sample_rate)

    signal = np.sin(fund_phase) * attenuation_curve

    # Coherent harmonics
    for harm_hz, harm_amp in zip(harmonics, harmonic_amplitudes):
        harm_ratio = harm_hz / fundamental_hz
        signal += np.sin(harm_ratio * fund_phase) * harm_amp * attenuation_curve

    # Broadband motor noise (bearing noise, ESC whine, turbulence)
    signal += rng.standard_normal(n_samples) * broadband_amplitude * attenuation_curve

    return signal


def generate_drone_signal(
    drone_type: str,
    distance_m: float,
    duration_s: float,
    sample_rate: int = 48000,
    channels: int = 4,
    approaching: bool = True,
    rng: np.random.Generator | None = None,
    rpm_session_offset: float = 0.0,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    profile = DRONE_PROFILES[drone_type]
    n_samples = int(sample_rate * duration_s)

    if n_samples == 0:
        return np.zeros((0, channels))

    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    # --- Flight path wander: distance varies +/- 5-10m during pass ---
    wander_amplitude = max(distance_m * 0.08, 2.0)  # 8% of distance or min 2m
    distance_wander = _bandlimited_noise(n_samples, 0.5, sample_rate, rng, order=2)
    distance_curve = np.clip(distance_m + wander_amplitude * distance_wander, 1.0, None)

    # Attenuation: 1/r, time-varying with flight path wander
    attenuation_curve = 1.0 / distance_curve

    # --- Battery voltage sag: slow downward drift over recording ---
    # 6S LiPo: 25.2V -> ~24.5V over 15s segment = ~0.5-1% RPM drop
    battery_drift_pct = rng.uniform(0.003, 0.010)  # 0.3-1% over the recording
    battery_drift = np.linspace(1.0, 1.0 - battery_drift_pct, n_samples)

    # --- 4 independent motors at slightly different RPMs ---
    n_motors = profile.get("n_motors", 4)
    rpm_spread = profile.get("hover_rpm_spread", 0.08)
    pid_jitter = profile.get("pid_jitter_pct", 0.10)

    # Each motor gets a random RPM offset within the spread range
    # Plus session-level shift (battery, temp, prop condition)
    motor_offsets = rng.uniform(-rpm_spread / 2, rpm_spread / 2, n_motors) + rpm_session_offset
    # CW/CCW pairs: motors 0,2 vs 1,3 have correlated offsets (torque balance)
    if n_motors == 4:
        torque_offset = rng.uniform(0.01, rpm_spread / 3)
        motor_offsets[0] += torque_offset
        motor_offsets[2] += torque_offset
        motor_offsets[1] -= torque_offset
        motor_offsets[3] -= torque_offset

    signal = np.zeros(n_samples)
    for m in range(n_motors):
        motor_sig = _generate_single_motor_signal(
            fundamental_hz=profile["fundamental_hz"],
            harmonics=profile["harmonics"],
            harmonic_amplitudes=profile["harmonic_amplitudes"],
            broadband_amplitude=profile["broadband_amplitude"],
            motor_rpm_offset=motor_offsets[m],
            pid_jitter_pct=pid_jitter,
            n_samples=n_samples,
            sample_rate=sample_rate,
            attenuation_curve=attenuation_curve,
            battery_drift=battery_drift,
            rng=rng,
        )
        signal += motor_sig / n_motors  # average contribution per motor

    # --- Doppler shift ---
    if approaching and n_samples > 0:
        speed_ms = 15.0
        doppler_shift = (speed_ms / 343.0) * np.linspace(1, -1, n_samples)
        dt = 1.0 / sample_rate
        tau = np.cumsum(1.0 + doppler_shift) * dt
        original_times = np.arange(n_samples) * dt
        signal = np.interp(original_times, tau, signal)

    # --- Multi-channel with time-of-arrival differences ---
    mic_spacing_m = 0.05
    multichannel = np.zeros((n_samples, channels))
    for ch in range(channels):
        delay_s = (ch * mic_spacing_m / 343.0) * np.sin(np.pi / 4)
        delay_samples = delay_s * sample_rate
        int_delay = int(delay_samples)
        frac_delay = delay_samples - int_delay
        if int_delay == 0 and frac_delay < 1e-6:
            multichannel[:, ch] = signal
        else:
            padded = np.concatenate([np.zeros(int_delay + 1), signal])
            idx = np.arange(n_samples)
            s0 = padded[idx + 1]
            s1 = padded[idx]
            multichannel[:, ch] = (1.0 - frac_delay) * s0 + frac_delay * s1

    return multichannel


def generate_ambient_noise(
    environment: str,
    duration_s: float,
    sample_rate: int = 48000,
    channels: int = 4,
    rng: np.random.Generator | None = None,
    ambient_scale: float = 1.0,
    wind_scale: float = 1.0,
) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    if rng is None:
        rng = np.random.default_rng()
    env = ENVIRONMENT_PROFILES[environment]
    n_samples = int(sample_rate * duration_s)

    if n_samples == 0:
        return np.zeros((0, channels))

    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    # Base amplitude from dB SPL, scaled by session variation
    amplitude = 10 ** (env["ambient_db"] / 20) * 1e-4 * ambient_scale

    # --- Per-channel noise with slightly different characteristics ---
    # Models mic self-noise + preamp noise (each mic is slightly different)
    noise = np.zeros((n_samples, channels))
    for ch in range(channels):
        mic_gain_variation = 1.0 + rng.uniform(-0.05, 0.05)  # +/- 5% gain mismatch
        noise[:, ch] = rng.standard_normal(n_samples) * amplitude * mic_gain_variation

    # --- Mic self-noise floor (~30 dB SPL equivalent) ---
    mic_self_noise_amp = 10 ** (30 / 20) * 1e-4
    for ch in range(channels):
        noise[:, ch] += rng.standard_normal(n_samples) * mic_self_noise_amp * 0.3

    # --- Wind noise: band-limited stochastic + gust events ---
    if n_samples > 1 and env["wind_amplitude"] > 0:
        wind_cutoff = max(env["wind_noise_hz"] * 2, 40.0)
        nyq = sample_rate / 2.0
        if wind_cutoff < nyq:
            sos_wind = butter(3, wind_cutoff, btype='low', fs=sample_rate, output='sos')

            # Base wind (continuous), scaled by session wind conditions
            wind_amp = env["wind_amplitude"] * wind_scale
            for ch in range(channels):
                base_wind = sosfilt(sos_wind, rng.standard_normal(n_samples))
                std_w = np.std(base_wind)
                if std_w > 1e-12:
                    base_wind /= std_w
                noise[:, ch] += base_wind * wind_amp

            # Wind gusts: Poisson-distributed bursts with 2-8 second duration
            gust_rate = env.get("gust_rate_per_min", 3)
            gust_mult = env.get("gust_amplitude_mult", 8.0)
            expected_gusts = gust_rate * duration_s / 60.0
            n_gusts = rng.poisson(max(expected_gusts, 0.1))

            gust_envelope = np.ones(n_samples)
            for _ in range(n_gusts):
                gust_center = rng.uniform(0, duration_s)
                gust_duration = rng.uniform(1.5, 6.0)  # 1.5-6 second gusts
                gust_strength = rng.uniform(2.0, gust_mult)
                # Hann-shaped gust envelope
                gust_start = max(0, int((gust_center - gust_duration / 2) * sample_rate))
                gust_end = min(n_samples, int((gust_center + gust_duration / 2) * sample_rate))
                gust_len = gust_end - gust_start
                if gust_len > 0:
                    hann = np.hanning(gust_len)
                    gust_envelope[gust_start:gust_end] += hann * (gust_strength - 1.0)

            for ch in range(channels):
                gust_noise = sosfilt(sos_wind, rng.standard_normal(n_samples))
                std_g = np.std(gust_noise)
                if std_g > 1e-12:
                    gust_noise /= std_g
                noise[:, ch] += gust_noise * wind_amp * (gust_envelope - 1.0)

    # --- Ambient noise events: transient sounds (birds, traffic, etc.) ---
    event_rate = env.get("noise_event_rate_per_min", 2)
    event_db = env.get("noise_event_db_above", 8)
    expected_events = event_rate * duration_s / 60.0
    n_events = rng.poisson(max(expected_events, 0.1))

    for _ in range(n_events):
        event_center = rng.uniform(0, duration_s)
        event_duration = rng.uniform(0.3, 2.5)  # 0.3-2.5 seconds
        event_amp = amplitude * 10 ** (rng.uniform(3, event_db) / 20)

        evt_start = max(0, int((event_center - event_duration / 2) * sample_rate))
        evt_end = min(n_samples, int((event_center + event_duration / 2) * sample_rate))
        evt_len = evt_end - evt_start
        if evt_len > 10:
            # Random spectral shape: narrowband (bird) or broadband (traffic)
            is_narrowband = rng.random() < 0.4
            if is_narrowband:
                # Bird/insect: tone at 1-5 kHz with some modulation
                freq = rng.uniform(1000, 5000)
                evt_t = np.arange(evt_len) / sample_rate
                event_sig = np.sin(2 * np.pi * freq * evt_t) * event_amp
                # AM modulation for realism
                event_sig *= 1.0 + 0.3 * np.sin(2 * np.pi * rng.uniform(3, 15) * evt_t)
            else:
                # Broadband: traffic rumble, footstep, etc.
                event_sig = rng.standard_normal(evt_len) * event_amp

            # Apply fade in/out envelope
            env_window = np.hanning(evt_len)
            event_sig *= env_window

            # Apply to random subset of channels (not all events hit all mics equally)
            active_channels = rng.choice(channels, size=max(1, channels - rng.integers(0, 2)),
                                         replace=False)
            for ch in active_channels:
                ch_gain = rng.uniform(0.5, 1.0)  # different level per mic
                noise[evt_start:evt_end, ch] += event_sig * ch_gain

    return noise


def generate_session_conditions(rng: np.random.Generator, environment: str) -> dict:
    env = ENVIRONMENT_PROFILES[environment]
    base_db = env["ambient_db"]
    return {
        "ambient_db_offset": rng.normal(0, 4.0),    # +/- 4 dB session variation
        "wind_mult": rng.lognormal(0, 0.5),          # log-normal wind intensity
        "propagation_scatter_db": rng.normal(0, 1.5), # ground effect, refraction
        "temp_c": rng.uniform(5, 30),
        "wind_beaufort": max(0, min(6, int(rng.normal(2, 1.2)))),
        "battery_start_pct": rng.uniform(70, 100),
    }


def generate_mock_recording(
    drone_type: str = "fpv_5inch",
    environment: str = "open_field",
    distance_m: float = 75.0,
    duration_s: float = 10.0,
    sample_rate: int = 48000,
    channels: int = 4,
    approaching: bool = True,
    seed: int | None = None,
    session: dict | None = None,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)

    if session is None:
        session = generate_session_conditions(rng, environment)

    # Session-level RPM shift (battery state, temperature, prop wear)
    profile = DRONE_PROFILES[drone_type]
    session_rpm_var = profile.get("session_rpm_var_pct", 0.08)
    rpm_session_offset = rng.normal(0, session_rpm_var)

    drone = generate_drone_signal(
        drone_type, distance_m, duration_s, sample_rate, channels, approaching, rng,
        rpm_session_offset=rpm_session_offset,
    )

    # Apply propagation scatter: log-normal deviation from ideal 1/r
    prop_scatter = 10 ** (session["propagation_scatter_db"] / 20)
    drone *= prop_scatter

    # Session ambient level variation
    ambient_scale = 10 ** (session["ambient_db_offset"] / 20)
    wind_scale = session["wind_mult"]

    ambient = generate_ambient_noise(
        environment, duration_s, sample_rate, channels, rng,
        ambient_scale=ambient_scale, wind_scale=wind_scale,
    )

    combined = drone + ambient
    peak = np.max(np.abs(combined))
    if peak > 0.95:
        combined = combined * (0.95 / peak)
    return combined, session
