"""
drone_roll_sim.py — "What happens when my drone tilts in the wind?"







A plain-English flight simulator for one thing: tilting the drone sideways (roll).
You answer a few simple questions about your drone and where you're flying.
The program then:
  1. Looks up the real wind where you are (or lets you type it in).
  2. Simulates the drone tilting to the angle you ask for, then getting hit by a gust.
  3. Shows graphs and prints a plain-English report: how wobbly it was, how far the
     gust pushed it, and which controller settings would fly best in that wind.

Run:      python drone_roll_sim.py
Options:  --defaults   skip the questions and use typical values
          --no-show    save the graphs without opening windows
Needs:    pip install numpy matplotlib

Files it saves (in the folder you run it from):
  flight_summary.txt   the plain-English report
  roll_sim.png         graphs of this flight
  gain_sweep.png       which controller settings handle this wind best
  roll_sim.csv         second-by-second flight data (100 rows per second)
  gain_sweep.csv       results for every controller setting tried

------------------------------------------------------------------------------
Behind the scenes (for anyone who wants the maths) — the 4 equations:
    Eq.1  e(t)      = theta_ref(t) - theta(t)                    error
    Eq.2  u(t)      = Kp*e + Ki*integral(e) + Kd*de/dt           PID controller (the brain)
    Eq.3  dtau_m/dt = (u - tau_m) / Tm                           motor lag (the muscle)
    Eq.4  J*theta'' = tau_m + d(t)                               drone body (the plant)
------------------------------------------------------------------------------
"""

from dataclasses import dataclass, replace
import argparse
import csv
import json
import math
import urllib.parse
import urllib.request

import numpy as np
import matplotlib.pyplot as plt

MPH_PER_MS = 2.23694


# ---------------------------------------------------------------------------
# 1. SETTINGS — internal values, in science units. Users never type these
#    directly unless they pick "Advanced"; the questions below fill them in.
# ---------------------------------------------------------------------------
@dataclass
class Params:
    # Controller ("autopilot") settings
    Kp: float = 0.5        # how hard it pushes back toward the target     N·m/rad
    Ki: float = 0.3        # how it fixes small leftover errors over time  N·m/(rad·s)
    Kd: float = 0.08       # how much it brakes to avoid overshooting       N·m·s/rad
    preset_name: str = "Balanced"
    # Drone
    weight_g: float = 650.0          # grams
    frame_mm: float = 250.0          # motor-to-motor diagonal, mm
    height_cm: float = 5.0           # body height, cm
    J: float = 0.005                 # resistance to tilting (from weight + size)   kg·m²
    Tm: float = 0.03                 # how long motors take to respond               s
    u_max: float = 0.5               # strongest twist the motors can make           N·m
    side_area: float = 0.0125        # side area the wind hits                       m²
    cp_offset: float = 0.03          # how off-centre the wind pushes                m
    Cd: float = 1.0                  # how "blocky" the drone is to the wind
    rho: float = 1.225               # air density                                    kg/m³
    # Manoeuvre
    step_time: float = 0.5           # s, when you push the stick
    step_angle_deg: float = 20.0     # how far you tilt, degrees
    heading_deg: float = 0.0         # which way the drone faces, degrees (0 = north)
    # Wind
    wind_speed: float = 5.0          # m/s steady wind
    wind_gust: float = 12.0          # m/s gust peak
    wind_dir_deg: float = 270.0      # where the wind comes FROM, degrees (270 = west)
    wind_source: str = "typical values"
    gust_time: float = 2.0           # s
    gust_duration: float = 1.0       # s
    # Simulation
    dt: float = 0.001                # s, 1000 calculations per simulated second
    t_end: float = 4.0               # s
    d_on_measurement: bool = True    # avoids a jolt when you move the stick
    anti_windup: bool = True         # avoids a big overshoot after the motors max out
    gyro_noise_std: float = 0.0      # rad, sensor jitter (0 = perfect sensor)


# Ready-made controller settings, so nobody has to pick Kp/Ki/Kd by hand
PRESETS = {
    1: ("Smooth",         0.4, 0.2, 0.06, "gentle, relaxed feel; good for filming"),
    2: ("Balanced",       0.5, 0.3, 0.08, "good all-round choice (recommended)"),
    3: ("Locked-in",      0.8, 0.4, 0.10, "snappy and holds steady in wind; works the motors harder"),
    4: ("Original notes", 0.3, 0.6, 0.03, "the values from the class notes; wobbly"),
}
MOTOR_SPEED = {1: ("Fast (small racing motors)", 0.02),
               2: ("Normal (most hobby drones)", 0.03),
               3: ("Slow (big props, camera drones)", 0.06)}


# ---------------------------------------------------------------------------
# 2. ASKING QUESTIONS
# ---------------------------------------------------------------------------
def ask(question: str, default, cast=float, help_text: str = ""):
    """Ask one question. Enter keeps the default. Bad input asks again."""
    if help_text:
        print(f"    ({help_text})")
    while True:
        raw = input(f"  {question} [{default}]: ").strip()
        try:
            return cast(default) if raw == "" else cast(raw)
        except ValueError:
            print("    Sorry, I didn't understand that. Please try again.")


def ask_choice(question: str, options: dict, default: int) -> int:
    print(f"\n  {question}")
    for k, v in options.items():
        label = v[0] if isinstance(v, tuple) else v
        extra = f" — {v[4]}" if isinstance(v, tuple) and len(v) > 4 else ""
        print(f"    {k}. {label}{extra}")
    while True:
        c = ask("Your choice", default, int)
        if c in options:
            return c
        print("    Please pick one of the numbers listed.")


COMPASS = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}


def to_degrees(text) -> float:
    """Accepts 'N', 'sw', 'west' or a number like 270."""
    if isinstance(text, (int, float)):
        return float(text)
    t = str(text).strip().upper()
    words = {"NORTH": "N", "EAST": "E", "SOUTH": "S", "WEST": "W"}
    t = words.get(t, t)
    if t in COMPASS:
        return float(COMPASS[t])
    return float(t) % 360


def compass_name(deg: float) -> str:
    names = ["north", "north-east", "east", "south-east",
             "south", "south-west", "west", "north-west"]
    return names[int(((deg % 360) + 22.5) // 45) % 8]


def estimate_drone(p: Params) -> Params:
    """Turn weight and size into the physics numbers (rough but reasonable)."""
    arm = (p.frame_mm / 1000) / 2                    # centre to motor, m
    p.J = 0.5 * (p.weight_g / 1000) * arm ** 2       # resistance to tilting
    p.side_area = (p.frame_mm / 1000) * (p.height_cm / 100)
    p.u_max = 0.6 * (p.weight_g / 1000) * 9.81 * arm # ~ what the motors can twist
    return p


def ask_everything(p: Params) -> Params:
    print("\n=== Drone tilt simulator ===")
    print("Answer a few questions. Just press Enter to accept the suggestion in [brackets].")

    print("\n--- About your drone ---")
    p.weight_g = ask("How much does your drone weigh, in grams?", p.weight_g,
                     help_text="a small racing drone is ~500 g, a DJI Mini ~250 g, a camera drone ~900 g")
    p.frame_mm = ask("How wide is it, motor to opposite motor, in millimetres?", p.frame_mm,
                     help_text="measure diagonally across; a '5-inch' racing drone is ~220–250 mm")
    p.height_cm = ask("How tall is the body, in centimetres?", p.height_cm)
    m = ask_choice("How quick are its motors?", MOTOR_SPEED, 2)
    p.Tm = MOTOR_SPEED[m][1]
    p = estimate_drone(p)

    print("\n--- Autopilot style ---")
    c = ask_choice("How should the autopilot fly?",
                   {**PRESETS, 5: ("Advanced", 0, 0, 0, "type in my own Kp / Ki / Kd")}, 2)
    if c == 5:
        p.preset_name = "Custom"
        p.Kp = ask("Kp — push-back strength", p.Kp, help_text="bigger = reacts harder to being off target")
        p.Ki = ask("Ki — slow correction", p.Ki, help_text="bigger = fixes leftover lean faster, but can overshoot")
        p.Kd = ask("Kd — braking", p.Kd, help_text="bigger = less wobble, but slower and twitchier with noisy sensors")
    else:
        p.preset_name, p.Kp, p.Ki, p.Kd, _ = PRESETS[c]

    print("\n--- The manoeuvre ---")
    p.step_angle_deg = ask("How far should the drone tilt, in degrees?", p.step_angle_deg,
                           help_text="10° is gentle, 20° is normal, 40° is aggressive")
    p.heading_deg = ask("Which way is the drone facing? (N, NE, E, ... or degrees)", "N", to_degrees)
    return p


# ---------------------------------------------------------------------------
# 3. WIND — live lookup or typed in
# ---------------------------------------------------------------------------
def get_json(url: str, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "drone-roll-sim/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def geocode(place: str) -> tuple[float, float, str]:
    """City name (or 'lat,lon') → coordinates, using Open-Meteo's free lookup."""
    if "," in place:
        try:
            lat, lon = (float(x) for x in place.split(",", 1))
            return lat, lon, f"{lat:.3f}, {lon:.3f}"
        except ValueError:
            pass
    q = urllib.parse.urlencode({"name": place, "count": 1})
    data = get_json(f"https://geocoding-api.open-meteo.com/v1/search?{q}")
    if not data.get("results"):
        raise ValueError(f"couldn't find a place called '{place}'")
    r = data["results"][0]
    label = ", ".join(x for x in [r.get("name"), r.get("admin1"), r.get("country")] if x)
    return r["latitude"], r["longitude"], label


def wind_open_meteo(lat: float, lon: float) -> tuple[float, float, float]:
    """Current wind from Open-Meteo (free, no key). Returns m/s, m/s, degrees."""
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "wind_speed_unit": "ms",
        "current": "wind_speed_10m,wind_gusts_10m,wind_direction_10m",
    })
    c = get_json(f"https://api.open-meteo.com/v1/forecast?{q}")["current"]
    return c["wind_speed_10m"], c["wind_gusts_10m"], c["wind_direction_10m"]


def wind_weather_com(lat: float, lon: float, api_key: str) -> tuple[float, float, float]:
    """Current wind from The Weather Company (weather.com). Needs your own API key.
    If your key's documentation shows a different address or field names, change them here."""
    q = urllib.parse.urlencode({"geocode": f"{lat},{lon}", "units": "m", "language": "en-US",
                                "format": "json", "apiKey": api_key})
    o = get_json(f"https://api.weather.com/v3/wx/observations/current?{q}")
    speed = (o.get("windSpeed") or 0) / 3.6                       # km/h → m/s
    gust = (o.get("windGust") or o.get("windSpeed") or 0) / 3.6   # no gust reported when calm
    return speed, gust, o.get("windDirection") or 0.0


def wind_nws(lat: float, lon: float, contact: str) -> tuple[float, float, float, str]:
    """Latest observed wind from the US National Weather Service (api.weather.gov).
    Free, no key. NWS asks every program to send a User-Agent that names the app
    and gives a contact email, e.g. "(drone-roll-sim, you@example.com)".
    US locations only. Returns m/s, m/s, degrees, station name."""
    ua = f"(drone-roll-sim, {contact})" if contact else "(drone-roll-sim)"

    def nws(url):
        req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": "application/geo+json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())

    # 1) location → list of nearby weather stations, 2) nearest station → latest observation
    point = nws(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}")
    stations = nws(point["properties"]["observationStations"])
    for feature in stations["features"][:5]:          # try the 5 nearest stations
        st = feature["properties"]
        obs = nws(f"https://api.weather.gov/stations/{st['stationIdentifier']}/observations/latest")
        o = obs["properties"]

        def to_ms(field):
            f = o.get(field) or {}
            v = f.get("value")
            if v is None:
                return None
            return v / 3.6 if "km_h" in f.get("unitCode", "") else v   # km/h → m/s

        speed = to_ms("windSpeed")
        if speed is None:                              # station not reporting wind; try next
            continue
        gust = to_ms("windGust") or speed              # no gust reported when it isn't gusty
        direction = (o.get("windDirection") or {}).get("value") or 0.0
        return speed, gust, direction, st.get("name", st["stationIdentifier"])
    raise ValueError("no nearby NWS station is reporting wind right now")


def ask_wind(p: Params) -> Params:
    print("\n--- Wind ---")
    src = ask_choice("Where should the wind come from?", {
        1: "Look up the wind right now (free, no account needed)",
        2: "Look it up on weather.com (needs a weather.com API key)",
        3: "Look it up from the National Weather Service (free, US only)",
        4: "I'll type the wind speed in myself",
    }, 1)
    if src in (1, 2, 3):
        try:
            place = ask("Where are you flying? (city name)", "Phoenix", str)
            lat, lon, label = geocode(place)
            if src == 1:
                p.wind_speed, p.wind_gust, p.wind_dir_deg = wind_open_meteo(lat, lon)
                p.wind_source = f"live wind for {label} (Open-Meteo)"
            elif src == 3:
                email = ask("Your email (the weather service asks for it; Enter to skip)", "", str)
                p.wind_speed, p.wind_gust, p.wind_dir_deg, station = wind_nws(lat, lon, email)
                p.wind_source = f"National Weather Service, {station}"
            else:
                key = ask("Paste your weather.com API key", "", str)
                p.wind_speed, p.wind_gust, p.wind_dir_deg = wind_weather_com(lat, lon, key)
                p.wind_source = f"live wind for {label} (weather.com)"
            print(f"  Right now in {label}: wind {p.wind_speed * MPH_PER_MS:.0f} mph, "
                  f"gusts up to {p.wind_gust * MPH_PER_MS:.0f} mph, "
                  f"coming from the {compass_name(p.wind_dir_deg)}.")
            return p
        except Exception as exc:
            print(f"  Couldn't get the live wind ({exc}).")
            print("  No problem — type it in instead.")
    mph = ask("Normal wind speed, in mph", round(p.wind_speed * MPH_PER_MS))
    gust_mph = ask("Strongest gust, in mph", round(p.wind_gust * MPH_PER_MS))
    p.wind_speed, p.wind_gust = mph / MPH_PER_MS, gust_mph / MPH_PER_MS
    p.wind_dir_deg = ask("Wind is coming FROM which direction? (N, NE, E, ... or degrees)", "W", to_degrees)
    p.wind_source = "wind you typed in"
    return p


# ---------------------------------------------------------------------------
# 4. WHAT THE PILOT ASKS FOR, AND WHAT THE WIND DOES
# ---------------------------------------------------------------------------
def theta_ref(t: float, p: Params) -> float:
    """Target tilt (radians): level, then your chosen angle at step_time."""
    return math.radians(p.step_angle_deg) if t >= p.step_time else 0.0


def wind_torque(v: float, p: Params) -> float:
    """Twist (N·m) from wind speed v (m/s). Only wind hitting the side tilts it."""
    v_side = v * math.sin(math.radians(p.wind_dir_deg - p.heading_deg))
    force = 0.5 * p.rho * p.Cd * p.side_area * v_side * abs(v_side)
    return force * p.cp_offset


def disturbance(t: float, p: Params) -> float:
    """Steady wind all the time, plus the stronger gust while it lasts."""
    in_gust = p.gust_time <= t < p.gust_time + p.gust_duration
    return wind_torque(p.wind_gust if in_gust else p.wind_speed, p)


# ---------------------------------------------------------------------------
# 5. THE AUTOPILOT (PID controller) — Eq.1 + Eq.2
# ---------------------------------------------------------------------------
class PID:
    def __init__(self, p: Params):
        self.p, self.integral, self.prev_error, self.prev_meas = p, 0.0, None, None

    def update(self, ref: float, meas: float, dt: float) -> tuple[float, dict]:
        p = self.p
        error = ref - meas
        if p.d_on_measurement:
            deriv = 0.0 if self.prev_meas is None else -(meas - self.prev_meas) / dt
        else:
            deriv = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        new_integral = self.integral + error * dt
        u_unsat = p.Kp * error + p.Ki * new_integral + p.Kd * deriv
        u = max(-p.u_max, min(p.u_max, u_unsat))
        if not p.anti_windup or u == u_unsat:
            self.integral = new_integral
        self.prev_error, self.prev_meas = error, meas
        return u, {"P": p.Kp * error, "I": p.Ki * self.integral, "D": p.Kd * deriv}


# ---------------------------------------------------------------------------
# 6. THE PHYSICS — Eq.3 + Eq.4, one millisecond at a time
# ---------------------------------------------------------------------------
def simulate(p: Params) -> dict[str, np.ndarray]:
    n = int(round(p.t_end / p.dt)) + 1
    t = np.linspace(0.0, p.t_end, n)
    theta = omega = tau_m = 0.0
    pid = PID(p)
    rng = np.random.default_rng(0)
    cols = ["t", "theta_ref", "theta", "omega", "error", "u", "tau_m", "d", "P", "I", "D"]
    log = {c: np.zeros(n) for c in cols}

    for k in range(n):
        ref, d = theta_ref(t[k], p), disturbance(t[k], p)
        meas = theta + (rng.normal(0.0, p.gyro_noise_std) if p.gyro_noise_std else 0.0)
        u, terms = pid.update(ref, meas, p.dt)
        alpha = (tau_m + d) / p.J
        for c, v in zip(cols, [t[k], ref, theta, omega, ref - theta, u, tau_m, d,
                               terms["P"], terms["I"], terms["D"]]):
            log[c][k] = v
        tau_m += (u - tau_m) / p.Tm * p.dt
        omega += alpha * p.dt
        theta += omega * p.dt
    return log


# ---------------------------------------------------------------------------
# 7. SCORING THE FLIGHT
# ---------------------------------------------------------------------------
def metrics(log: dict, p: Params) -> dict:
    t, th = log["t"], log["theta"]
    target = math.radians(p.step_angle_deg)
    w = (t >= p.step_time) & (t < p.gust_time)
    tw, thw = t[w], th[w]
    overshoot = max(0.0, (thw.max() - target) / target * 100) if target else 0.0

    def first_time(cond):
        return float(tw[np.argmax(cond)] - p.step_time) if cond.any() else float("nan")

    rise = first_time(thw >= 0.9 * target) - first_time(thw >= 0.1 * target)
    outside = np.abs(thw - target) > 0.05 * abs(target)
    if not outside.any():
        settle = 0.0
    elif outside[-1]:
        settle = float("nan")
    else:
        settle = float(tw[np.where(outside)[0][-1] + 1] - p.step_time)
    g = t >= p.gust_time
    return {
        "Kp": p.Kp, "Ki": p.Ki, "Kd": p.Kd,
        "peak_tilt_deg": round(math.degrees(float(thw.max())), 2),
        "overshoot_pct": round(float(overshoot), 2),
        "rise_time_s": round(rise, 4),
        "settling_time_s": round(settle, 4),
        "max_gust_push_deg": round(math.degrees(float(np.max(np.abs(th[g] - target)))), 3) if g.any() else 0.0,
        "motor_effort_pct": round(100 * float(np.max(np.abs(log["tau_m"]))) / p.u_max, 1),
        "motors_maxed_out": bool(np.any(np.abs(log["u"]) >= p.u_max - 1e-9)),
    }


def rate_wobble(over: float) -> str:
    if over < 10: return "very smooth"
    if over < 25: return "a little bouncy"
    if over < 50: return "noticeably wobbly"
    return "very wobbly"


def rate_gust(deg: float) -> str:
    if deg < 3: return "barely moved"
    if deg < 8: return "pushed a bit"
    if deg < 15: return "pushed a lot"
    return "thrown well off course"


def best_settings(rows: list[dict]) -> dict:
    """Lowest combined score of wobble + gust push, ignoring runs that max out the motors."""
    ok = [r for r in rows if not r["motors_maxed_out"] and not math.isnan(r["settling_time_s"])]
    return min(ok or rows, key=lambda r: r["overshoot_pct"] + 3 * r["max_gust_push_deg"])


# ---------------------------------------------------------------------------
# 8. THE PLAIN-ENGLISH REPORT
# ---------------------------------------------------------------------------
def summary_text(p: Params, m: dict, best: dict) -> str:
    mph = lambda v: v * MPH_PER_MS
    side = abs(math.sin(math.radians(p.wind_dir_deg - p.heading_deg)))
    side_word = ("straight from the side" if side > 0.9 else
                 "partly from the side" if side > 0.3 else "mostly head-on or from behind")
    settle = ("did not settle before the gust hit" if math.isnan(m["settling_time_s"])
              else f"settled within {m['settling_time_s']:.1f} seconds")

    lines = [
        "FLIGHT SUMMARY",
        "=" * 60,
        "",
        "CONDITIONS",
        f"  Wind: {mph(p.wind_speed):.0f} mph, gusting to {mph(p.wind_gust):.0f} mph, "
        f"coming from the {compass_name(p.wind_dir_deg)} ({p.wind_source}).",
        f"  Your drone faces {compass_name(p.heading_deg)}, so the wind hits it {side_word}.",
        "",
        "YOUR DRONE",
        f"  {p.weight_g:.0f} g, {p.frame_mm:.0f} mm across, {p.height_cm:.0f} cm tall.",
        f"  Autopilot style: {p.preset_name} (Kp {p.Kp}, Ki {p.Ki}, Kd {p.Kd}).",
        "",
        "WHAT HAPPENED",
        f"  You asked for a {p.step_angle_deg:.0f}° tilt. The drone swung to {m['peak_tilt_deg']:.0f}° "
        f"before coming back ({m['overshoot_pct']:.0f}% too far) — {rate_wobble(m['overshoot_pct'])}.",
        f"  It got most of the way there in {m['rise_time_s']:.2f} seconds and {settle}.",
        (f"  When the {mph(p.wind_gust):.0f} mph gust hit, it was pushed up to "
         f"{m['max_gust_push_deg']:.1f}° off target — {rate_gust(m['max_gust_push_deg'])}."
         if p.wind_gust > p.wind_speed * 1.05 else
         f"  No gusts were reported, so the wind stayed steady. In the last part of the flight it "
         f"stayed within {m['max_gust_push_deg']:.1f}° of target — {rate_gust(m['max_gust_push_deg'])}."),
        f"  Motors used up to {m['motor_effort_pct']:.0f}% of their strength"
        + (" and MAXED OUT at some point — the drone may lose control in this wind."
           if m["motors_maxed_out"] else "."),
        "",
        "RECOMMENDATION",
    ]
    same = (best["Kp"], best["Kd"]) == (p.Kp, p.Kd)
    if same:
        lines.append("  Your current settings were the best of the ones tested for this wind.")
    else:
        lines += [
            f"  For this wind, the steadiest settings tested were Kp {best['Kp']}, Ki {best['Ki']}, Kd {best['Kd']}:",
            f"  about {best['overshoot_pct']:.0f}% overshoot (vs {m['overshoot_pct']:.0f}% now) and "
            f"{best['max_gust_push_deg']:.1f}° gust push (vs {m['max_gust_push_deg']:.1f}° now).",
        ]
    if mph(p.wind_gust) > 25:
        lines.append("  Gusts over 25 mph are rough for small drones — consider waiting for calmer air.")
    lines += [
        "",
        "IMPORTANT: this is a simplified computer model of one tilt direction. Use it to compare",
        "settings, then test changes carefully, low and slow, on the real drone.",
        "",
        "FILES",
        "  roll_sim.png      graphs of this flight",
        "  gain_sweep.png    how other autopilot settings would handle this wind",
        "  roll_sim.csv      the flight data, 100 rows per second",
        "  gain_sweep.csv    scores for every setting tried",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. FILES AND GRAPHS
# ---------------------------------------------------------------------------
def save_timeseries_csv(log: dict, path: str, every: int = 10):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "target_tilt_deg", "actual_tilt_deg", "tilt_speed_deg_per_s",
                    "off_target_deg", "autopilot_request_Nm", "motor_twist_Nm", "wind_twist_Nm",
                    "P_part_Nm", "I_part_Nm", "D_part_Nm"])
        for k in range(0, len(log["t"]), every):
            w.writerow([round(log["t"][k], 3)] +
                       [round(math.degrees(log[c][k]), 3) for c in ("theta_ref", "theta", "omega", "error")] +
                       [round(log[c][k], 5) for c in ("u", "tau_m", "d", "P", "I", "D")])


def save_rows_csv(rows: list[dict], path: str):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_run(log: dict, p: Params, m: dict, path: str):
    t = log["t"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    gust = (p.gust_time, min(p.gust_time + p.gust_duration, p.t_end))
    for a in ax:
        a.axvspan(*gust, color="tab:gray", alpha=0.12, lw=0)
        a.grid(alpha=0.3)

    ax[0].plot(t, np.degrees(log["theta_ref"]), "--", color="k", lw=1, label="Tilt you asked for")
    ax[0].plot(t, np.degrees(log["theta"]), color="tab:blue", lw=2, label="What the drone did")
    ax[0].set_ylabel("Tilt (degrees)")
    ax[0].set_title(f"Tilt: {rate_wobble(m['overshoot_pct'])} ({m['overshoot_pct']:.0f}% overshoot)  ·  "
                    f"gust: {rate_gust(m['max_gust_push_deg'])} ({m['max_gust_push_deg']:.1f}°)",
                    fontsize=10, loc="left")
    ax[0].legend(loc="lower right")

    ax[1].plot(t, log["u"], color="tab:blue", label="Twist the autopilot asks for")
    ax[1].plot(t, log["tau_m"], color="tab:orange", label="Twist the motors deliver")
    ax[1].plot(t, log["d"], color="tab:green", label="Twist from the wind")
    ax[1].set_ylabel("Twist (N·m)")
    ax[1].legend(loc="upper right", fontsize=9)

    ax[2].plot(t, log["P"], label="Push-back (P)")
    ax[2].plot(t, log["I"], label="Slow correction (I)")
    ax[2].plot(t, log["D"], label="Braking (D)")
    ax[2].set_ylabel("Autopilot parts (N·m)")
    ax[2].set_xlabel("Time (seconds)   ·   grey band = wind gust")
    ax[2].legend(loc="upper right", fontsize=9)

    fig.suptitle(f"Drone tilt in {p.wind_speed * MPH_PER_MS:.0f} mph wind, gusts "
                 f"{p.wind_gust * MPH_PER_MS:.0f} mph  —  autopilot: {p.preset_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def plot_sweep(rows: list[dict], Kps: list, Kds: list, best: dict, path: str):
    grid = lambda key: np.array([[r[key] for r in rows if r["Kp"] == kp] for kp in Kps])
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for a, data, title in [(ax[0], grid("overshoot_pct"), "Wobble: % past the target"),
                           (ax[1], grid("max_gust_push_deg"), "Gust push: degrees off target")]:
        im = a.imshow(data, cmap="viridis_r", aspect="auto")
        a.set_xticks(range(len(Kds)), [str(k) for k in Kds])
        a.set_yticks(range(len(Kps)), [str(k) for k in Kps])
        a.set_xlabel("Braking (Kd)  →  more")
        a.set_ylabel("Push-back (Kp)  →  more")
        a.set_title(f"{title}\n(lower is better)", fontsize=10)
        for i in range(len(Kps)):
            for j in range(len(Kds)):
                a.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=9,
                       color="white" if im.norm(data[i, j]) > 0.5 else "black")
        bi, bj = Kps.index(best["Kp"]), Kds.index(best["Kd"])
        a.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec="red", lw=2.5))
    fig.suptitle(f"Which autopilot settings handle this wind best?  Red box = best "
                 f"(Kp {best['Kp']}, Kd {best['Kd']}, Ki {best['Ki']})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


# ---------------------------------------------------------------------------
# 10. MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Drone tilt simulator")
    ap.add_argument("--defaults", action="store_true", help="skip the questions")
    ap.add_argument("--no-show", action="store_true", help="don't open graph windows")
    args = ap.parse_args()

    p = estimate_drone(Params())
    if not args.defaults:
        p = ask_everything(p)
        p = ask_wind(p)

    print("\nSimulating...")
    log = simulate(p)
    m = metrics(log, p)

    Kps, Kds = [0.2, 0.3, 0.5, 0.8], [0.02, 0.03, 0.05, 0.08, 0.1]
    rows = [metrics(simulate(q), q)
            for q in (replace(p, Kp=kp, Kd=kd) for kp in Kps for kd in Kds)]
    best = best_settings(rows)

    report = summary_text(p, m, best)
    print("\n" + report)
    with open("flight_summary.txt", "w") as f:
        f.write(report + "\n")
    save_timeseries_csv(log, "roll_sim.csv")
    save_rows_csv(rows, "gain_sweep.csv")
    plot_run(log, p, m, "roll_sim.png")
    plot_sweep(rows, Kps, Kds, best, "gain_sweep.png")

    if not args.no_show:
        print("\nOpening graphs... (close the windows to finish)")
        plt.show()


if __name__ == "__main__":
    main()