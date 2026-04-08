"""
UAV PID Altitude Control Simulation
====================================
Task 2 - Maincrafts Technology UAV Design & Simulation Internship
Simulates a drone reaching and stabilizing at a target height using PID control.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ─────────────────────────────────────────────
# Physical Constants
# ─────────────────────────────────────────────
MASS    = 1.5       # kg  – drone mass
GRAVITY = 9.81      # m/s²
DT      = 0.05      # s   – simulation timestep (smaller = smoother)
T_END   = 20.0      # s   – total simulation time

TARGET_HEIGHT = 10.0  # m

# ─────────────────────────────────────────────
# PID Tuning Sets (for comparison)
# ─────────────────────────────────────────────
tuning_configs = {
    "High Kp (Oscillation)":      {"Kp": 8.0,  "Ki": 0.1, "Kd": 0.5},
    "Low Kp (Slow Response)":     {"Kp": 0.5,  "Ki": 0.1, "Kd": 0.5},
    "High Ki (Overshoot)":        {"Kp": 2.0,  "Ki": 3.0, "Kd": 0.5},
    "Optimal (Stable Hover)":     {"Kp": 2.0,  "Ki": 0.1, "Kd": 1.0},
}


# ─────────────────────────────────────────────
# PID Controller Class
# ─────────────────────────────────────────────
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral   = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint, measured, dt):
        error            = setpoint - measured
        self.integral   += error * dt
        derivative       = (error - self.prev_error) / dt
        output           = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error  = error
        return output, error


# ─────────────────────────────────────────────
# Flight Dynamics Simulator
# ─────────────────────────────────────────────
def simulate(Kp, Ki, Kd, add_wind=False, seed=42):
    rng      = np.random.default_rng(seed)
    time     = np.arange(0, T_END, DT)
    pid      = PIDController(Kp, Ki, Kd)

    height   = 0.0
    velocity = 0.0

    heights      = []
    errors       = []
    thrusts      = []
    velocities   = []

    for t in time:
        pid_output, error = pid.compute(TARGET_HEIGHT, height, DT)

        # Thrust = PID correction + gravity compensation
        thrust = pid_output + MASS * GRAVITY

        # Wind disturbance (optional)
        wind = rng.uniform(-2.0, 2.0) if add_wind else 0.0

        # Newton's second law: F = ma  →  a = (F - mg) / m
        acceleration  = (thrust - MASS * GRAVITY) / MASS + wind / MASS
        velocity     += acceleration * DT
        height       += velocity * DT
        height        = max(height, 0.0)   # ground constraint

        heights.append(height)
        errors.append(error)
        thrusts.append(thrust)
        velocities.append(velocity)

    return time, np.array(heights), np.array(errors), np.array(thrusts), np.array(velocities)


# ─────────────────────────────────────────────
# Plot 1 – Tuning Comparison
# ─────────────────────────────────────────────
def plot_tuning_comparison(output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("PID Tuning Comparison – UAV Altitude Control", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    colors = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"]

    for ax, (label, cfg), color in zip(axes, tuning_configs.items(), colors):
        time, heights, errors, thrusts, _ = simulate(**cfg)
        ax.plot(time, heights,                  color=color,       lw=2,   label="Drone Height")
        ax.axhline(TARGET_HEIGHT, linestyle="--", color="#2c3e50", lw=1.5, label=f"Target {TARGET_HEIGHT} m")
        ax.fill_between(time, heights, TARGET_HEIGHT, alpha=0.10, color=color)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Height (m)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, T_END)

        # Annotate Kp/Ki/Kd
        txt = f"Kp={cfg['Kp']}  Ki={cfg['Ki']}  Kd={cfg['Kd']}"
        ax.annotate(txt, xy=(0.03, 0.04), xycoords="axes fraction", fontsize=8, color="#555")

    plt.tight_layout()
    path = os.path.join(output_dir, "pid_tuning_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────
# Plot 2 – Optimal Hover Deep-Dive
# ─────────────────────────────────────────────
def plot_optimal_hover(output_dir):
    cfg = tuning_configs["Optimal (Stable Hover)"]
    time, heights, errors, thrusts, velocities = simulate(**cfg)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Optimal PID – Stable Hover Analysis", fontsize=16, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Altitude ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, heights,  color="#2ecc71", lw=2.5, label="Drone Height")
    ax1.axhline(TARGET_HEIGHT, linestyle="--", color="#e74c3c", lw=1.5, label="Target 10 m")
    ax1.fill_between(time, heights, TARGET_HEIGHT, alpha=0.1, color="#2ecc71")
    ax1.set_title("Altitude vs Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Height (m)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Error ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, errors, color="#e67e22", lw=2, label="Error (m)")
    ax2.axhline(0, linestyle="--", color="black", lw=1)
    ax2.set_title("Tracking Error vs Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error (m)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── Thrust ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time, thrusts, color="#9b59b6", lw=2, label="Motor Thrust (N)")
    ax3.axhline(MASS * GRAVITY, linestyle="--", color="black", lw=1, label="Hover Thrust")
    ax3.set_title("PID Thrust Output vs Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Thrust (N)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "optimal_hover_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────
# Plot 3 – Wind Disturbance Rejection
# ─────────────────────────────────────────────
def plot_wind_disturbance(output_dir):
    cfg = tuning_configs["Optimal (Stable Hover)"]
    time, h_clean, _, _, _  = simulate(**cfg, add_wind=False)
    time, h_wind,  e, t, _  = simulate(**cfg, add_wind=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Wind Disturbance Rejection – PID vs No Disturbance", fontsize=14, fontweight="bold")

    axes[0].plot(time, h_clean, color="#2ecc71",  lw=2, label="No Wind")
    axes[0].plot(time, h_wind,  color="#3498db", lw=2, label="With Wind", alpha=0.85)
    axes[0].axhline(TARGET_HEIGHT, linestyle="--", color="#e74c3c", lw=1.5, label="Target")
    axes[0].set_title("Altitude: Wind vs No Wind")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Height (m)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, e, color="#e74c3c", lw=2)
    axes[1].axhline(0, linestyle="--", color="black", lw=1)
    axes[1].set_title("Tracking Error (Wind Case)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Error (m)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "wind_disturbance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────
# Print Tuning Metrics
# ─────────────────────────────────────────────
def print_metrics():
    print("\n{'='*60}")
    print("  PID TUNING METRICS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':<30} {'Settle(s)':>10} {'Overshoot(m)':>14} {'SS Error(m)':>12}")
    print(f"  {'-'*68}")
    for label, cfg in tuning_configs.items():
        _, heights, errors, _, _ = simulate(**cfg)
        time = np.arange(0, T_END, DT)
        overshoot = max(0, heights.max() - TARGET_HEIGHT)
        # settling time: first t where |error| < 0.5 m and stays there
        within = np.where(np.abs(errors) < 0.5)[0]
        settle = time[within[0]] if len(within) > 0 else T_END
        ss_err = abs(errors[-1])
        print(f"  {label:<30} {settle:>10.2f} {overshoot:>14.3f} {ss_err:>12.4f}")
    print()


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out, exist_ok=True)

    print("\n🚁  UAV PID Altitude Control Simulation")
    print("=" * 45)
    print("  Generating plots...")

    p1 = plot_tuning_comparison(out)
    p2 = plot_optimal_hover(out)
    p3 = plot_wind_disturbance(out)
    print_metrics()

    print("✅  All plots saved to results/")
