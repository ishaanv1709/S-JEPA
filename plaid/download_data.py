"""
Vortaz Labs — PLAID AirfRANS Data Download & Preparation
==========================================================
Downloads the AirfRANS dataset (RANS CFD simulations over airfoils)
and preprocesses it for Symbolic-JEPA training.

Key dataset properties:
  - 1000 simulations over NACA 4 & 5-digit airfoils
  - Each sim = point cloud (mesh nodes)
  - Input features: inlet velocity (u,v), distance to airfoil, normals (5D)
  - Target fields: velocity, pressure, turbulent viscosity (4D)
  - Action: Angle of Attack, Reynolds number
  - Small subset used for fast training on RTX 3050

Usage: python plaid/download_data.py
"""

import numpy as np
import os
import sys
import time
import json
from pathlib import Path
from tqdm import tqdm

# Use a SMALL subset for fast training on RTX 3050
MAX_SIMULATIONS = 1000  # Full 1000 simulations
MAX_POINTS_PER_SIM = 5000  # subsample large meshes
AGGREGATE_DIM = 64  # pooled feature dimension per simulation


def download_airfrans(data_dir: str):
    """Download AirfRANS dataset using the airfrans library."""
    print("  Downloading AirfRANS dataset...")
    print("  (This may take a few minutes on first run)\n")

    try:
        import airfrans as af

        af.dataset.download(
            root=data_dir,
            file_name='Dataset',
            unzip=True,
            OpenFOAM=False,  # pre-processed version (smaller)
        )
        print(f"  Dataset downloaded to {data_dir}")
        return True

    except ImportError:
        print("  WARNING: 'airfrans' library not installed.")
        print("  Install: pip install airfrans")
        print("  Generating SYNTHETIC AirfRANS-like data instead...")
        return False

    except Exception as e:
        print(f"  WARNING: Download failed: {e}")
        print("  Generating SYNTHETIC AirfRANS-like data instead...")
        return False


def generate_synthetic_airfrans(data_dir: str, n_sims=MAX_SIMULATIONS):
    """
    Generate synthetic AirfRANS-like data for development/testing.
    Follows the real data format but uses simplified NACA physics.
    """
    print(f"  Generating {n_sims} synthetic RANS simulations...")
    rng = np.random.RandomState(42)

    sims = []
    for i in tqdm(range(n_sims), desc="  Generating sims", unit="sim",
                  bar_format='{l_bar}{bar:30}{r_bar}'):
        # NACA airfoil parameters
        is_4digit = i < int(n_sims * 0.7)  # 70% 4-digit, 30% 5-digit
        airfoil_type = "NACA4" if is_4digit else "NACA5"

        # Flight conditions (action space)
        aoa = rng.uniform(-5.0, 15.0)  # Angle of Attack (degrees)
        reynolds = rng.uniform(2e6, 6e6)  # Reynolds number
        mach = rng.uniform(0.3, 0.7)  # Mach number

        # Simulate point cloud (mesh nodes around airfoil)
        n_points = rng.randint(2000, MAX_POINTS_PER_SIM)

        # Coordinates around airfoil (polar + random mesh)
        theta = rng.uniform(0, 2 * np.pi, n_points)
        r = 1.0 + rng.exponential(0.5, n_points)  # distance from airfoil
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Input features (5D per point)
        aoa_rad = np.radians(aoa)
        u_inlet = mach * 340 * np.cos(aoa_rad)  # inlet velocity x
        v_inlet = mach * 340 * np.sin(aoa_rad)  # inlet velocity y
        dist = r - 1.0  # distance to airfoil surface
        nx = np.cos(theta)  # surface normals
        ny = np.sin(theta)

        inputs = np.column_stack([
            np.full(n_points, u_inlet),
            np.full(n_points, v_inlet),
            dist,
            nx,
            ny,
        ]).astype(np.float32)

        # Target fields (4D per point) — simplified RANS solution
        # Velocity magnitude decreases near airfoil, affected by AoA
        vel_mag = mach * 340 * (1.0 - 0.5 * np.exp(-dist))
        vel_x = vel_mag * np.cos(aoa_rad + 0.1 * np.sin(theta))
        vel_y = vel_mag * np.sin(aoa_rad + 0.1 * np.sin(theta))

        # Pressure (Bernoulli-like)
        pressure = 0.5 * 1.225 * (mach * 340)**2 * (1 - (vel_mag / (mach * 340))**2)
        pressure += rng.normal(0, 50, n_points)  # turbulence noise

        # Turbulent kinematic viscosity
        nu_t = 1e-4 * np.exp(-dist) * reynolds / 1e6
        nu_t += rng.exponential(1e-5, n_points)

        targets = np.column_stack([
            vel_x, vel_y, pressure, nu_t
        ]).astype(np.float32)

        sims.append({
            "id": i,
            "type": airfoil_type,
            "aoa": float(aoa),
            "reynolds": float(reynolds),
            "mach": float(mach),
            "n_points": n_points,
            "inputs": inputs,     # (n_points, 5)
            "targets": targets,   # (n_points, 4)
        })

    # Save
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save as numpy arrays (fast loading)
    np.savez_compressed(
        str(out_dir / "airfrans_processed.npz"),
        n_sims=len(sims),
        types=np.array([s["type"] for s in sims]),
        aoas=np.array([s["aoa"] for s in sims]),
        reynolds=np.array([s["reynolds"] for s in sims]),
        machs=np.array([s["mach"] for s in sims]),
        n_points=np.array([s["n_points"] for s in sims]),
    )

    # Save individual sim data
    sim_dir = out_dir / "simulations"
    sim_dir.mkdir(exist_ok=True)
    for s in tqdm(sims, desc="  Saving sims", unit="sim",
                  bar_format='{l_bar}{bar:30}{r_bar}'):
        np.savez_compressed(
            str(sim_dir / f"sim_{s['id']:04d}.npz"),
            inputs=s["inputs"],
            targets=s["targets"],
            aoa=s["aoa"],
            reynolds=s["reynolds"],
        )

    # Save metadata
    metadata = {
        "n_simulations": len(sims),
        "input_features": ["u_inlet", "v_inlet", "dist_to_airfoil", "normal_x", "normal_y"],
        "target_features": ["vel_x", "vel_y", "pressure", "nu_t"],
        "action_features": ["angle_of_attack", "reynolds_number"],
        "naca_4_count": sum(1 for s in sims if s["type"] == "NACA4"),
        "naca_5_count": sum(1 for s in sims if s["type"] == "NACA5"),
        "synthetic": True,
    }
    with open(str(out_dir / "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved {len(sims)} simulations to {out_dir}")
    print(f"  NACA 4-digit: {metadata['naca_4_count']}")
    print(f"  NACA 5-digit: {metadata['naca_5_count']}")

    return sims


def main():
    t0 = time.time()
    print("=" * 60)
    print("  VORTAZ LABS — PLAID AirfRANS Data Preparation")
    print("=" * 60)

    data_dir = str(Path(__file__).resolve().parent / "data")

    # Try downloading real data, fall back to synthetic
    success = download_airfrans(data_dir)
    if not success:
        generate_synthetic_airfrans(data_dir)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
