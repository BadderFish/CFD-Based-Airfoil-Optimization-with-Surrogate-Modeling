import subprocess
import os
import pandas as pd

# ── CHANGE THIS TO YOUR ACTUAL XFOIL PATH ─────────────────
XFOIL_PATH = r"C:\Arjun Stuff\XFOIL\XFOIL6.99\xfoil.exe"   

# ── SWEEP PARAMETERS ──────────────────────────────────────
AIRFOIL    = "2412"
ALPHA_LIST = [-2, 0, 4, 8, 12]
RE_LIST    = [500000, 1000000, 2000000]


def write_xfoil_input(airfoil, alpha, reynolds, output_file="polar.txt"):
    """
    Writes the XFOIL command sequence to a text file.
    This mimics exactly what you typed manually in the verification step.
    """
    commands = f"""NACA {airfoil}
OPER
VISC {reynolds}
PACC
{output_file}

ALFA {alpha}
PACC
QUIT
"""
    with open("xfoil_input.txt", "w") as f:
        f.write(commands)


def run_xfoil(xfoil_path):
    """
    Runs XFOIL as a subprocess, feeding it the command file.
    """
    with open("xfoil_input.txt", "r") as input_file:
        subprocess.run(
            [xfoil_path],
            stdin=input_file,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30
        )


def parse_xfoil_output(output_file="polar.txt"):
    """
    Reads the XFOIL polar output file and returns Cl, Cd, Cm.
    Returns None if the run failed to converge.
    """
    if not os.path.exists(output_file):
        return None

    with open(output_file, "r") as f:
        lines = f.readlines()

    data_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("--"):
            try:
                values = [float(x) for x in line.split()]
                if len(values) >= 6:
                    data_lines.append(values)
            except ValueError:
                continue

    if not data_lines:
        return None

    row = data_lines[-1]
    # XFOIL polar columns: alpha, Cl, Cd, CDp, Cm, Top_xtr, Bot_xtr
    return {
        "alpha": row[0],
        "Cl":    row[1],
        "Cd":    row[2],
        "Cm":    row[4]
    }


def run_sweep(xfoil_path, airfoil, alpha_list, reynolds_list):
    """
    Runs the full parameter sweep across all AoA and Re combinations.
    Returns a DataFrame with all results.
    """
    results = []

    for re in reynolds_list:
        for alpha in alpha_list:
            print(f"Running: NACA {airfoil}, AoA={alpha}deg, Re={re:,}")

            if os.path.exists("polar.txt"):
                os.remove("polar.txt")

            write_xfoil_input(airfoil, alpha, re)
            run_xfoil(xfoil_path)
            data = parse_xfoil_output()

            if data is not None:
                data["airfoil"] = f"NACA {airfoil}"
                data["Re"]      = re
                data["LD"]      = data["Cl"] / data["Cd"]
                results.append(data)
                print(f"  -> Cl={data['Cl']:.4f}, Cd={data['Cd']:.5f}, L/D={data['LD']:.2f}")
            else:
                print(f"  -> FAILED TO CONVERGE (normal at extreme AoA)")

    df = pd.DataFrame(results)
    return df


# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting XFOIL sweep...")
    print(f"Airfoil : NACA {AIRFOIL}")
    print(f"AoA     : {ALPHA_LIST}")
    print(f"Reynolds: {RE_LIST}")
    print(f"Total runs: {len(ALPHA_LIST) * len(RE_LIST)}\n")

    df = run_sweep(XFOIL_PATH, AIRFOIL, ALPHA_LIST, RE_LIST)

    df.to_csv("sweep_results.csv", index=False)
    print("\nSweep complete. Results saved to sweep_results.csv")
    print("\n" + df.to_string(index=False))