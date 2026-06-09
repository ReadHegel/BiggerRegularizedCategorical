import os
import sys
import subprocess
import concurrent.futures

# Zabezpieczenie przed zajęciem 100% VRAM przez pierwszy proces (dla JAX / XLA)
ENV = os.environ.copy()
ENV["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
ENV["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

EXPERIMENTS = {
    #4: ["bro", "flashsac"],
    2: ["simbaV2", "bro", "flashsac"],
}

# Ustawiamy 2 sloty (można nadpisać zmienną środowiskową)
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 1))

def build_commands():
    extra_args = sys.argv[1:]
    commands = []
    for seed, architectures in EXPERIMENTS.items():
        for arch in architectures:
            cmd = [
                "uv", "run", "train.py",
                "--arch", arch,
                "--env_names", "DMC_DOGS",
                "--max_steps", "500000",
                "--seed", str(seed)
            ] + extra_args
            commands.append((arch, seed, cmd))
    return commands

def run_experiment(item):
    arch, seed, cmd = item
    cmd_str = " ".join(cmd)
    print(f"[STARTING] Arch: {arch:10} | Seed: {seed} | Command: {cmd_str}", flush=True)

    try:
        subprocess.run(cmd, env=ENV, check=True)
        print(f"[COMPLETED] Arch: {arch:10} | Seed: {seed}", flush=True)
        return True
    except subprocess.CalledProcessError as e:
        print(
            f"[FAILED]    Arch: {arch:10} | Seed: {seed} | Exit code: {e.returncode}",
            file=sys.stderr,
            flush=True,
        )
        return False

def main():
    commands = build_commands()
    total_runs = len(commands)
    
    print(f"Rozpoczynam {total_runs} eksperymentów używając {MAX_WORKERS} równoległych slotów.")
    
    # ThreadPoolExecutor zajmie się utrzymaniem dokładnie MAX_WORKERS procesów naraz
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(run_experiment, commands))
        
    success_count = sum(1 for r in results if r)
    print(f"\n[ZAKOŃCZONO] Sukces: {success_count}/{total_runs}")

if __name__ == "__main__":
    main()