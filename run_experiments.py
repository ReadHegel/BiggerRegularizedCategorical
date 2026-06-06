import os
import sys
import subprocess
import concurrent.futures

ENV = os.environ.copy()
ENV["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
ENV["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

EXPERIMENTS = {
    1: ["bro", "flashsac"],
    2: ["bro", "flashsac", "xqc", "simbaV2"],
    3: ["bro", "flashsac", "xqc", "simbaV2"],
    4: ["bro", "flashsac", "xqc", "simbaV2"],
}
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))
LOG_DIR = "experiment_logs"

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
    
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file_path = os.path.join(LOG_DIR, f"{arch}_seed{seed}.log")
    
    with open(log_file_path, "w") as log_file:
        try:
            subprocess.run(
                cmd,
                env=ENV,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )
            print(f"[COMPLETED] Arch: {arch:10} | Seed: {seed} | Log: {log_file_path}", flush=True)
            return True
        except subprocess.CalledProcessError as e:
            print(
                f"[FAILED]    Arch: {arch:10} | Seed: {seed} | Exit code: {e.returncode} | Log: {log_file_path}",
                file=sys.stderr,
                flush=True
            )
            return False

def main():
    commands = build_commands()
    total_runs = len(commands)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(run_experiment, commands))
        
    success_count = sum(1 for r in results if r)

if __name__ == "__main__":
    main()
