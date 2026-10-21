"""
run_logged.py
-------------
Small helper used by run.bat: runs a command, shows its output live in the
terminal AND saves a copy to a log file (like the Linux `tee` command).

    python tools/run_logged.py logs/train.log python src/train.py --batch-size 32
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: run_logged.py <log file> <command> [args...]")
    log_path = Path(sys.argv[1])
    command = sys.argv[2:]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        for raw in iter(process.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace")
            sys.stdout.write(line.encode(sys.stdout.encoding or "utf-8", errors="replace")
                             .decode(sys.stdout.encoding or "utf-8", errors="replace"))
            sys.stdout.flush()
            log.write(line)
            log.flush()
        code = process.wait()
        log.write(f"\n[exit code {code} after {time.time() - start:.0f} s]\n")
    sys.exit(code)


if __name__ == "__main__":
    main()
