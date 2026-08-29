from pathlib import Path
import subprocess, sys

ROOT=Path(__file__).resolve().parent
scripts=["rsa_pipeline.py"]
with (ROOT/"RSA_full_run.log").open("w",encoding="utf-8") as log:
    for s in scripts:
        p=subprocess.run([sys.executable,str(ROOT/"scripts"/s)],stdout=log,stderr=subprocess.STDOUT)
        if p.returncode: raise SystemExit(f"Failed {s}; inspect {ROOT/'RSA_full_run.log'}")
print("RSA pipeline complete")
