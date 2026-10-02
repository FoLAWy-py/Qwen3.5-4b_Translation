"""Run native uv under the same process-local CPU affinity as verified tests."""
import subprocess
import psutil
from witrans_tools.common import now,write_json

def main():
    result=subprocess.run(['uv','build'],capture_output=True,text=True,encoding='utf-8')
    write_json('runs/takeover-20261002/final-build-receipt.json',dict(at=now(),command=['uv','build'],
        cpu_affinity=psutil.Process().cpu_affinity(),exit_code=result.returncode,
        stdout=result.stdout,stderr=result.stderr,
        prior_unisolated_attempt='uv access violation 0xC0000005; preserved here, not treated as success'))
    print(result.stdout+result.stderr,flush=True)
    if result.returncode:raise RuntimeError('Native build failed: '+str(result.returncode))

if __name__=='__main__':main()
