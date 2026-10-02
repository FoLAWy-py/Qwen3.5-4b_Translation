"""Run one long local job with durable logs and an explicit completion record."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from witrans_tools.common import now, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    parser.add_argument('--log')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    job_path = Path(args.job).resolve()
    if args.worker:
        job = json.loads(job_path.read_text(encoding='utf-8'))
        job.update(status='running',supervisor_pid=os.getpid(),started_at=now())
        write_json(job_path,job)
        try:
            with Path(job['log']).open('xb') as stream:
                process = subprocess.Popen(job['command'],cwd=job['cwd'],stdout=stream,
                    stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                job['process_pid'] = process.pid
                write_json(job_path,job)
                code = process.wait()
            job.update(status='finished',exit_code=code,finished_at=now())
        except Exception as exc:
            job.update(status='failed_to_launch',error_type=type(exc).__name__,finished_at=now())
        write_json(job_path,job)
        return
    command = args.command[1:] if args.command[:1]==['--'] else args.command
    if job_path.exists() or not args.log or Path(args.log).exists() or not command:
        raise ValueError('Fresh job and log paths with explicit command required')
    job = {'status':'launching','created_at':now(),'cwd':str(Path.cwd().resolve()),
        'command':command,'log':str(Path(args.log).resolve())}
    write_json(job_path,job)
    worker = [sys.executable,'-m','scripts.background_job','--worker','--job',str(job_path)]
    flags = subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    breakaway = False
    if os.name=='nt':
        try:
            process = subprocess.Popen(worker,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,creationflags=flags|subprocess.CREATE_BREAKAWAY_FROM_JOB)
            breakaway = True
        except PermissionError:
            process = subprocess.Popen(worker,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,creationflags=flags)
    else:
        process = subprocess.Popen(worker,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,start_new_session=True)
    # The worker owns all subsequent job status writes; the launcher prints its
    # actual process handle without racing the worker's running/finished record.
    print({'job':str(job_path),'supervisor_pid':process.pid,'breakaway_from_job':breakaway},flush=True)


if __name__=='__main__':
    main()
