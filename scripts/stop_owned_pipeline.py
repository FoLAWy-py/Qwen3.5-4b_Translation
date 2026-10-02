"""Stop only this recorded pipeline tree when its owning goal is paused."""
import json
from pathlib import Path
import psutil
from witrans_tools.common import now, write_json

job_path = Path('runs/v11-pipeline-job.json').resolve()
job = json.loads(job_path.read_text(encoding='utf-8'))
supervisor = psutil.Process(job['supervisor_pid'])
command = supervisor.cmdline()
if 'scripts.background_job' not in command or str(job_path) not in command:
    raise ValueError('Recorded PID no longer belongs to this pipeline')
targets = supervisor.children(recursive=True)+[supervisor]
identities = [{'pid':p.pid,'created_at':p.create_time()} for p in targets]
for process in reversed(targets):
    try:
        process.terminate()
    except psutil.NoSuchProcess:
        pass
gone,alive = psutil.wait_procs(targets,timeout=5)
for process in alive:
    try:
        process.kill()
    except psutil.NoSuchProcess:
        pass
_,remaining = psutil.wait_procs(alive,timeout=5)
if remaining:
    raise RuntimeError('Owned pipeline still has live processes')
progress_path = Path('runs/v11-pipeline-progress.json')
progress = json.loads(progress_path.read_text(encoding='utf-8'))
progress.update(previous_phase=progress['phase'],phase='stopped_on_paused_goal',updated_at=now(),
    reason='Authoritative get_goal returned paused; no full training or semantic acceptance claim.')
write_json(progress_path,progress)
job.update(status='stopped_on_paused_goal',finished_at=now(),stopped_processes=identities,
    reason='Authoritative goal is paused; owned process tree stopped, all existing artifacts preserved.')
write_json(job_path,job)
print({'status':job['status'],'stopped_pids':[p['pid'] for p in identities]})
