"""Stop only verified task-owned processes after user's explicit scope change."""
import json
from pathlib import Path
import psutil
from witrans_tools.common import now, write_json


def main():
    evidence = []
    for name, module in [('v12-factorial-evaluation-job','scripts.run_v12_factorial_evaluation'),
                         ('v12-factorial-job-2','scripts.run_v12_factorial')]:
        path = Path('runs')/(name+'.json')
        job = json.loads(path.read_text(encoding='utf-8'))
        if job['status'] != 'running':
            evidence.append({'job':str(path),'prior_status':job['status'],'action':'already terminal'})
            continue
        worker = psutil.Process(job['process_pid'])
        supervisor = psutil.Process(job['supervisor_pid'])
        if module not in worker.cmdline() or 'scripts.background_job' not in supervisor.cmdline():
            raise ValueError('Process ownership verification failed')
        owned = {p.pid:(p,p.create_time()) for root in (supervisor,worker)
                 for p in [root,*root.children(recursive=True)]}
        # Prevent the supervisor/pipeline from starting another branch first.
        order = [supervisor.pid,worker.pid]+[pid for pid in owned if pid not in (supervisor.pid,worker.pid)]
        stopped = []
        for pid in order:
            process, created = owned[pid]
            try:
                if process.is_running() and process.create_time() == created:
                    process.kill()
                    stopped.append({'pid':pid,'created_at':created})
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs([item[0] for item in owned.values()],timeout=10)
        remaining = [pid for pid,(process,created) in owned.items()
                     if process.is_running() and process.create_time() == created]
        if remaining:
            raise ValueError(f'Task processes still alive: {remaining}')
        job.update(status='stopped_by_user_scope_change', finished_at=now(), stopped_processes=stopped,
                   reason='User explicitly requested no more low-value base comparisons; focus all effort on model optimization')
        write_json(path,job)
        evidence.append({'job':str(path),'stopped':stopped})
    write_json('runs/v12-optimization-redirection-stop.json',{'at':now(),'evidence':evidence,
              'artifacts_preserved':True,'default_weights_changed':False})
    print(evidence,flush=True)


if __name__ == '__main__':
    main()
