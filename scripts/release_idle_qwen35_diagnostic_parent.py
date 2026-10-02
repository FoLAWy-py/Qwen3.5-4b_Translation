"""Terminate ONLY a verified idle owned parent, preserving its active child."""
import json
import psutil
from pathlib import Path
from witrans_tools.common import now, read_jsonl, write_json


def main():
    destination=Path('runs/qwen35-v3-idle-parent-release.json')
    assert not destination.exists()
    parent=psutil.Process(67972);child=psutil.Process(57036)
    assert abs(parent.create_time()-1790899308.0070195)<0.001
    assert abs(child.create_time()-1790900639.7177055)<0.001
    assert 'scripts.diagnose_qwen35_stage' in parent.cmdline()
    assert 'scripts.evaluate_qwen35' in child.cmdline()
    assert parent.pid in {p.pid for p in child.parents()}
    progress=json.loads(Path('runs/qwen35-v3-diagnostic-recovery1/progress.json').read_text(encoding='utf-8'))
    assert progress['phase']=='train_generation'
    timing=json.loads(Path('runs/qwen35-v3-diagnostic-recovery1/performance-summary.json').read_text(encoding='utf-8'))
    assert timing['calls']==72 and timing['rounds']==3
    adopted=json.loads(Path('runs/qwen35-v3-train-adopted-observer.json').read_text(encoding='utf-8'))
    assert adopted['status']=='observing' and adopted['target_pid']==child.pid
    observer=psutil.Process(adopted['observer_pid'])
    assert 'scripts.observe_qwen35_train_child' in observer.cmdline()
    record=dict(at=now(),parent_pid=parent.pid,parent_created=parent.create_time(),parent_command=parent.cmdline(),
        preserved_child_pid=child.pid,child_created=child.create_time(),child_command=child.cmdline(),
        observer_pid=observer.pid,rows_before=len(read_jsonl('runs/qwen35-v3-train-discovery-recovery1.jsonl')),
        reason='Windows GPU counters show idle diagnostic parent still owns3.0GiB after del/gc/empty_cache. Finish only owned waiting parent after child process-handle adoption; do not kill tree or restart child.',
        state='verified_before_release',stage_goal_complete=False,release_approved=False)
    write_json(destination,record)
    parent.terminate()
    record['parent_exit_code']=parent.wait(timeout=10)
    assert child.is_running() and abs(child.create_time()-record['child_created'])<0.001
    record.update(at=now(),state='idle_parent_released_child_preserved',
        rows_after=len(read_jsonl('runs/qwen35-v3-train-discovery-recovery1.jsonl')))
    write_json(destination,record)
    print(record['state'],flush=True)


if __name__=='__main__':
    main()
