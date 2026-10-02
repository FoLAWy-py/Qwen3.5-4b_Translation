"""Adopt an already running owned TRAIN child; never restart or run CUDA."""
import argparse
import json
import os
from pathlib import Path
import psutil
from witrans_tools.common import fingerprint, now, read_jsonl, write_json

PID = 57036
CREATED = 1790900639.7177055
OUTPUT = Path('runs/qwen35-v3-train-discovery-recovery1.jsonl')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--state',required=True)
    args=parser.parse_args()
    dest=Path(args.state)
    assert os.name=='nt' and not dest.exists()
    process=psutil.Process(PID)
    command=process.cmdline()
    assert abs(process.create_time()-CREATED)<0.001
    assert 'scripts.evaluate_qwen35' in command and command[command.index('--output')+1]==str(OUTPUT).replace('\\','/')
    assert command[command.index('--adapter-dir')+1]=='models/witrans-qwen35-v2-critical-cpo'
    import _winapi
    # Hold the existing process HANDLE through exit, so PID reuse and a gap
    # between polling calls cannot lose the actual Windows exit code.
    handle=_winapi.OpenProcess(0x00100000|0x1000,False,PID)
    state=dict(at=now(),status='observing',observer_pid=os.getpid(),target_pid=PID,
        target_created=CREATED,target_command=command,initial_rows=len(read_jsonl(OUTPUT)),
        gpu_work_executed=False,stage_goal_complete=False,release_approved=False)
    write_json(dest,state)
    try:
        while _winapi.WaitForSingleObject(handle,20000)==258:
            state.update(at=now(),rows=len(read_jsonl(OUTPUT)))
            write_json(dest,state)
        code=_winapi.GetExitCodeProcess(handle)
        state.update(target_exit_code=code,status='child_exited',at=now())
        write_json(dest,state)
        assert code==0, f'Original TRAIN child exited {code}; preserve partial evidence'
        summary=json.loads(OUTPUT.with_suffix('.summary.json').read_text(encoding='utf-8'))
        rows=read_jsonl(OUTPUT)
        assert len(rows)==summary['rows']==416 and fingerprint(rows)==summary['generation_hash']
        assert summary['adapter_sha256']=='fe983cd436a3d2672e33071bb8e466a4e1ed2f64b76961c836880d8d5f8dfb27'
        assert summary['cpu_parameter_count']==0 and summary['peak_reserved_gib']<=6.5
        state.update(status='finished',rows=416,summary_hash=fingerprint(summary),at=now())
        write_json(dest,state)
    except Exception as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error),at=now())
        write_json(dest,state)
        raise
    finally:
        _winapi.CloseHandle(handle)


if __name__=='__main__':
    main()
