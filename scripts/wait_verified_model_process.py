"""Bounded read-only wait on a verified Windows model process handle."""
import argparse
import time
from pathlib import Path

import psutil
from witrans_tools.common import now, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--created',type=float,required=True)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--module',required=True)
    parser.add_argument('--milliseconds',type=int,default=40000)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    dest=Path(args.output)
    assert not dest.exists() and 1<=args.milliseconds<=50000
    process=psutil.Process(args.pid)
    command=process.cmdline()
    assert abs(process.create_time()-args.created)<.001
    assert args.plan in command and args.module in command
    import _winapi
    handle=_winapi.OpenProcess(0x00100000|0x1000,False,args.pid)
    started=now()
    began=time.perf_counter()
    try:
        result=_winapi.WaitForSingleObject(handle,args.milliseconds)
        assert result in (0,258)
        code=_winapi.GetExitCodeProcess(handle)
        terminal=result==0
        write_json(dest,dict(at=now(),started_at=started,
            verified_process=dict(pid=args.pid,created_epoch=args.created,command=command),
            requested_wait_ms=args.milliseconds,elapsed_seconds=time.perf_counter()-began,
            wait_result=result,actual_process_exit_code=code if terminal else None,
            observed_handle_exit_code=code,terminal=terminal,
            classification='verified wait on actual process handle',
            gpu_executed=False,process_modified_or_restarted=False,
            scope='Observation timeout is not task failure or permission to restart. Terminal exit must still be bound to durable owner job, final training budget and saved weights.',
            stage_goal_complete=False))
        print(dict(output=str(dest),terminal=terminal,wait_result=result,exit_code=code if terminal else None))
    finally:
        _winapi.CloseHandle(handle)


if __name__=='__main__':
    main()
