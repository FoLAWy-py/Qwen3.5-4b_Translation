"""Stop only the stalled owned network download, preserving partial evidence."""
import json
from pathlib import Path
import psutil
from witrans_tools.common import now, write_json


def main():
    owner=json.loads(Path('runs/qwen35-download-job.json').read_text(encoding='utf-8'))
    assert owner['status']=='running' and owner['command'][-1]=='scripts.download_qwen35'
    process=psutil.Process(owner['process_pid'])
    assert process.cmdline()==owner['command']
    record=dict(at=now(),reason='Original public weight stream stalled at zero bytes; Range probe succeeds. Switch transport, preserve original log/partials.',
        pid=process.pid,created_at=process.create_time(),command=process.cmdline(),children=[])
    children=process.children(recursive=True)
    for child in children:
        record['children'].append(dict(pid=child.pid,created_at=child.create_time(),command=child.cmdline()))
    write_json('runs/qwen35-download-stall-intervention.json',record)
    for child in reversed(children):
        try: child.terminate()
        except psutil.NoSuchProcess: pass
    process.terminate()
    print({'stopped_owned_download':process.pid,'gpu_training_untouched':True})


if __name__=='__main__':
    main()
