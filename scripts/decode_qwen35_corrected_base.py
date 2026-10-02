"""After the owned training pipeline, replace failed API evidence with fresh outputs."""
import json
import subprocess
import sys
import time
from pathlib import Path
from witrans_tools.common import now,write_json


def main():
    state_path=Path('runs/qwen35-corrected-base-progress.json')
    assert not state_path.exists()
    state=dict(at=now(),phase='waiting_for_owned_qwen35_pipeline',release_approved=False)
    write_json(state_path,state)
    try:
        while True:
            owner=json.loads(Path('runs/qwen35-comparison-recovery2-job.json').read_text(encoding='utf-8'))
            if owner['status'] not in ('launching','running'): break
            time.sleep(5)
        # A training or fine-tuned decoding failure does not make the base
        # unnecessary; the owner must be terminal so GPU work remains serial.
        assert owner['status']=='finished'
        plan=json.loads(Path('data/prepared/qwen35-v2/plan.json').read_text(encoding='utf-8'))
        for split,spec in plan['datasets'].items():
            state.update(phase='decoding_'+split,updated_at=now())
            write_json(state_path,state)
            subprocess.run([sys.executable,'-X','utf8','-u','-m','scripts.low_cpu_run','--module',
                'scripts.evaluate_qwen35','--input',spec['path'],'--output',f'runs/qwen35-base-v2-{split}.jsonl'],check=True)
        state.update(phase='individual_base_semantic_review_pending',updated_at=now())
    except Exception as exc:
        state.update(phase='failed_preserving_evidence',error=str(exc),updated_at=now())
        raise
    finally: write_json(state_path,state)


if __name__=='__main__': main()
