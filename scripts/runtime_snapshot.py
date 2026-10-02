"""Read current project Python process status and generation progress safely."""
import json
from pathlib import Path
import psutil

def main():
    processes = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']):
        try:
            info = p.info
            if 'python' not in (info['name'] or '').lower() or Path(info['cwd'] or '.').resolve() != Path.cwd().resolve():
                continue
            args = info['cmdline'] or []
            modules = [arg for arg in args if arg.startswith(('scripts.', 'witrans_tools'))]
            processes.append({'pid': info['pid'], 'name': info['name'], 'project_modules': modules, 'status': p.status()})
        except (psutil.Error, OSError):
            continue
    progress = {}
    for version in ('v4', 'v5', 'v6', 'v7','v8','v11'):
        for name in ({'v6':('cpo','sft-control'),'v7':('critical',),'v8':('selected',),'v11':('selected',)}.get(version,('selected','start'))):
            path = Path(f'runs/{version}-{name}-dev.jsonl')
            if path.exists():
                lines = path.read_text(encoding='utf-8').splitlines()
                rows = []
                for index, line in enumerate(lines):
                    try:
                        if line.strip():
                            rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        # A live writer may still be completing the final line.
                        if index != len(lines) - 1:
                            raise
                progress[f'{version}/{name}'] = {'rows': len(rows), 'last_id': rows[-1]['id'] if rows else None,
                    'summary_exists': path.with_suffix('.summary.json').exists()}
    performance = {}
    for stem in ('v7-short-eager','v7-short-compiled','v7-short-compiled-eager-rounding'):
        path = Path('runs')/(stem+'.jsonl')
        if not path.exists():
            continue
        rows = []
        lines = path.read_text(encoding='utf-8').splitlines()
        for index,line in enumerate(lines):
            try:
                if line.strip():
                    rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index!=len(lines)-1:
                    raise
        performance[stem] = {'rows':len(rows),'last_id':rows[-1]['id'] if rows else None,
            'last_round':rows[-1]['round'] if rows else None,
            'summary_exists':path.with_suffix('.summary.json').exists()}
    print(json.dumps({'processes': processes, 'progress': progress, 'performance':performance}, ensure_ascii=False))

if __name__ == '__main__':
    main()
