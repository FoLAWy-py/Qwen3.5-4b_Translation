"""Build opt-in RAG wheel without replacing the earlier package evidence."""
import hashlib
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
import psutil
from witrans_tools.common import write_json, now


def main():
    command=['uv','build','--out-dir','dist/rag-20261002']
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
    receipt={'at':now(),'command':command,'cpu_affinity':psutil.Process().cpu_affinity(),
             'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    path=Path('runs/takeover-20261002/rag/package-receipt.json')
    write_json(path,receipt)
    if result.returncode:
        raise RuntimeError('Native uv build failed: '+str(result.returncode))
    archives=[]
    for artifact in Path('dist/rag-20261002').iterdir():
        if artifact.suffix=='.whl':
            with zipfile.ZipFile(artifact) as archive:
                names=archive.namelist()
            if 'witrans_tools/rag.py' not in names:
                raise RuntimeError('Wheel omits experimental RAG module')
            code="import sys; sys.path.insert(0,sys.argv[1]); import witrans_tools.rag as r; print(r.__file__); print(r.contains('the eigenvalue','eigenvalue'))"
            imported=subprocess.run([sys.executable,'-I','-X','utf8','-c',code,str(artifact.resolve())],capture_output=True,text=True,encoding='utf-8')
            if imported.returncode or '.whl' not in imported.stdout or 'True' not in imported.stdout:
                raise RuntimeError('Isolated wheel import failed: '+imported.stderr)
            receipt['isolated_wheel_import']={'exit_code':imported.returncode,'stdout':imported.stdout}
        elif artifact.name.endswith('.tar.gz'):
            with tarfile.open(artifact) as archive:
                names=archive.getnames()
        else:
            continue
        forbidden=[n for n in names if any(s in n.split('/') for s in ('models','.venv','.venv-qwen35','.env','runs','data','archive'))]
        if forbidden:
            raise RuntimeError('Forbidden archive contents: '+str(forbidden))
        archives.append({'path':str(artifact),'bytes':artifact.stat().st_size,
            'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'file_count':len(names)})
    receipt.update(packages=archives,no_weights_or_secrets=True,verified=True)
    write_json(path,receipt)
    print({'package_count':len(archives),'wheel_import':receipt['isolated_wheel_import'],'verified':True})


if __name__=='__main__':main()
