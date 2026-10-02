"""Record package inventory and exact shared-generation migration evidence."""
import ast
import copy
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
import tarfile
import zipfile

from witrans_tools.common import now,write_json


def main():
    old=subprocess.check_output(['git','show','HEAD:witrans.py'],text=True,encoding='utf-8')
    a=next(n for n in ast.parse(old).body if isinstance(n,ast.ClassDef) and n.name=='_LocalTranslator')
    b=next(n for n in ast.parse(Path('witrans_tools/runtime.py').read_text()).body if isinstance(n,ast.ClassDef))
    checks={}
    for n in a.body:
        if isinstance(n,ast.FunctionDef) and n.name in ('translate','generate_raw'):
            revised=copy.deepcopy(n);revised.args.kw_defaults[-1]=ast.Constant(value=256)
            current=next(j for j in b.body if isinstance(j,ast.FunctionDef) and j.name==n.name)
            checks[n.name]=ast.dump(revised,include_attributes=False)==ast.dump(current,include_attributes=False)
    if not all(checks.values()):raise RuntimeError('Unexplained shared-generation changes')
    archives=[]
    for path in Path('dist').iterdir():
        if path.suffix=='.whl':
            with zipfile.ZipFile(path) as z:names=z.namelist()
        elif path.name.endswith('.tar.gz'):
            with tarfile.open(path) as t:names=t.getnames()
        else:continue
        forbidden=[n for n in names if any(part in n.split('/') for part in ('models','.venv','.venv-qwen35','.env'))]
        if forbidden:raise RuntimeError(f'Forbidden package contents: {forbidden}')
        archives.append(dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),files=names))
    write_json('runs/takeover-20261002/engineering-receipt.json',dict(at=now(),generation_ast_checks=checks,
        allowed_difference='default output budget512 to256; formal calls supply explicit128/256',
        packages=archives,dependencies={k:importlib.metadata.version(k) for k in ('witrans-qwen35','torch','transformers','peft','bitsandbytes')},
        original_environment_preserved='.venv-qwen35',migration_environment='.venv-qwen35-mainline',
        no_weights_no_secrets_in_package=True,uv_lock_sha256=hashlib.sha256(Path('uv.lock').read_bytes()).hexdigest(),
        runtime_regression_status='complete; see development-final/semantic-summary.json',
        independent_confirmation='complete; see confirmation/semantic-summary.json; release entry failed',release_approved=False))
    print(dict(generation_ast_checks=checks,package_files=len(archives)),flush=True)


if __name__=='__main__':main()
