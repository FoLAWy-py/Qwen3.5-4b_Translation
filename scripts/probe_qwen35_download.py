"""Probe public download endpoints without displaying signed redirect URLs."""
import time
from urllib.parse import urlsplit
import httpx
from witrans_tools.common import now, write_json


def main():
    rows = []
    name = 'model.safetensors-00001-of-00002.safetensors'
    revision = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
    for host in ('huggingface.co','hf-mirror.com'):
        result = dict(host=host,file=name,revision=revision)
        try:
            started = time.perf_counter()
            with httpx.Client(follow_redirects=True,timeout=20) as client:
                with client.stream('GET',f'https://{host}/Qwen/Qwen3.5-4B/resolve/{revision}/{name}',
                    params={'download':'true','witrans_probe':str(time.time_ns())},headers={'Range':'bytes=0-1023'}) as response:
                    result.update(status=response.status_code,final_host=urlsplit(str(response.url)).hostname,
                        content_length=response.headers.get('content-length'),
                        first_chunk_bytes=len(next(response.iter_bytes(chunk_size=1024),b'')))
            result['seconds'] = time.perf_counter()-started
        except Exception as exc:
            result.update(error_type=type(exc).__name__)
        rows.append(result)
        print(result,flush=True)
    write_json('runs/qwen35-download-probe.json',dict(at=now(),probes=rows))


if __name__=='__main__':
    main()
