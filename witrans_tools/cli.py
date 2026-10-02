"""Installed Qwen3.5 CLI; historical commands live in archive/qwen3."""
import argparse
import json
import sys


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='Qwen3.5 v2 local NF4 translation')
    commands=parser.add_subparsers(dest='command',required=True)
    download=commands.add_parser('download',help='Download and verify the fixed official revision')
    download.add_argument('--base-dir',default='models/Qwen3.5-4B')
    translate=commands.add_parser('translate',help='Translate locally using the frozen v2 adapter')
    translate.add_argument('text')
    translate.add_argument('--target-lang',choices=['en','zh-CN'],required=True)
    translate.add_argument('--context',default='')
    translate.add_argument('--glossary',default='{}',help='JSON string-to-string mapping')
    translate.add_argument('--base-dir',default='models/Qwen3.5-4B')
    translate.add_argument('--adapter-dir',default='models/witrans-qwen35-v2-critical-cpo')
    translate.add_argument('--max-new-tokens',type=int,default=256)
    translate.add_argument('--runtime',choices=['eager','compiled','decode_compiled'],default='eager')
    translate.add_argument('--lora-precision',choices=['float32','bfloat16'],default='float32')
    args=parser.parse_args()
    if args.command=='download':
        from .download import download as execute
        execute(args)
    else:
        from .qwen35 import Qwen35Translator
        translator=Qwen35Translator(base_dir=args.base_dir,adapter_dir=args.adapter_dir,
            runtime=args.runtime,lora_precision=args.lora_precision)
        result=translator.translate(args.text,args.target_lang,context=args.context,
            glossary=json.loads(args.glossary),max_new_tokens=args.max_new_tokens)
        print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
