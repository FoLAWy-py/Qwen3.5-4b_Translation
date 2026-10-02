"""Near-limit synthetic context pressure check, separate from semantic accuracy."""
import argparse
import time
from pathlib import Path
from archive.qwen3.runtime import WiTrans, make_messages, parse_translation
from witrans_tools.common import now, write_json

def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter-dir',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError('Preserve previous budget evidence')
    model = WiTrans('models/Qwen3-4B',args.adapter_dir,max_length=1024,quantization='nf4')
    text, target, budget = 'Hello.','zh-CN',32
    context = ''
    last = None
    # Repetition stresses cache and input budget only, not long-document meaning.
    while True:
        candidate = context+' We are greeting a visitor.'
        messages = make_messages(text,target,candidate,{})
        prompt = model.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        count = len(model.tokenizer.encode(prompt,add_special_tokens=False))
        if count+budget>1024:
            break
        context, last = candidate,count
    if last is None or last+budget<1016:
        raise ValueError('Pressure input is not sufficiently close to1024 budget')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    raw, ended = model.generate_raw(text,target,context=context,max_new_tokens=budget)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    prediction = parse_translation(raw)
    placement = {device:sum(p.numel() for p in model.model.parameters() if p.device.type==device) for device in ('cpu','cuda')}
    reserved = torch.cuda.max_memory_reserved()/1024**3
    report = {'at':now(),'adapter_sha256':model.adapter_sha256,'quantization':'nf4',
        'synthetic_context_repetitions':context.count('We are greeting a visitor.'),
        'requested_prompt_tokens':last,'max_new_tokens':budget,'total_budget':last+budget,
        'generation_stats':model.last_generation_stats,'ended':ended,'raw':raw,'prediction':prediction,
        'seconds':elapsed,'parameter_placement':placement,'peak_reserved_gib':reserved,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
        'technical_passed':ended and placement['cpu']==0 and reserved<=6.5
            and model.last_generation_stats['prompt_tokens']==last,
        'release_approved':False,
        'scope':'Synthetic repeated context and a short real greeting. Tests near-limit prefill/budget/end/device behavior; cannot establish long-text translation quality or every1024-token workload.'}
    write_json(args.output,report)
    print(report)

if __name__ == '__main__':
    main()
