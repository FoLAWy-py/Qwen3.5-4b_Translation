"""Isolated text-only Qwen3.5 runtime, with the unchanged translation protocol."""
import hashlib
import json
from pathlib import Path
from witrans_tools.protocol import SYSTEM_PROMPT
from witrans_tools.runtime import GenerationRuntime
from witrans_tools.common import fingerprint


MODEL_ID = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
ADAPTER_SHA256 = 'fe983cd436a3d2672e33071bb8e466a4e1ed2f64b76961c836880d8d5f8dfb27'
PROMPT_HASH = '16465a8a3a3ef925155322e6fbd13df95a1c9348ece436eee5428646cbf7ba28'
DEFAULT_ADAPTER = 'models/witrans-qwen35-v2-critical-cpo'
BASE_SMALL_HASHES = {
    'config.json': 'ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670',
    'chat_template.jinja': 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715',
    'tokenizer.json': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42',
    'tokenizer_config.json': '316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8',
    'vocab.json': 'ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003',
    'merges.txt': 'a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d',
}
ADAPTER_CONFIG_SHA256 = 'afed26e98b4dc38c543308177fdebfe9652fe4a80244e3e8b501c0b650bfcc44'


def validate_runtime_files(base_dir, adapter_dir):
    for name, expected in BASE_SMALL_HASHES.items():
        with (Path(base_dir)/name).open('rb') as stream:
            actual = hashlib.file_digest(stream,'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Frozen base config/tokenizer/template mismatch: {name}')
    if adapter_dir is not None:
        with (Path(adapter_dir)/'adapter_config.json').open('rb') as stream:
            actual = hashlib.file_digest(stream,'sha256').hexdigest()
        if actual != ADAPTER_CONFIG_SHA256:
            raise ValueError('Frozen v2 adapter configuration mismatch')


def validate_identity(base_dir, adapter_dir, max_length):
    """Fail before importing CUDA libraries; checks survive python -O."""
    if type(max_length) is not int or max_length != 1024:
        raise ValueError('Qwen3.5 total token budget must be exactly 1024')
    base = Path(base_dir)
    receipt = json.loads((base/'witrans_base.json').read_text(encoding='utf-8'))
    if receipt.get('model_id') != MODEL_ID or receipt.get('revision') != REVISION or receipt.get('status') != 'complete':
        raise ValueError('Base model identity/revision/completion mismatch')
    config = json.loads((base/'config.json').read_text(encoding='utf-8'))
    if config.get('model_type') != 'qwen3_5' or config.get('text_config',{}).get('model_type') != 'qwen3_5_text':
        raise ValueError('Expected official Qwen3.5 text configuration')
    if fingerprint(SYSTEM_PROMPT) != PROMPT_HASH:
        raise ValueError('Frozen translation prompt changed')
    digest = None
    if adapter_dir is not None:
        adapter = Path(adapter_dir)
        if not (adapter/'adapter_config.json').is_file():
            raise FileNotFoundError('Missing adapter_config.json')
        metadata = json.loads((adapter/'witrans_adapter.json').read_text(encoding='utf-8'))
        if metadata.get('base_model_id') != MODEL_ID or metadata.get('base_revision') != REVISION:
            raise ValueError('Adapter base model/revision mismatch')
        if metadata.get('prompt_hash') != PROMPT_HASH or metadata.get('smoke_only') is not False:
            raise ValueError('Adapter prompt binding/smoke status mismatch')
        with (adapter/'adapter_model.safetensors').open('rb') as stream:
            digest = hashlib.file_digest(stream,'sha256').hexdigest()
        if digest != ADAPTER_SHA256 or metadata.get('adapter_sha256') != ADAPTER_SHA256:
            raise ValueError('Frozen v2 adapter SHA256 mismatch')
    return receipt, digest


def validate_cuda(torch):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Qwen3.5 NF4 requires CUDA with BF16 support')


def validate_placement(model):
    if any(p.device.type != 'cuda' for p in model.parameters()):
        raise RuntimeError('CPU or disk parameter offload is forbidden')
    if any(str(device) in ('cpu','disk') for device in getattr(model,'hf_device_map',{}).values()):
        raise RuntimeError('CPU or disk device map offload is forbidden')


class Qwen35Translator(GenerationRuntime):
    def __init__(self, base_dir='models/Qwen3.5-4B', adapter_dir=DEFAULT_ADAPTER, max_length=1024,
                 *, runtime='eager', lora_precision='float32', cache_dir='.cache/qwen35'):
        if runtime not in ('eager','compiled','decode_compiled') or lora_precision not in ('float32','bfloat16'):
            raise ValueError('Unknown runtime/LoRA precision')
        receipt, self.adapter_sha256 = validate_identity(base_dir, adapter_dir, max_length)
        validate_runtime_files(base_dir, adapter_dir)
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig, Qwen3_5ForCausalLM
        base = Path(base_dir)
        validate_cuda(torch)
        self._torch, self._generation_config = torch, GenerationConfig
        # Transformers5.18 removed this named generation option. All decoding
        # parameters are explicitly supplied by the shared GenerationConfig.
        self._supports_use_model_defaults = False
        self.max_length, self.quantization = max_length, 'nf4'
        self.tokenizer = AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
        quantization = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        model, loading = Qwen3_5ForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',
            quantization_config=quantization,use_safetensors=True,output_loading_info=True)
        missing = loading.get('missing_keys',[])
        unexpected = loading.get('unexpected_keys',[])
        if missing or loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise ValueError(f'Incomplete text checkpoint loading: {loading}')
        if any(not k.startswith(('model.visual.','mtp.','model.mtp.')) for k in unexpected):
            raise ValueError(f'Unexpected non-vision/non-MTP checkpoint keys: {unexpected}')
        self.loading_info = dict(missing_keys=sorted(missing),unexpected_count=len(unexpected),
            expected_unused='Vision tower/MTP excluded from text-only model',class_name=type(model).__name__)
        if adapter_dir is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model,adapter_dir,is_trainable=False,local_files_only=True)
        self.model = model.eval()
        self.model.config.use_cache = True
        validate_placement(self.model)
        self.runtime_config = dict(runtime=runtime,lora_precision=lora_precision,quantization='nf4',
            compute_dtype='bfloat16',cache='dynamic',compile=None)
        if lora_precision == 'bfloat16':
            from .qwen35_lora_precision import cast_qwen35_lora_to_bfloat16
            self.runtime_config['cast'] = cast_qwen35_lora_to_bfloat16(self.model)
        if runtime in ('compiled','decode_compiled'):
            from .compile_runtime import enable_compiled_runtime
            self.runtime_config.update(cache='static',compile=enable_compiled_runtime(self,cache_dir,decode_only=runtime=='decode_compiled'))

    def generate_raw(self, *args, **kwargs):
        validate_placement(self.model)
        result = super().generate_raw(*args, **kwargs)
        validate_placement(self.model)
        if self._torch.cuda.max_memory_reserved()/1024**3 > 6.5:
            raise RuntimeError('NF4 peak reserved memory exceeds 6.5 GiB')
        return result
