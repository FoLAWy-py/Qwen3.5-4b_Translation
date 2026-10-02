"""Isolated text-only Qwen3.5 runtime, with the unchanged translation protocol."""
import hashlib
import json
from pathlib import Path
from witrans import SYSTEM_PROMPT, _LocalTranslator
from witrans_tools.common import fingerprint


class Qwen35Translator(_LocalTranslator):
    def __init__(self, base_dir='models/Qwen3.5-4B', adapter_dir=None, max_length=1024):
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig, Qwen3_5ForCausalLM
        base = Path(base_dir)
        receipt = json.loads((base/'witrans_base.json').read_text(encoding='utf-8'))
        assert receipt['model_id']=='Qwen/Qwen3.5-4B' and receipt['status']=='complete'
        assert torch.cuda.is_available() and max_length==1024
        self.adapter_sha256 = None
        if adapter_dir is not None:
            adapter = Path(adapter_dir)
            metadata = json.loads((adapter/'witrans_adapter.json').read_text(encoding='utf-8'))
            assert metadata['base_model_id']==receipt['model_id'] and metadata['base_revision']==receipt['revision']
            assert metadata['prompt_hash']==fingerprint(SYSTEM_PROMPT) and not metadata['smoke_only']
            with (adapter/'adapter_model.safetensors').open('rb') as stream:
                self.adapter_sha256 = hashlib.file_digest(stream,'sha256').hexdigest()
            assert self.adapter_sha256==metadata['adapter_sha256']
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
        if any(p.device.type!='cuda' for p in self.model.parameters()):
            raise ValueError('CPU or disk parameter offload is forbidden')
