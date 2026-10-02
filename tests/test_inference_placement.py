"""Exercise device transfer with real CUDA kernels and a tiny tied Qwen3."""
import pytest


def test_bf16_cpu_vocab_keeps_tied_weights_and_lora_devices():
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 is required for this placement integration check")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model
    from witrans import _place_bf16_cpu_vocab

    torch.manual_seed(31)
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=32, tie_word_embeddings=True)
    base = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    model = get_peft_model(base, LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM")).eval()
    tokens = torch.tensor([[1, 5, 20, 17]], device="cuda:0")
    model.to("cuda:0")
    with torch.inference_mode():
        expected = model(tokens).logits
    model.to("cpu")
    _place_bf16_cpu_vocab(base)
    embedding, head = base.get_input_embeddings(), base.get_output_embeddings()
    assert embedding.weight.device.type == head.weight.device.type == "cpu"
    assert embedding.weight.data_ptr() == head.weight.data_ptr()
    assert all(p.device.type == "cuda" for p in base.model.layers.parameters())
    with torch.inference_mode():
        actual = model(tokens).logits
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    assert embedding.weight.device.type == head.weight.device.type == "cpu"
