"""CPU-only structural smoke test for the slot-LoRA TTT scaffolding.

Skips the imports that require GPU/flash_attn by stubbing them, then
imports the readable train script, builds a tiny GPT, and verifies:

1. Baseline forward_logits works (no LoRA bank attached).
2. attach_ttt_lora_bank allocates banks of the right shapes.
3. forward_logits with LoRA bank attached produces same output (delta=0
   at init since B=0) as baseline.
4. After perturbing a B parameter and rerunning, output changes only
   for the slot we touched.
5. _select_ttt_params returns the right set for each TTT_MODE.

Not a correctness test (we don't check numerical accuracy), just a
structural test that the wiring composes and shapes are right.
"""
import os
import sys
import types
import importlib.util
import math


def _stub_modules():
    """Stub out CUDA/GPU-dependent imports that we can't load on CPU."""
    if 'flash_attn_interface' not in sys.modules:
        m = types.ModuleType('flash_attn_interface')
        # Stub flash_attn_func with a CPU SDPA fallback (causal).
        import torch.nn.functional as F

        def flash_attn_func(q, k, v, causal=True):
            # q/k/v: (B, T, H, D). SDPA wants (B, H, T, D).
            qp = q.transpose(1, 2)
            kp = k.transpose(1, 2)
            vp = v.transpose(1, 2)
            # GQA: replicate kv heads to match q heads.
            if kp.size(1) != qp.size(1):
                rep = qp.size(1) // kp.size(1)
                kp = kp.repeat_interleave(rep, dim=1)
                vp = vp.repeat_interleave(rep, dim=1)
            out = F.scaled_dot_product_attention(qp, kp, vp, is_causal=causal)
            return out.transpose(1, 2)
        m.flash_attn_func = flash_attn_func
        sys.modules['flash_attn_interface'] = m

    if 'sentencepiece' not in sys.modules:
        m = types.ModuleType('sentencepiece')

        class SentencePieceProcessor:
            pass

        m.SentencePieceProcessor = SentencePieceProcessor
        sys.modules['sentencepiece'] = m


def _import_train_module():
    spec = importlib.util.spec_from_file_location(
        'train_gpt_readable',
        os.path.join(os.path.dirname(__file__), '..', 'sp8192_sota_readable.py'),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_tiny_hparams(mod):
    """Build a Hyperparameters-like object with tiny dimensions."""
    # The Hyperparameters class reads env at class-body time, so override
    # by setting env BEFORE importing. We do that in main().
    h = mod.Hyperparameters
    return h


def _build_tiny_model(mod):
    """Build a tiny model AND break its zero-init on attn.proj/mlp.proj so
    that perturbations actually propagate to the output. The shipped model
    starts with zero projections (residual-only init), which would mask
    any LoRA effect during testing."""
    import torch
    h = _build_tiny_hparams(mod)
    model = mod.GPT(h)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith('.proj.weight') and p.abs().max() < 1e-9:
                # Replace zero with small random so signal can propagate.
                torch.nn.init.normal_(p, std=0.05)
    return h, model


def test_baseline_forward(mod, h, model):
    import torch
    x = torch.randint(0, h.vocab_size, (2, 16))
    logits = model.forward_logits(x)
    assert logits.shape == (2, 16, h.vocab_size), f"baseline shape wrong: {logits.shape}"
    print(f"  baseline forward OK: {logits.shape}")


def test_lora_bank_attach(mod, h, model):
    import torch
    # Activate the looping schedule so num_slots is meaningful.
    model.looping_active = True
    schedule = list(model.encoder_indices) + list(model.decoder_indices)
    num_slots = len(schedule)
    print(f"  schedule (slot -> block_id): {list(enumerate(schedule))}")
    print(f"  num_slots: {num_slots}")

    # Target slots whose block_id is in the recurrent band [3, 5].
    active_slots = [s for s, bid in enumerate(schedule) if bid in (3, 4, 5)]
    print(f"  active slots for layers 3-5: {active_slots}")

    bank = model.attach_ttt_lora_bank(h, 'cpu', active_slots, ('q', 'v'), rank=4, max_bsz=2, init_scale=0.01)
    print(f"  bank attached: rank={bank.rank} num_slots={bank.num_slots} max_bsz={bank.max_bsz}")
    assert bank.A_q.shape == (2, num_slots, 4, h.model_dim), f"A_q shape wrong: {bank.A_q.shape}"
    assert bank.B_q.shape == (2, num_slots, h.model_dim, 4), f"B_q shape wrong: {bank.B_q.shape}"
    kv_dim = h.num_kv_heads * (h.model_dim // h.num_heads)
    assert bank.A_v.shape == (2, num_slots, 4, h.model_dim), f"A_v shape wrong: {bank.A_v.shape}"
    assert bank.B_v.shape == (2, num_slots, kv_dim, 4), f"B_v shape wrong: {bank.B_v.shape}"
    assert bank.A_k is None and bank.B_k is None, "k bank should be None when 'k' not in proj_set"
    assert bank.A_o is None and bank.B_o is None, "o bank should be None when 'o' not in proj_set"

    # Initial state: B=0 so delta=0 -> forward output should match baseline.
    model.ttt_lora_bsz = 2
    x = torch.randint(0, h.vocab_size, (2, 16))
    base_logits = None
    # Detach bank, compute baseline.
    model.detach_ttt_lora_bank()
    model.looping_active = True
    base_logits = model.forward_logits(x)
    # Reattach.
    bank2 = model.attach_ttt_lora_bank(h, 'cpu', active_slots, ('q', 'v'), rank=4, max_bsz=2, init_scale=0.01)
    model.ttt_lora_bsz = 2
    lora_logits = model.forward_logits(x)
    diff = (base_logits - lora_logits).abs().max().item()
    print(f"  delta=0 init: max abs diff base vs LoRA = {diff:.2e}")
    assert diff < 1e-4, f"With B=0 init, LoRA forward should match baseline; got diff={diff}"
    print("  delta=0 init: outputs match baseline OK")

    # Perturb B for one slot, rerun, expect different output.
    target_slot = active_slots[0]
    with torch.no_grad():
        bank2.B_q[:, target_slot] += 0.5
    perturbed_logits = model.forward_logits(x)
    diff_perturbed = (base_logits - perturbed_logits).abs().max().item()
    print(f"  after perturbing B_q[slot={target_slot}] += 0.5: max diff = {diff_perturbed:.2e}")
    assert diff_perturbed > 1e-6, f"After perturbing LoRA, output should differ; got diff={diff_perturbed}"

    # Larger perturbation to confirm linearity.
    with torch.no_grad():
        bank2.B_q[:, target_slot] += 5.0  # much bigger
    perturbed2 = model.forward_logits(x)
    diff_perturbed2 = (base_logits - perturbed2).abs().max().item()
    print(f"  after additional B_q[slot={target_slot}] += 5.0: max diff = {diff_perturbed2:.2e}")
    assert diff_perturbed2 > diff_perturbed, "larger perturb should give larger diff"

    # Perturbing an INACTIVE slot should give zero diff.
    inactive_slot = 0  # block 0 is not in {3,4,5}
    assert inactive_slot not in active_slots
    with torch.no_grad():
        # Reset bank, then perturb inactive slot.
        bank2.reset_state()
        bank2.B_q[:, inactive_slot] += 5.0
    inactive_logits = model.forward_logits(x)
    diff_inactive = (base_logits - inactive_logits).abs().max().item()
    print(f"  after perturbing INACTIVE B_q[slot={inactive_slot}]: max diff = {diff_inactive:.2e} (should be 0)")
    assert diff_inactive < 1e-9, f"Inactive slot perturb should not affect output; got diff={diff_inactive}"
    print("  inactive slot guard works OK")

    model.detach_ttt_lora_bank()


def test_gradient_flow(mod, h, model):
    """Verify gradient flows through the LoRA bank when computing loss."""
    import torch
    model.looping_active = True
    active_slots = [3, 5, 7, 8, 9]
    bank = model.attach_ttt_lora_bank(h, 'cpu', active_slots, ('q', 'v'), rank=4, max_bsz=2, init_scale=0.1)
    model.ttt_lora_bsz = 2

    # Freeze base params, only bank gets grad
    for p in model.parameters():
        p.requires_grad_(False)
    for p in bank.parameters():
        p.requires_grad_(True)

    x = torch.randint(0, h.vocab_size, (2, 16))
    y = torch.randint(0, h.vocab_size, (2, 16))
    loss = model(x, y)
    loss.backward()

    # Active slots should have nonzero grad on B
    grad_active = bank.B_q.grad[:, active_slots[0]].abs().max().item()
    grad_inactive = bank.B_q.grad[:, 0].abs().max().item() if 0 not in active_slots else 0.0
    print(f"  loss={loss.item():.4f}")
    print(f"  bank.B_q.grad[:, slot={active_slots[0]}] max abs: {grad_active:.4e}")
    print(f"  bank.B_q.grad[:, slot=0] (inactive) max abs: {grad_inactive:.4e}")
    assert grad_active > 1e-9, f"Active slot should accumulate gradient; got {grad_active}"
    assert grad_inactive < 1e-12, f"Inactive slot should have zero gradient; got {grad_inactive}"

    # Cleanup
    for p in model.parameters():
        p.requires_grad_(True)
    model.detach_ttt_lora_bank()


def test_select_ttt_params(mod, h, model):
    """Verify _select_ttt_params returns expected sets for each mode.
    Hyperparameters class evaluates env at import time, so we monkey-patch
    the attributes directly instead of re-reading env."""
    print("  testing _select_ttt_params modes...")

    def patched(**kwargs):
        """Return a shim Hyperparameters with overridden attrs but inheriting
        the rest from the imported class."""
        class HShim:
            pass
        for name in dir(h):
            if name.startswith('_'):
                continue
            try:
                setattr(HShim, name, getattr(h, name))
            except Exception:
                pass
        for k, v in kwargs.items():
            setattr(HShim, k, v)
        return HShim

    # control mode on layers 3,4,5
    h2 = patched(ttt_layer_ids='3,4,5', ttt_mode='control', ttt_include_global=False, ttt_lora_enabled=False)
    params, layer_ids, mode, bank, slots = mod._select_ttt_params(model, h2, 'cpu')
    n_params = sum(p.numel() for p in params)
    print(f"    control mode layers 3-5: {len(params)} tensors, {n_params} numel, mode={mode}")
    assert mode == 'control'
    assert layer_ids == [3, 4, 5]
    assert bank is None
    assert n_params > 0, "control mode should select some params"

    # qv mode
    h2 = patched(ttt_layer_ids='3,4,5', ttt_mode='qv', ttt_include_global=False, ttt_lora_enabled=False)
    params, _, mode, _, _ = mod._select_ttt_params(model, h2, 'cpu')
    assert len(params) == 6, f"qv on 3 layers should give 6 tensors, got {len(params)}"
    print(f"    qv mode layers 3-5: {len(params)} tensors OK")

    # qkv mode
    h2 = patched(ttt_layer_ids='3,4,5', ttt_mode='qkv', ttt_include_global=False, ttt_lora_enabled=False)
    params, _, mode, _, _ = mod._select_ttt_params(model, h2, 'cpu')
    assert len(params) == 9, f"qkv on 3 layers should give 9 tensors, got {len(params)}"
    print(f"    qkv mode layers 3-5: {len(params)} tensors OK")

    # full mode
    h2 = patched(ttt_layer_ids='3,4,5', ttt_mode='full', ttt_include_global=False, ttt_lora_enabled=False)
    params, _, mode, _, _ = mod._select_ttt_params(model, h2, 'cpu')
    n_full = sum(p.numel() for p in params)
    print(f"    full mode layers 3-5: {len(params)} tensors, {n_full} numel OK")
    assert n_full > 0

    # control + global
    h2 = patched(ttt_layer_ids='3,4,5', ttt_mode='control', ttt_include_global=True, ttt_lora_enabled=False)
    params, _, _, _, _ = mod._select_ttt_params(model, h2, 'cpu')
    ids = {id(p) for p in params}
    assert id(model.skip_weights) in ids, "control + global should include skip_weights"
    if model.skip_gates is not None:
        assert id(model.skip_gates) in ids, "control + global should include skip_gates"
    print(f"    control + global: {len(params)} tensors OK (includes skip_weights/skip_gates)")

    # LoRA mode
    h2 = patched(ttt_layer_ids='3,4,5', ttt_mode='none', ttt_include_global=False,
                 ttt_lora_enabled=True, ttt_lora_rank=4, ttt_lora_max_bsz=2,
                 ttt_lora_proj='q,v', ttt_lora_slot_ids='', ttt_lora_init_scale=0.02)
    params, layer_ids, mode, bank, lora_slots = mod._select_ttt_params(model, h2, 'cpu')
    assert bank is not None, "LoRA enabled should attach bank"
    assert mode == 'none', f"mode should be 'none' here, got {mode}"
    assert len(params) == 4, f"LoRA q,v should give 4 tensors, got {len(params)}"
    print(f"    LoRA q,v: {len(params)} bank tensors, slots={lora_slots} OK")
    model.detach_ttt_lora_bank()


def main():
    # Set tiny model env BEFORE importing (Hyperparameters reads env at class-body eval).
    os.environ['NUM_LAYERS'] = '6'
    os.environ['MODEL_DIM'] = '32'
    os.environ['EMBEDDING_DIM'] = '32'
    os.environ['NUM_HEADS'] = '4'
    os.environ['NUM_KV_HEADS'] = '2'
    os.environ['VOCAB_SIZE'] = '64'
    os.environ['ROPE_DIMS'] = '4'
    os.environ['MLP_MULT'] = '2'
    os.environ['XSA_LAST_N'] = '0'  # disable xsa for tiny model
    os.environ['NUM_LOOPS'] = '2'
    os.environ['LOOP_START'] = '2'
    os.environ['LOOP_END'] = '3'
    os.environ['PARALLEL_RESIDUAL_START'] = '4'
    os.environ['LN_SCALE'] = '0'
    os.environ['SKIP_GATES_ENABLED'] = '1'
    os.environ['TIE_EMBEDDINGS'] = '1'
    os.environ['LOGIT_SOFTCAP'] = '30'
    os.environ['QK_GAIN_INIT'] = '5'
    os.environ['DATA_DIR'] = '/tmp/parameter_golf_test'
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')

    _stub_modules()
    print(f"loading train_gpt_readable...")
    mod = _import_train_module()
    print(f"  module loaded OK")

    h, model = _build_tiny_model(mod)
    print(f"built tiny model: num_layers={h.num_layers} dim={h.model_dim} vocab={h.vocab_size}")
    print(f"  encoder_indices: {model.encoder_indices}")
    print(f"  decoder_indices: {model.decoder_indices}")

    print("test_baseline_forward...")
    test_baseline_forward(mod, h, model)

    print("test_lora_bank_attach...")
    test_lora_bank_attach(mod, h, model)

    print("test_gradient_flow...")
    test_gradient_flow(mod, h, model)

    print("test_select_ttt_params...")
    test_select_ttt_params(mod, h, model)

    print("\nALL TESTS PASSED")


if __name__ == '__main__':
    main()
