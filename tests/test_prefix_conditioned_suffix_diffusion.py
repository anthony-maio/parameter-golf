import importlib.util
import pathlib
import unittest

import torch


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "records"
    / "track_non_record_16mb"
    / "2026-03-26_PrefixConditionedSuffixDiffusion"
    / "train_gpt.py"
)


def load_submission_module():
    spec = importlib.util.spec_from_file_location("literal_diffusion_submission_train_gpt", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PrefixConditionedSuffixDiffusionTests(unittest.TestCase):
    def test_module_exposes_literal_diffusion_helpers(self):
        module = load_submission_module()
        self.assertTrue(hasattr(module, "q_sample_suffix"))
        self.assertTrue(hasattr(module, "sample_diffusion_timesteps"))
        self.assertTrue(hasattr(module, "diffusion_pll_one_step"))

    def test_q_sample_suffix_keeps_prefix_and_corrupts_only_suffix(self):
        module = load_submission_module()
        clean = torch.tensor([[1, 10, 11, 12, 13, 14]], dtype=torch.int64)
        prefix_lengths = torch.tensor([3], dtype=torch.int64)
        timesteps = torch.tensor([4], dtype=torch.int64)
        noisy, target_mask = module.q_sample_suffix(
            clean,
            prefix_lengths,
            timesteps,
            num_steps=8,
            mask_token_id=2,
            generator=torch.Generator().manual_seed(0),
        )
        self.assertTrue(torch.equal(noisy[:, :3], clean[:, :3]))
        self.assertTrue(torch.equal(target_mask[:, :3], torch.zeros_like(target_mask[:, :3], dtype=torch.bool)))
        self.assertTrue(target_mask[:, 3:].any())

    def test_build_diffusion_conditioning_marks_prefix_and_suffix_roles(self):
        module = load_submission_module()
        input_ids = torch.tensor([[1, 10, 2, 2]], dtype=torch.int64)
        prefix_lengths = torch.tensor([2], dtype=torch.int64)
        timesteps = torch.tensor([3], dtype=torch.int64)
        time_ids, role_ids = module.build_diffusion_conditioning(input_ids, prefix_lengths, timesteps)
        self.assertTrue(torch.equal(time_ids, torch.tensor([[3, 3, 3, 3]], dtype=torch.int64)))
        self.assertTrue(torch.equal(role_ids, torch.tensor([[0, 0, 1, 1]], dtype=torch.int64)))

    def test_diffusion_pll_one_step_scores_first_masked_token_only(self):
        module = load_submission_module()
        logits = torch.log(torch.tensor([[[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]]], dtype=torch.float32))
        targets = torch.tensor([[2, 1]], dtype=torch.int64)
        mask = torch.tensor([[True, False]], dtype=torch.bool)
        nll = module.diffusion_pll_one_step(logits, targets, mask)
        self.assertEqual(nll.shape, torch.Size([1]))
        self.assertTrue(torch.all(nll > 0))

    def test_diffusion_denoising_loss_ignores_clean_prefix_positions(self):
        module = load_submission_module()
        logits = torch.zeros((1, 4, 8), dtype=torch.float32)
        targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        target_mask = torch.tensor([[False, False, True, True]], dtype=torch.bool)
        loss = module.diffusion_denoising_loss(logits, targets, target_mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_gpt_forward_logits_accepts_diffusion_conditioning(self):
        module = load_submission_module()
        model = module.GPT(
            vocab_size=16,
            num_layers=2,
            model_dim=8,
            num_heads=2,
            num_kv_heads=1,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.01,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.0,
            max_diffusion_step=8,
        )
        input_ids = torch.tensor([[1, 10, 2, 2]], dtype=torch.int64)
        time_ids = torch.tensor([[3, 3, 3, 3]], dtype=torch.int64)
        role_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.int64)
        logits = model.forward_logits(input_ids, time_ids=time_ids, role_ids=role_ids)
        self.assertEqual(logits.shape, torch.Size([1, 4, 16]))


if __name__ == "__main__":
    unittest.main()
