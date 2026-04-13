import unittest
from unittest.mock import patch

import torch

from omnivoice.models.omnivoice import _build_block_mask_document_ids, _mask_mod_packed
from omnivoice.utils.common import resolve_inference_dtype


class InferenceRuntimeTests(unittest.TestCase):
    def test_resolve_inference_dtype_prefers_bfloat16_on_ampere(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(8, 6)),
        ):
            self.assertEqual(resolve_inference_dtype("cuda:0"), torch.bfloat16)

    def test_resolve_inference_dtype_falls_back_to_float16_on_pre_ampere_cuda(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(7, 5)),
        ):
            self.assertEqual(resolve_inference_dtype("cuda:0"), torch.float16)

    def test_resolve_inference_dtype_uses_float32_off_cuda(self) -> None:
        self.assertEqual(resolve_inference_dtype("cpu"), torch.float32)
        self.assertEqual(resolve_inference_dtype("mps"), torch.float32)

    def test_build_block_mask_document_ids_keeps_padding_self_only(self) -> None:
        document_ids = _build_block_mask_document_ids(
            [4, 2],
            max_seq_len=4,
            device=torch.device("cpu"),
        )
        expected = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, -1, -2],
            ],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(document_ids.cpu(), expected))

        self.assertTrue(_mask_mod_packed(document_ids, 1, None, 0, 1))
        self.assertFalse(_mask_mod_packed(document_ids, 1, None, 0, 2))
        self.assertFalse(_mask_mod_packed(document_ids, 1, None, 2, 3))
        self.assertTrue(_mask_mod_packed(document_ids, 1, None, 2, 2))


if __name__ == "__main__":
    unittest.main()
