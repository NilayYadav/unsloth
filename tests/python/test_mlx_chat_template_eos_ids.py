import importlib
import importlib.util
import platform

import pytest


def _import_mlx_unsloth():
    if not (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx") is not None
    ):
        pytest.skip("MLX only")
    unsloth = importlib.import_module("unsloth")
    if getattr(unsloth, "DEVICE_TYPE", None) != "mlx":
        pytest.skip("MLX only")
    return unsloth


def test_get_chat_template_keeps_generation_config_stop_ids_on_mlx():
    _import_mlx_unsloth()
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from unsloth.chat_templates import get_chat_template

    words = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "hi", "hello"]
    tk = Tokenizer(
        models.WordLevel({"[UNK]": 0, **{w: i + 1 for i, w in enumerate(words)}}, unk_token = "[UNK]")
    )
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object = tk, unk_token = "[UNK]", eos_token = "<|im_end|>")
    wrapper = TokenizerWrapper(tok, eos_token_ids = [2, 3])

    wrapper = get_chat_template(
        wrapper, chat_template = "chatml", map_eos_token = False, patch_saving = False
    )

    assert type(wrapper).__name__ == "TokenizerWrapper"
    assert wrapper.eos_token_ids == {2, 3}
