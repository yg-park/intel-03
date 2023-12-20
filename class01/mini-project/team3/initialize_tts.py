from pathlib import Path
from bark.generation import load_model, codec_decode, _flatten_codebooks
import torch
import openvino as ov

def initialize():
    # << Download and convert models >>
    models_dir = Path("../model")
    models_dir.mkdir(exist_ok=True)

    # << Text encoder >>
    text_use_small = True

    text_encoder = load_model(
        model_type="text", use_gpu=False, use_small=text_use_small, force_reload=False
    )

    text_encoder_model = text_encoder["model"]
    tokenizer = text_encoder["tokenizer"]

    text_model_suffix = "_small" if text_use_small else ""
    text_model_dir = models_dir / f"text_encoder{text_model_suffix}"
    text_model_dir.mkdir(exist_ok=True)
    text_encoder_path1 = text_model_dir / "bark_text_encoder_1.xml"
    text_encoder_path0 = text_model_dir / "bark_text_encoder_0.xml"

    if not text_encoder_path0.exists() or not text_encoder_path1.exists():
        text_encoder_exportable = TextEncoderModel(text_encoder_model)
        ov_model = ov.convert_model(
            text_encoder_exportable, example_input=torch.ones((1, 513), dtype=torch.int64)
        )
        ov.save_model(ov_model, text_encoder_path0)
        logits, kv_cache = text_encoder_exportable(torch.ones((1, 513), dtype=torch.int64))
        ov_model = ov.convert_model(
            text_encoder_exportable,
            example_input=(torch.ones((1, 1), dtype=torch.int64), kv_cache),
        )
        ov.save_model(ov_model, text_encoder_path1)
        del ov_model
        del text_encoder_exportable
    del text_encoder_model, text_encoder

    # << Coarse encoder >>
    coarse_use_small = True

    coarse_model = load_model(
        model_type="coarse", use_gpu=False, use_small=coarse_use_small, force_reload=False, 
    )

    coarse_model_suffix = "_small" if coarse_use_small else ""
    coarse_model_dir = models_dir / f"coarse{coarse_model_suffix}"
    coarse_model_dir.mkdir(exist_ok=True)
    coarse_encoder_path = coarse_model_dir / "bark_coarse_encoder.xml"

    if not coarse_encoder_path.exists():
        coarse_encoder_exportable = CoarseEncoderModel(coarse_model)
        logits, kv_cache = coarse_encoder_exportable(
            torch.ones((1, 886), dtype=torch.int64)
        )
        ov_model = ov.convert_model(
            coarse_encoder_exportable,
            example_input=(torch.ones((1, 1), dtype=torch.int64), kv_cache),
        )
        ov.save_model(ov_model, coarse_encoder_path)
        del ov_model
        del coarse_encoder_exportable
    del coarse_model

    fine_use_small = False

    fine_model = load_model(model_type="fine", use_gpu=False, use_small=fine_use_small, force_reload=False)

    fine_model_suffix = "_small" if fine_use_small else ""
    fine_model_dir = models_dir / f"fine_model{fine_model_suffix}"
    fine_model_dir.mkdir(exist_ok=True)

    fine_feature_extractor_path = fine_model_dir / "bark_fine_feature_extractor.xml"

    # << Fine encoder >>
    if not fine_feature_extractor_path.exists():
        lm_heads = fine_model.lm_heads
        fine_feature_extractor = FineModel(fine_model)
        feature_extractor_out = fine_feature_extractor(
            3, torch.zeros((1, 1024, 8), dtype=torch.int32)
        )
        ov_model = ov.convert_model(
            fine_feature_extractor,
            example_input=(
                torch.ones(1, dtype=torch.long),
                torch.zeros((1, 1024, 8), dtype=torch.long),
            ),
        )
        ov.save_model(ov_model, fine_feature_extractor_path)
        for i, lm_head in enumerate(lm_heads):
            ov.save_model(
                ov.convert_model(lm_head, example_input=feature_extractor_out),
                fine_model_dir / f"bark_fine_lm_{i}.xml",
            )

    return tokenizer, text_encoder_path0, text_encoder_path1, coarse_encoder_path, fine_model_dir


class TextEncoderModel(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, idx, past_kv=None):
        return self.encoder(idx, merge_context=True, past_kv=past_kv, use_cache=True)


class CoarseEncoderModel(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, idx, past_kv=None):
        return self.encoder(idx, past_kv=past_kv, use_cache=True)


class FineModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pred_idx, idx):
        b, t, codes = idx.size()
        pos = torch.arange(0, t, dtype=torch.long).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_embs = [
            wte(idx[:, :, i]).unsqueeze(-1)
            for i, wte in enumerate(self.model.transformer.wtes)
        ]  # token embeddings of shape (b, t, n_embd)
        tok_emb = torch.cat(tok_embs, dim=-1)
        pos_emb = self.model.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = tok_emb[:, :, :, : pred_idx + 1].sum(dim=-1)
        x = self.model.transformer.drop(x + pos_emb)
        for block in self.model.transformer.h:
            x = block(x)
        x = self.model.transformer.ln_f(x)
        return x
