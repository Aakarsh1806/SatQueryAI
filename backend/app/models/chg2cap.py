"""Real Chg2Cap adapter with strict pinned architecture and embedded vocabulary."""
from importlib.util import find_spec
from time import perf_counter

from .base import SpecialistModel
from .device import select_device, synchronize, free_cache, memory_stats
from .loading import strict_state_load
from ..preprocessing import PreparedInputs, prepare_for
from ..schemas import ChangeInput, StandardResult, Metadata


class Chg2CapModel(SpecialistModel):
    accepts_prepared_inputs = True

    @staticmethod
    def available():
        return all(find_spec(name) is not None for name in ("torch", "torchvision", "einops"))

    def __init__(self, *, entry, weights_path):
        self.entry, self.weights_path = entry, weights_path
        self.encoder = self.attentive = self.decoder = None
        self.torch = self.device = None
        self.vocab = self.id_to_word = None
        self.load_report = []

    @property
    def is_loaded(self):
        return self.decoder is not None and self.encoder is not None and self.attentive is not None

    def load(self):
        if self.is_loaded: return
        import torch
        from .vendor.chg2cap.model_encoder import Encoder, AttentiveEncoder
        from .vendor.chg2cap.model_decoder import DecoderTransformer
        self.torch = torch
        self.device = select_device(torch, self.entry.settings.get("device"))
        try:
            # weights_only avoids arbitrary object unpickling; no ImageNet downloads.
            checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=True)
            required = {"encoder_dict", "encoder_trans_dict", "decoder_dict", "vocab", "max_length"}
            if not required <= checkpoint.keys():
                raise RuntimeError(f"Checkpoint missing keys: {sorted(required - checkpoint.keys())}")
            self.vocab = checkpoint["vocab"]
            if len(self.vocab) != 436 or sorted(self.vocab.values()) != list(range(436)):
                raise RuntimeError("Expected the embedded 436-token vocabulary with unique contiguous explicit ids.")
            if checkpoint["max_length"] != 41:
                raise RuntimeError(f"Expected max_length=41; got {checkpoint['max_length']}")
            for token in ("<START>", "<END>", "<NULL>", "<UNK>"):
                if token not in self.vocab: raise RuntimeError(f"Vocabulary missing {token}")
            self.id_to_word = {index: word for word, index in self.vocab.items()}
            self.encoder = Encoder("resnet101")
            self.attentive = AttentiveEncoder(n_layers=3, feature_size=[16,16,2048], heads=8,
                                              hidden_dim=512, attention_dim=2048, dropout=.1)
            self.decoder = DecoderTransformer(encoder_dim=2048, feature_dim=2048, vocab_size=436,
                max_lengths=41, word_vocab=self.vocab, n_head=8, n_layers=1, dropout=.1)
            self.load_report = [strict_state_load(model, checkpoint[key], key) for model,key in (
                (self.encoder,"encoder_dict"), (self.attentive,"encoder_trans_dict"), (self.decoder,"decoder_dict"))]
            del checkpoint
            for model in (self.encoder,self.attentive,self.decoder): model.eval().to(self.device)
        except Exception:
            self.unload()
            raise

    def unload(self):
        self.encoder = self.attentive = self.decoder = None
        self.vocab = self.id_to_word = None
        if self.torch: free_cache(self.torch)

    def run(self, inputs, query="", options=None):
        started = perf_counter()
        try:
            if not isinstance(inputs, PreparedInputs):
                inputs = ChangeInput.model_validate(inputs)
                inputs = prepare_for("change", [inputs.before_path,inputs.after_path], config=self.entry.preprocessing)
            if inputs.model_id != "change": raise ValueError("Expected change-native inputs.")
            self.load()
            a = self.torch.from_numpy(inputs.tensors["before"]).to(self.device)
            b = self.torch.from_numpy(inputs.tensors["after"]).to(self.device)
            if a.shape != (1,3,256,256) or b.shape != a.shape:
                raise ValueError("Chg2Cap expects before/after tensors [1,3,256,256].")
            with self.torch.inference_mode():
                features = self.encoder(a,b)
                features = self.attentive(*features)
                token_ids = self.decoder.sample(*features)
            omit = {"<START>","<END>","<NULL>"}
            caption = " ".join(self.id_to_word[index] for index in token_ids if self.id_to_word[index] not in omit)
            synchronize(self.torch,self.device)
            params = dict(options or {}, normalization=self.entry.preprocessing or {"normalization":"upstream"},
                          preprocessing_notes=inputs.notes, strict_load=self.load_report,
                          token_ids=token_ids, **memory_stats(self.torch,self.device))
            return StandardResult(answer=caption, evidence=None, confidence=None,
                metadata=Metadata(model_name=self.entry.display_name, version="chg2cap-7b8cda9-best", runtime_ms=(perf_counter()-started)*1000,
                    device=str(self.device), parameters=params, mode="real", confidence_reason="not available"))
        finally:
            # Drop inference tensors before releasing model/cache memory.
            a = b = features = None
            self.unload()
