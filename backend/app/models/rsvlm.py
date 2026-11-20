"""Real Qwen2.5-VL + LoRA adapter; no ML imports until load()."""
from importlib.util import find_spec
import logging
import json
from pathlib import Path
from time import perf_counter

from .base import SpecialistModel
from .device import select_device, synchronize, free_cache, memory_stats
from ..preprocessing import PreparedInputs, prepare_for
from ..schemas import VLMInput, StandardResult, Metadata, Confidence

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
QA_SUFFIX = " Answer with a short phrase."
CAPTION_PROMPT = "Describe this satellite image in detail."
logger = logging.getLogger(__name__)


def prompt_for(task, query):
    if task == "caption": return CAPTION_PROMPT
    if not query.strip(): raise ValueError("QA requires a nonempty question.")
    return query if query.endswith(QA_SUFFIX) else query + QA_SUFFIX


def verify_adapter(model, directory):
    """Verify every adapter key, shape, and loaded value after a dtype cast."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    expected = load_file(str(directory / "adapter_model.safetensors"), device="cpu")
    actual = get_peft_model_state_dict(model)
    missing, extra = sorted(set(expected)-set(actual)), sorted(set(actual)-set(expected))
    shapes = [{"key":key,"checkpoint":list(expected[key].shape),"model":list(actual[key].shape)}
              for key in sorted(set(expected)&set(actual)) if expected[key].shape != actual[key].shape]
    if missing or extra or shapes:
        raise RuntimeError(f"Strict LoRA mismatch: missing={missing}; unexpected={extra}; shape_mismatches={shapes}")
    if any(tensor.is_meta for tensor in actual.values()):
        raise RuntimeError("Strict LoRA value verification requires materialized adapter tensors, not meta tensors.")
    different = [key for key in expected if not torch.equal(actual[key].detach().cpu(), expected[key].to(actual[key].dtype))]
    if different: raise RuntimeError(f"LoRA tensors were not loaded exactly (after dtype cast): {different}")
    return {"strict": True, "adapter_tensors":len(expected), "verified_values":True}


class RSVLMModel(SpecialistModel):
    accepts_prepared_inputs = True
    required_files = ("adapter_model.safetensors", "adapter_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")

    @staticmethod
    def available():
        return all(find_spec(name) is not None for name in ("torch", "transformers", "peft", "accelerate", "qwen_vl_utils"))

    def __init__(self, *, entry, weights_path):
        self.entry, self.weights_path = entry, weights_path
        self.model = self.processor = self.torch = self.device = None
        self.load_report = None

    @property
    def is_loaded(self): return self.model is not None and self.processor is not None

    def load(self):
        if self.is_loaded: return
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from peft import PeftModel
        self.torch = torch
        self.device = select_device(torch,self.entry.settings.get("device"))
        dtype_name = self.entry.settings.get("dtype", "bfloat16")
        if dtype_name not in {"float16", "bfloat16"}: raise ValueError("RS-VLM dtype must be float16 or bfloat16.")
        if self.device.type == "cuda" and dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
            dtype_name = "float16"
        dtype = getattr(torch,dtype_name)
        try:
            logger.warning("RS-VLM first real load downloads base model %s unless already cached; adapter files do not contain it.", BASE_MODEL)
            if getattr(self, "progress", None):
                from huggingface_hub import try_to_load_from_cache
                index = try_to_load_from_cache(BASE_MODEL, "model.safetensors.index.json")
                cached = False
                if isinstance(index, str):
                    shards = set(json.loads(Path(index).read_text())["weight_map"].values())
                    cached = all(isinstance(try_to_load_from_cache(BASE_MODEL, name), str) for name in shards)
                self.progress("Loading cached RS-VLM base weights" if cached else
                    "RS-VLM base weights are not fully cached; downloading missing files, then loading. This may take several minutes.")
            self.processor = AutoProcessor.from_pretrained(self.weights_path,local_files_only=True,
                min_pixels=100352,max_pixels=313600)
            # Use the supplied tokenizer chat template, including vision markers.
            template = getattr(self.processor,"chat_template",None) or self.processor.tokenizer.chat_template
            if not template: raise RuntimeError("Adapter processor/tokenizer chat template is missing.")
            self.processor.tokenizer.chat_template = template
            base = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=dtype,
                attn_implementation="eager", local_files_only=self.entry.settings.get("local_files_only",False))
            if getattr(self, "progress", None):
                self.progress("Base model loaded; applying LoRA and strictly verifying adapter tensors")
            self.model = PeftModel.from_pretrained(base,self.weights_path,is_trainable=False,
                                                   autocast_adapter_dtype=False,low_cpu_mem_usage=True,torch_device="cpu")
            del base
            self.load_report = verify_adapter(self.model, self.weights_path)
            self.model.eval().to(device=self.device,dtype=dtype)
        except Exception:
            self.unload()
            raise

    def unload(self):
        self.model = self.processor = None
        if self.torch: free_cache(self.torch)

    def run(self, inputs, query="", options=None):
        started = perf_counter(); options = dict(options or {})
        try:
            task = options.get("task", "qa")
            if not isinstance(inputs,PreparedInputs):
                item = VLMInput.model_validate(inputs); task = item.task
                inputs = prepare_for("rs_vlm",[item.image_path])
            if inputs.model_id != "rs_vlm": raise ValueError("Expected RS-VLM-native inputs.")
            if task not in {"qa","caption"}: raise ValueError("RS-VLM task must be qa or caption.")
            prompt = prompt_for(task,query)
            self.load()
            from PIL import Image
            import numpy as np
            from qwen_vl_utils import process_vision_info
            image = Image.fromarray(np.rint(inputs.tensors["rgb"]).astype(np.uint8))
            messages = [{"role":"user","content":[{"type":"image","image":image,"min_pixels":100352,"max_pixels":313600},{"type":"text","text":prompt}]}]
            text = self.processor.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            image_inputs,_ = process_vision_info(messages)
            batch = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(self.device)
            # Floating visual inputs follow the model precision, token ids stay integer.
            for key,value in batch.items():
                if value.is_floating_point(): batch[key] = value.to(self.model.dtype)
            max_tokens = self.entry.settings.get("caption_max_new_tokens",128) if task == "caption" else self.entry.settings.get("qa_max_new_tokens",20)
            with self.torch.inference_mode():
                generated = self.model.generate(**batch,do_sample=False,max_new_tokens=max_tokens,
                    return_dict_in_generate=True,output_scores=True)
            new_tokens = generated.sequences[:,batch.input_ids.shape[1]:]
            answer = self.processor.tokenizer.batch_decode(new_tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
            probabilities = [float(score.float().softmax(-1)[0,int(token)]) for score,token in zip(generated.scores,new_tokens[0])]
            confidence = Confidence(value=sum(probabilities)/len(probabilities),calibrated=False,
                note="uncalibrated mean generated-token probability (includes EOS when generated); not an accuracy guarantee") if probabilities else None
            synchronize(self.torch,self.device)
            params = dict(options,prompt=prompt,base_model=BASE_MODEL,min_pixels=100352,max_pixels=313600,
                dtype=str(self.model.dtype),do_sample=False,max_new_tokens=max_tokens,strict_load=self.load_report,token_count=len(probabilities),
                preprocessing_notes=inputs.notes,**memory_stats(self.torch,self.device))
            return StandardResult(answer=answer,evidence=None,confidence=confidence,metadata=Metadata(
                model_name=self.entry.display_name,version="qwen2.5-vl-3b-lora-r32",runtime_ms=(perf_counter()-started)*1000,
                device=str(self.device),parameters=params,mode="real",warnings=["Generated text may hallucinate objects, counts or source details. Verify against the image; no model-generated spatial evidence is available."],confidence_reason="not available" if confidence is None else None))
        finally:
            batch = generated = new_tokens = None
            self.unload()
