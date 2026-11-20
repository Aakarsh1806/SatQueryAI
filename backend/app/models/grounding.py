"""Teammate Grounding DINO inference logic adapted in-process, without its server."""
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .base import SpecialistModel
from .device import select_device,synchronize,free_cache
from ..preprocessing import PreparedInputs,prepare_for
from ..schemas import GroundingInput,GroundingOutput,StandardResult,Metadata


class GroundingModel(SpecialistModel):
    accepts_prepared_inputs=True
    required_files=('model.safetensors','config.json','preprocessor_config.json','tokenizer.json','tokenizer_config.json')

    @staticmethod
    def available():return all(find_spec(name) is not None for name in ('torch','transformers'))

    def __init__(self,*,entry,weights_path):
        self.entry,self.weights_path=entry,weights_path
        self.model=self.processor=self.torch=self.device=None
        self.load_report=None
        self.device_fallback=None

    @property
    def is_loaded(self):return self.model is not None and self.processor is not None

    def load(self):
        if self.is_loaded:return
        import torch
        from transformers import AutoProcessor,AutoConfig,GroundingDinoForObjectDetection
        self.torch=torch
        self.device=select_device(torch,self.entry.settings.get('device'))
        self.device_fallback=None
        try:
            config=AutoConfig.from_pretrained(self.weights_path,local_files_only=True)
            # Use native Torch ops rather than CUDA custom extensions on Apple Silicon.
            config.disable_custom_kernels=True
            self.processor=AutoProcessor.from_pretrained(self.weights_path,local_files_only=True)
            self.model,info=GroundingDinoForObjectDetection.from_pretrained(self.weights_path,config=config,
                local_files_only=True,output_loading_info=True)
            bad={key:list(info.get(key,[])) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') if info.get(key)}
            if bad:raise RuntimeError(f'Strict Grounding DINO checkpoint mismatch: {bad}')
            self.load_report={'strict':True,'missing_keys':[],'unexpected_keys':[],'mismatched_keys':[],'training_image_size':512}
            self.model.eval()
            try:self.model.to(self.device)
            except Exception as exc:
                if self.device.type!='mps':raise
                self._cpu_fallback(exc)
        except Exception:
            self.unload();raise

    def unload(self):
        self.model=self.processor=None
        if self.torch:free_cache(self.torch)

    def _cpu_fallback(self,exc):
        self.device_fallback={'from':'mps','to':'cpu','reason':f'{type(exc).__name__}: {exc}'}
        self.device=self.torch.device('cpu')
        self.model.to(self.device)
        free_cache(self.torch)

    def _infer(self,image,prompt,threshold):
        batch=self.processor(images=image,text=prompt,return_tensors='pt').to(self.device)
        with self.torch.inference_mode():outputs=self.model(**batch)
        # Negative infinity is intentional for masked/padded text tokens.
        if self.torch.isnan(outputs.logits).any() or self.torch.isposinf(outputs.logits).any() or not self.torch.isfinite(outputs.pred_boxes).all():
            raise RuntimeError('Grounding returned non-finite logits or boxes.')
        # Transformers 5.22.2 renamed the teammate wrapper's box_threshold arg.
        rows=self.processor.post_process_grounded_object_detection(outputs,batch.input_ids,
            threshold=threshold,text_threshold=.25,target_sizes=[image.size[::-1]])[0]
        labels=rows['text_labels'] if 'text_labels' in rows else rows['labels']
        detections=[]
        for box,score,label in zip(rows['boxes'],rows['scores'],labels):
            values=box.detach().float().cpu().tolist()
            # Target coordinates are original pixels. Clip for rendering/schema safety.
            x1,y1,x2,y2=values
            clipped=[max(0.,min(image.width,x1)),max(0.,min(image.height,y1)),
                     max(0.,min(image.width,x2)),max(0.,min(image.height,y2))]
            if clipped[0]>=clipped[2] or clipped[1]>=clipped[3]:continue
            detections.append({'box':clipped,'label':str(label),'confidence':float(score)})
        synchronize(self.torch,self.device)
        # Map teammate detections rows into the common grounding contract.
        return GroundingOutput(boxes=[row['box'] for row in detections],labels=[row['label'] for row in detections],scores=[row['confidence'] for row in detections])

    def _ground_image(self,image,text_prompt,box_threshold):
        prompt=text_prompt.strip().lower()
        if not prompt:raise ValueError('Grounding requires a nonempty text phrase.')
        if not 0<=box_threshold<=1:raise ValueError('box_threshold must be between 0 and 1.')
        if not prompt.endswith('.'):prompt+='.'
        self.load()
        try:return self._infer(image,prompt,box_threshold),prompt
        except Exception as exc:
            if self.device.type!='mps':raise
            self._cpu_fallback(exc)
            return self._infer(image,prompt,box_threshold),prompt

    def ground(self,image_path,text_prompt,box_threshold=.3):
        from PIL import Image
        with Image.open(image_path) as source:image=source.convert('RGB')
        try:
            evidence,_=self._ground_image(image,text_prompt,box_threshold)
            return {'boxes':[list(box) for box in evidence.boxes],'labels':evidence.labels,'scores':evidence.scores}
        finally:self.unload()

    def run(self,inputs,query='',options=None):
        import numpy as np
        from PIL import Image,ImageDraw
        started=perf_counter();options=dict(options or {})
        try:
            threshold=options.get('box_threshold',.3)
            if not isinstance(inputs,PreparedInputs):
                item=GroundingInput.model_validate(inputs);query=item.text_prompt;threshold=item.box_threshold
                directory=Path(options.pop('artifact_dir',None) or Path(__file__).resolve().parents[3]/'uploads/prepared'/str(uuid4()))
                inputs=prepare_for('grounding',[item.image_path],output_dir=directory)
            if inputs.model_id!='grounding':raise ValueError('Expected grounding-native RGB inputs.')
            image=Image.fromarray(np.rint(inputs.tensors['rgb']).astype(np.uint8))
            evidence,prompt=self._ground_image(image,query,threshold)
            if inputs.previews:
                with Image.open(inputs.previews[0]) as source:overlay=source.convert('RGB')
                draw=ImageDraw.Draw(overlay)
                for box,label,score in zip(evidence.boxes,evidence.labels,evidence.scores):
                    draw.rectangle(box,outline='red',width=2);draw.text((box[0],box[1]),f'{label[:40]} {score:.2f} (uncalibrated)',fill='red')
                target=Path(inputs.previews[0]).parent/'grounding_overlay.png';overlay.save(target)
                evidence.overlay_path=str(target)
            warnings=['Grounding has no trainer-supplied benchmark metrics. Local small-sample sanity checks do not establish benchmark accuracy.',
                      'Per-box scores are uncalibrated; there is no supported aggregate answer confidence.']
            if self.device_fallback: warnings.append(f'MPS failed; grounding ran on CPU: {self.device_fallback["reason"]}')
            return StandardResult(answer=f'Located {len(evidence.boxes)} candidate region(s) for "{prompt}".',evidence=evidence,confidence=None,
                metadata=Metadata(model_name=self.entry.display_name,version='grounding-dino-vrsbench-step1300',runtime_ms=(perf_counter()-started)*1000,
                    device=str(self.device),mode='real',confidence_reason='not available',validation_status='unvalidated',
                    warnings=warnings,
                    parameters=dict(options,prompt=prompt,box_threshold=threshold,text_threshold=.25,training_image_size=512,
                                    strict_load=self.load_report,device_fallback=self.device_fallback,preprocessing_notes=inputs.notes)))
        finally:self.unload()
