"""In-process CPU TorchScript SEN12MS fusion specialist."""
from importlib.util import find_spec
from time import perf_counter

from .base import SpecialistModel
from .device import free_cache
from ..preprocessing import PreparedInputs, prepare_for
from ..preprocessing.fusion import load_fusion_artifacts
from ..schemas import FusionInput,FusionOutput,StandardResult,Metadata,Confidence

FUSION_WARNINGS=[
    'Real adapter enabled before genuine-pair verification; local real-pair validation is still pending.',
    'Trained only on SEN12MS summer data: eight classes and 256x256 patches.',
    'Teammate-reported test accuracy is about 76% and macro-F1 about 63%; these results were not independently reproduced here.',
    'Grassland, Forest and Wetlands are weak classes; other regions or seasons may perform worse.',
    'Fusion confidence is an uncalibrated softmax score, not a guarantee of correctness.',
]


class FusionModel(SpecialistModel):
    accepts_prepared_inputs=True
    required_files=('model4_fusion_torchscript.pt','model4_normalization_stats.json','metadata.json')

    @staticmethod
    def available():return find_spec('torch') is not None

    def __init__(self,*,entry,weights_path):
        self.entry,self.weights_path=entry,weights_path
        self.model=self.torch=None
        self.class_names=None

    @property
    def is_loaded(self):return self.model is not None

    def load(self):
        if self.is_loaded:return
        import torch
        self.torch=torch
        try:
            _,metadata=load_fusion_artifacts(self.weights_path)
            self.class_names=metadata['class_names']
            self.model=torch.jit.load(str(self.weights_path/'model4_fusion_torchscript.pt'),map_location='cpu').eval()
        except Exception:
            self.unload();raise

    def unload(self):
        self.model=self.class_names=None
        if self.torch:free_cache(self.torch)

    def _predict(self,prepared):
        optical=self.torch.from_numpy(prepared.tensors['optical']).to(device='cpu',dtype=self.torch.float32)
        sar=self.torch.from_numpy(prepared.tensors['sar']).to(device='cpu',dtype=self.torch.float32)
        if tuple(optical.shape)!=(1,13,256,256) or tuple(sar.shape)!=(1,2,256,256):
            raise ValueError('Fusion model-native tensors must be [1,13,256,256] and [1,2,256,256].')
        with self.torch.inference_mode():
            logits=self.model(optical,sar)
            if tuple(logits.shape)!=(1,8) or not self.torch.isfinite(logits).all():
                raise RuntimeError('Fusion returned invalid logits; expected finite [1,8] output.')
            probabilities=logits.softmax(dim=1)[0]
            values,indices=self.torch.topk(probabilities,3)
        top3=[(self.class_names[int(index)],float(value)) for index,value in zip(indices,values)]
        return FusionOutput(label=top3[0][0],confidence=top3[0][1],top3=top3)

    def fuse_predict(self,optical_path,sar_path):
        prepared=prepare_for('fusion',[optical_path,sar_path],artifact_dir=self.weights_path)
        try:
            self.load();return self._predict(prepared).model_dump()
        finally:self.unload()

    def run(self,inputs,query='',options=None):
        started=perf_counter()
        try:
            if not isinstance(inputs,PreparedInputs):
                item=FusionInput.model_validate(inputs)
                inputs=prepare_for('fusion',[item.optical_path,item.sar_path],artifact_dir=self.weights_path)
            if inputs.model_id!='fusion' or not any(note.get('operation')=='fusion_raw_normalization' for note in inputs.notes):
                raise ValueError('Fusion requires validated native GeoTIFF tensors, not previews or generic SAR normalization.')
            self.load();prediction=self._predict(inputs)
            return StandardResult(answer=prediction.label,evidence=prediction,
                confidence=Confidence(value=prediction.confidence,calibrated=False,note='uncalibrated softmax class probability'),
                metadata=Metadata(model_name=self.entry.display_name,version='sen12ms-summer-torchscript',runtime_ms=(perf_counter()-started)*1000,
                    device='cpu',mode='real',validation_status='reported_only',warnings=FUSION_WARNINGS,
                    parameters=dict(options or {},preprocessing_notes=inputs.notes,cpu_only=True,
                        classes=self.class_names,reported_test_accuracy=.7615,reported_test_macro_f1=.6289)))
        finally:self.unload()
