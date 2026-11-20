"""Ten-sample real Grounding DINO sanity check, not a benchmark."""
import sys
from pathlib import Path
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.eval_common import arguments, read_rows, real_model, sample_path, best_iou, save_report, no_samples


def draw_example(source, predicted, truth, target):
    from PIL import Image, ImageDraw
    with Image.open(source) as original: image=original.convert('RGB')
    canvas=Image.new('RGB',(image.width,image.height+24),'white')
    canvas.paste(image,(0,0))
    draw=ImageDraw.Draw(canvas)
    for box in truth: draw.rectangle(box,outline='lime',width=4)
    for index,box in enumerate(predicted['boxes']):
        draw.rectangle(box,outline='red',width=2)
        draw.text((box[0],max(0,box[1]-12)),f'P{index+1}: {predicted["scores"][index]:.2f}',fill='red')
    draw.text((4,image.height+5),'Green: ground truth | Red: prediction (uncalibrated)',fill='black')
    target.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(target)


def main(argv=None):
    args=arguments(__doc__,'grounding','model2_eval.md',argv)
    rows=read_rows(args.samples/'boxes.json')
    if not rows:return no_samples(args,'Model 2 grounding sanity check')
    from backend.app.schemas import GroundingInput
    model=real_model('grounding',args.device)
    results=[]
    for index,row in enumerate(rows,1):
        started=perf_counter();record=dict(index=index,image=row['image'],phrase=row['phrase'],truth=row['boxes'])
        try:
            path=sample_path(args.samples,row['image'])
            result=model.run(GroundingInput(image_path=str(path),text_prompt=row['phrase'],box_threshold=.3))
            assert result.metadata.mode=='real'
            prediction=result.evidence.model_dump(mode='json')
            overlap=best_iou(prediction['boxes'],row['boxes'])
            target=args.output.parent/'model2_examples'/f'{index:02d}_{Path(row["image"]).stem}.png'
            draw_example(path,prediction,row['boxes'],target)
            record.update(predicted=prediction,iou=overlap,acc_at_05=overlap>=.5,boxes=len(prediction['boxes']),
                          example=f'[view]({target.parent.name}/{target.name})',device=result.metadata.device,
                          metadata=result.metadata.model_dump(mode='json'))
        except Exception as exc:
            record.update(error=f'{type(exc).__name__}: {exc}',iou=0.,acc_at_05=False,boxes=0)
        finally:model.unload()
        record['seconds']=round(perf_counter()-started,3);results.append(record)
        print(f'{index}/{len(rows)} {row["image"]}: IoU={record["iou"]:.4f}, boxes={record["boxes"]}, seconds={record["seconds"]}, error={record.get("error")}',flush=True)
    failures=sum('error' in r for r in results)
    save_report(args.output,'Model 2: 10-sample grounding sanity check',
        'The trainer supplied no benchmark accuracy metrics. This local 10-sample sanity check is not a benchmark. '
        'IoU is the best overlap across all retained predicted boxes and ground-truth boxes for each prompt '
        '(oracle best-box selection, not highest-score selection). No predictions means IoU=0; errors count as misses. '
        'Acc@0.5 is the fraction of prompts whose best IoU is at least 0.5. This does not penalize extra false-positive '
        'boxes and is not mAP, precision or recall. Both prediction and reference coordinates use original absolute pixels. '
        'Box threshold=0.3, text threshold=0.25; scores are uncalibrated. Green boxes are reference; red boxes are predictions. '
        'Timings include lazy load, inference, cleanup and example rendering.',results,
        {'samples':len(rows),'errors':failures,'mean_IoU':round(sum(r['iou'] for r in results)/len(rows),6),
         'Acc@0.5':f'{sum(r["acc_at_05"] for r in results)}/{len(rows)} ({sum(r["acc_at_05"] for r in results)/len(rows):.1%})',
         'total_seconds':round(sum(r['seconds'] for r in results),3)},
        [('index','#'),('image','Image'),('phrase','Phrase'),('iou','Best IoU'),('acc_at_05','IoU ≥ 0.5'),
         ('boxes','Predicted boxes'),('seconds','Seconds'),('example','Overlay'),('error','Error')])
    return int(bool(failures))


if __name__=='__main__':raise SystemExit(main())
