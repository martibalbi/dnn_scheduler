# dnn_scheduler

`dnn_scheduling.ipynb`: notebook containing normal PyTorch model 

`scheduling_functions.ipynb`: notebook to generate dataset

`models/dnn_scheduler.py`: proposed model for the MAX78000

`datasets/tasa.py`: data loader for the TASA dataset -> `database.h5` can be downloaded from here: https://drive.google.com/file/d/12mhUPZLrKCp1rZLf-t7YrK7NIkIrlXVK/view?usp=sharing

`ai8x-synthesis/best.pth.tar`: best checkpoint file generated when training running `python train.py --lr 0.01 --optimizer ADAM --epochs 200 --deterministic --compress policies/schedule.yaml --model dnn_scheduler --dataset TASA  --param-hist --device MAX78000 --regression --workers=0`
  - note: --workers has to be 0 or else the data loading fails

`ai8x-synthesis/dnn_q8.pth.tar`: quantized weights generated by running `python quantize.py .\dnn\best.pth.tar .\dnn\dnn_q8.pth.tar --device MAX78000 --scale 1 --clip-method SCALE`

`tests/sample_dnn_scheduler.npy`: random input

evaluation fails when running `python train.py --model dnn_scheduler --dataset TASA --evaluate --exp-load-weights-from ../ai8x-synthesis/dnn_q8.pth.tar -8 --device MAX78000 --workers=0`:

```
{
Log file for this run: C:\dnn_scheduling\ai8x-training\logs\2024.03.28-113307\2024.03.28-113307.log
Traceback (most recent call last):
  File "train.py", line 1901, in <module>
    main()
  File "train.py", line 540, in main
    return evaluate_model(model, criterion, test_loader, pylogger, activations_collectors,
  File "train.py", line 1654, in evaluate_model
    top1, _, _, mAP = test(test_loader, model, criterion, loggers, activations_collectors,
  File "train.py", line 1033, in test
    top1, top5, vloss, mAP = _validate(test_loader, model, criterion, loggers, args)
  File "train.py", line 1275, in _validate
    loss = criterion(output, target)
  File "C:\Users\MBalbi2\AppData\Local\Programs\Python\Python38\lib\site-packages\torch\nn\modules\module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "C:\Users\MBalbi2\AppData\Local\Programs\Python\Python38\lib\site-packages\torch\nn\modules\loss.py", line 1047, in forward
    return F.cross_entropy(input, target, weight=self.weight,
  File "C:\Users\MBalbi2\AppData\Local\Programs\Python\Python38\lib\site-packages\torch\nn\functional.py", line 2693, in cross_entropy
    return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)
  File "C:\Users\MBalbi2\AppData\Local\Programs\Python\Python38\lib\site-packages\torch\nn\functional.py", line 2388, in nll_loss
    ret = torch._C._nn.nll_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
RuntimeError: 1D target tensor expected, multi-target not supported
}
```
