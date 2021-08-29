YoloR gives great result without even augmentation:

```
Epoch   gpu_mem       box       obj       cls     total   targets  img_size
49/49     2.73G   0.03137   0.01865         0   0.05002         0       448: 100% 43/43 [00:09<00:00,  4.62it/s]

 Class      Images     Targets           P           R          mAP@.5  mAP@.5:.95
   all          96           138             0.681       0.833       0.831       0.408
```

Note the generated `best.pt` model was 148 MB so is not included in this repo