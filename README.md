# UNIT-model-comparison
This project aim to make the data format of the UNIT align with https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

## Dataset format

You should check out the size of A domain images are the same with B domain image.

```
./[your own path]/dataset
----/trainA   % A domain training set image file [png]
----/trainB   % B domain training set image file [png]
----/testA    % A domain testing set image file [png]
----/testB    % B domain testing set image file [png]
```
## Model Training
```
python unit.py --dataroot [your own path] --dataset_name dataset --in_channel 3 --out_channel 1 --n_epochs 300 --batch_size 1
```

## Model Testing
```
python test.py --dataroot [your own path] --dataset_name dataset --in_channel 3 --out_channel 1 --n_epochs 300 --batch_size 1
```
