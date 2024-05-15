
# Breaking the Positive Sample Barrier in Person Re-Identification: Towards Domain Generalization without Paired Samples

## Pipeline
![framework](figs/1.png)

## Environment
```
pip install -r requirement.txt
```

* requirement:
```
Python 3.9.0
Pytorch 1.10.0 & torchvision 0.11.0
```

## Dataset Preparation

1. Download the datasets([Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565))and then unzip them to `your_dataset_dir`.
2. Split Market-1501 and MSMT to Market-SCT and MSMT-SCT according to [CCFP](https://github.com/g3956/CCFP).
3. Make new directories in data and organize them as follows:
<pre>
+-- data
|   +-- market1501
|       +-- boudning_box_train
|       +-- query
|       +-- boudning_box_test
|   +-- market1501_sct
|       +-- boudning_box_train_sct
|       +-- query
|       +-- boudning_box_test
|   +-- MSMT17
|       +-- train_sct
|       +-- test
|       +-- list_train.txt
|       +-- MSMT_mixSCT.txt
</pre>


## Train and test

train
```
CUDA_VISIBLE_DEVICES=0 python train.py --config-file configs/Market_SCT/vit_transreid.yml
```
test
```
CUDA_VISIBLE_DEVICES=0 python test.py --config-file configs/Market_SCT/vit_transreid.yml
```

## Contact

If you have any questions, please feel free to contact me(lyx520419@163.com).