# Face Recognition Based on Center Bias Estimation and Adaptive Margin

Official github repository for CenterBiasFace. 

> **Abstract:**  The design of the loss function is crucial in deep face recognition. A common practice is to add a fixed margin term to all classes to modify the decision boundary between classes, compress the distance between intra-class features, and improve the ability of the model to separate features of different classes. However, adding the same margin term for all classes may ignore the inconsistency between classes in the face recognition dataset. In order to further improve the effectiveness of the model, we argue that the model should pay different attention to the samples of different classes according to the learning difficulty of the class. In this paper, we introduce a method for hard class mining based on the bias between the center of the class mean and the center of the class weight, called center bias estimation. The method proposed in this paper adaptively assigns margin terms of different sizes to different classes according to the value of center bias estimation. At the same time, to solve the problem of unstable calculation of center bias estimation in the early stage of training, we propose an adaptively changing convergence parameter to adjust the credibility of center bias estimation and design relevant experiments to prove the effectiveness of the convergence parameters. In the face verification baseline dataset, the proposed method in this paper is improved by 0.26% on average compared with the baseline method. In two large face verification test datasets, our method is improved by 0.58% and 0.22%, respectively, and multiple experimental results show that our method is better than the general existing algorithms. 

## Installation

```bash
conda create -n center_bias python=3.8
conda activate center_bias
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Train

### Download Dataset

Download the dataset from [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)(**MS1MV2**: MS1M-ArcFace and **MS1MV3**: MS1M-RetinaFace), the storage directory is arbitrary, here it is represented as {source_dataset_dir}, run the script under tools directory to convert the .rec training dataset and .bin validation dataset to .jpg format.

```shell
python ./tools/mx_recordio_to_images.py --root_dir {source_dataset_dir} --output_dir ./work_dirs/dataset/{fmt}
python ./tools/convert_bin_to_images.py --root_dir {source_dataset_dir} --output_dir ./work_dirs/dataset/{fmt}/val
```

[NOTE] If it is **MS1MV2** dataset, {fmt} should be replaced by ms1mv2, if it is **MS1MV3** dataset, {fmt} should be replaced by ms1mv3

### Run Training Scripts

```shell
chmod 776 ./scripts/*
```

If you only have a single GPU, run:

```shell
./scripts/train.sh --batch_size 512
```

If you have multiple GPUs, run:

```shell
./scripts/ddp_train.sh --batch_size 512
```

## Results

Training on MS1MV2 with IResNet18 in two RTX 3090:

| AgeDB | CAL-FW | CFP-FF | CFP-FP | CPL-FW | LFW   | AVG   | IJB-B  TPR(FPR=0.01%) | IJB-C  TPR(FPR=0.01%) |
| :---- | :----- | :----- | :----- | :----- | :---- | :---- | :-------------------- | :-------------------- |
| 97.57 | 95.70  | 99.74  | 94.69  | 90.00  | 99.57 | 96.21 | 85.74                 | 91.21                 |

Training on MS1MV3 with IResNet18 in two RTX 3090:

| AgeDB | CAL-FW | CFP-FF | CFP-FP | CPL-FW | LFW   | AVG   | IJB-B  TPR(FPR=0.01%) | IJB-C  TPR(FPR=0.01%) |
| :---- | :----- | :----- | :----- | :----- | :---- | :---- | :-------------------- | :-------------------- |
| 97.50 | 95.82  | 99.79  | 96.24  | 90.72  | 99.63 | 96.62 | 88.47                 | 92.29                 |
