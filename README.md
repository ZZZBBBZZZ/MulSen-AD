# MulSen-AD: A Dataset and Benchmark for Multi-Sensor Anomaly Detection
## [[Project Page](https://zzzbbbzzz.github.io/MulSen_AD/index.html)]



## Abstract
> Although remarkable progress has been made in object-level anomaly detection for data captured by a single type of sensor, two critical problems still hinder the deployment of the algorithms. First, in  practical industrial environments, anomalies are often long-tailed and complex, making it inherently difficult for a single type of sensor to detect certain types of defects. For instance, color defects are invisible in 3D laser scanner, while subtle texture defects are also difficult to detect by infrared thermography sensor. Second, environment conditions, such as ambient lighting in factories, often greatly affect the detection process, leading to unreliable or invalid results under a single type of sensor. To address these issues, we propose the first Multi-Sensor Anomaly Detection (Mulsen-AD) dataset and develop a comprehensive benchmark (Mulsen-AD Bench) on Mulsen-AD dataset. Specifically,  we build Mulsen-AD dataset with high-resolution industrial camera, high-precision laser scanner and lock-in infrared thermography. To exactly replicate real industrial scenarios, Mulsen-AD dataset is comprised of 15 types of distinct and authentic industrial products, featuring a variety of defects that require detection through the integration of multiple sensors. Additionally, we propose Mulsen-TripleAD algorithm, a decision level gating method, to solve the problem of object anomaly detection with the combination of three sensors. In our benchmark, our method achieved 81.1\% object-level AUROC detection accuracy using three sensors, significantly outperforming single-sensor methods, demonstrating the potential value of multi-sensor datasets for the research community.




## Download

### Dataset

To download the `MulSen-AD` dataset, click [MulSen_AD.rar(google drive)](https://drive.google.com/file/d/16peKMQ6KYnPK7v-3rFZB3aIHWdqNtQc5/view?usp=drive_link) or [MulSen_AD.rar(huggingface)](https://huggingface.co/datasets/orgjy314159/MulSen_AD/tree/main).

The dataset can be quickly reviewed on the [website](https://zzzbbbzzz.github.io/MulSen_AD/index.html).

After download, put the dataset in `dataset` folder.

### Checkpoint

To download the pre-trained PointMAE model using [this link](https://drive.google.com/file/d/1-wlRIz0GM8o6BuPTJz4kTt6c_z1Gh6LX/view?usp=sharing). 

After download, put the dataset in `checkpoints` folder.


## Data preparation
- Download MulSen_AD.rar and extract into `./dataset`
```
MulSen_AD
├── ashtray0
    ├── train
        ├── *template.pcd
        ...
    ├── test
        ├── 1_bulge.pcd
        ├── 2_concavity.pcd
        ...
    ├── GT
        ├── 1_bulge.txt
        ├── 2_sink.txt
        ... 
├── bag0
...
```
