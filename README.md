# MulSen-AD: A Dataset and Benchmark for Multi-Sensor Anomaly Detection
## [[Project Page](https://zzzbbbzzz.github.io/MulSen_AD/index.html)]



## 1.Introduction
> In this project, we propose the first Multi-Sensor Anomaly Detection (Mulsen-AD) dataset and develop a comprehensive benchmark (Mulsen-AD Bench) on Mulsen-AD dataset. Specifically,  we build Mulsen-AD dataset with high-resolution industrial camera, high-precision laser scanner and lock-in infrared thermography. To exactly replicate real industrial scenarios, Mulsen-AD dataset is comprised of 15 types of distinct and authentic industrial products, featuring a variety of defects that require detection through the integration of multiple sensors. Additionally, we propose Mulsen-TripleAD algorithm, a decision level gating method, to solve the problem of object anomaly detection with the combination of three sensors.

![piplien](./img/figure1_min.png)
- (a) is Existing single-sensor object-level anomaly detection.
- (b) is our introduced multi-sensor object-level anomaly detection setting.

The main contributions are summarized as follows:
-  We introduced Multi-Sensor Anomaly Detection (**Mulsen-AD**) setting, a practical and challenging setting for object level anomaly detection based on three different industrial sensors, solving the limitations of single sensor anomaly detection, and bridged the gap between industrial and research in object anomaly detection.
-  We developed **Mulsen-AD** dataset, the first real dataset for evaluating multi-sensor anomaly detection, featuring diverse, high-quality object data captured by high-resolution devices and anomalies based on actual factory conditions.
-  We conducted a comprehensive benchmark using the Mulsen-AD dataset and provided a universal toolkit to facilitate further exploration and reproduction of the benchmark.
-  We proposed **Mulsen-TripleAD**, a decision fusion gating method for multi-sensor anomaly detection. Utilizing data from three types of sensors and combining multiple memory banks with a decision gating unit, *Mulsen-TripleAD* outperforms setups using fewer sensors and sets a new baseline for our multi-sensor anomaly detection task.


## 2. Mulsen AD dataset

### 2.1 Collection pipeline
MulSen-AD includes **RGB images** acquired by cameras, **infrared images**(gray-scale images) by lock-in infrared thermography and **high-resolution 3D point clouds** by laser scanners.
![piplien](./img/figure1_min.png)

### 2.2 Meet our 15 categories
We selected 15 objects made by different materials, including metal, plastic, fiber, rubber, semiconductor and composite materials, with different shapes, sizes and colors.
![piplien](./img/figure1_min.png)

### 2.3 Anomaly types and samples
we manually created 14 types of anomalies, including cracks, holes, squeeze, external and internal broken, creases, scratches, foreign bodies, label, bent, color, open, substandard, and internal detachments. The anomalies are designed to closely resemble real industrial situations, with a wide distribution of types, including surface, internal, and 3D geometric anomalies. 

Capsule：

![Capsule](./img/capsule.PNG)

Light：

![Light](./img/light.PNG)

Cotton：

![Light](./img/cotton.PNG)

*More samples can be found on the [website](https://zzzbbbzzz.github.io/MulSen_AD/index.html).

### 2.4 Data Directory
- Download [MulSen_AD.rar](https://drive.google.com/file/d/16peKMQ6KYnPK7v-3rFZB3aIHWdqNtQc5/view?usp=drive_link) and extract into `./dataset/MulSen_AD`
```
MulSen_AD
├── capsule                              ---Object class folder.
    ├── RGB                              ---RGB images
        ├── train                        ---A training set of RGB images
            ├── 0.png
            ...
        ├── test                         ---A test set of RGB images
            ├── hole                     ---Types of anomalies, such as hole. 
                ├── 0.png
                ...
            ├── crack                    ---Types of anomalies, such as crack.
                ├── 0.png
                ...
            ├── good                     ---RGB images without anomalies.
                ├── 0.png
                ...
            ...
        ├── GT                           ---GT segmentation mask for various kinds of anomalies.
            ├── hole
                ├── 0.png
                ├── data.csv             ---Label information
                ...
            ├── crack
                ├── 0.png
                ├── data.csv
                ...
            ├── good
                ├── data.csv
            ...
        ...
    ├── Infrared                        ---Infrared images
        ├── train
        ├── test
        ├── GT
    ├── Pointcloud                      ---Point Clouds
        ├── train
        ├── test
        ├── GT
├── cotton                             ---Object class folder.                      
    ... 
...
```

## 3. Benchmark
### 3.1 MulsenAD Bench on Mulsen AD dataset

The score indicates object-level AUROC ↑. PC means Point cloud. The best result of each category is highlighted in bold.

| Category          | RGB   | Infrared | PC    | RGB+Infrared | PC+RGB | PC+Infrared | PC+RGB+Infrared |
|-------------------|-------|----------|-------|--------------|--------|-------------|------------------|
| Capsule           | 0.927 | 0.754    | 0.625 | 0.923        | 0.942  | 0.775       | **0.958**        |
| Cotton            | 0.720 | 0.818    | 0.795 | 0.830        | 0.742  | **0.835**   | 0.762            |
| Cube              | 0.947 | 0.960    | 0.338 | **0.978**    | 0.947  | 0.949       | **0.978**        |
| Spring pad        | **0.900** | 0.542    | 0.446 | 0.575        | 0.875  | 0.538       | 0.712            |
| Screw             | 0.148 | **0.910**| 0.465 | 0.794        | 0.152  | 0.890       | 0.639            |
| Screen            | 0.412 | 0.972    | 0.612 | 0.912        | 0.444  | **0.984**   | 0.872            |
| Piggy             | 0.980 | 0.903    | 0.537 | 0.983        | 0.980  | 0.903       | **0.997**        |
| Nut               | **0.814** | 0.103    | 0.483 | 0.138        | 0.790  | 0.103       | 0.252            |
| Flat pad          | **0.813** | 0.417    | 0.747 | 0.543        | **0.813**  | 0.400       | 0.610            |
| Plastic cylinder  | 0.843 | **1.000**| 0.579 | **1.000**    | 0.821  | **1.000**   | **1.000**        |
| Zipper            | 0.637 | 0.982    | 0.595 | 0.946        | 0.699  | **1.000**   | 0.943            |
| Button cell       | **1.000** | 0.797    | 0.468 | 0.923        | **1.000**  | 0.790       | 0.971            |
| Tooth brush       | 0.587 | 0.852    | 0.875 | 0.852        | 0.670  | **0.898**   | 0.822            |
| Solar panel       | 0.528 | **0.923**| 0.374 | 0.821        | 0.482  | 0.877       | 0.836            |
| Light             | 0.814 | **0.839**| 0.244 | 0.814        | 0.789  | **0.839**   | 0.808            |
| **Average**       | 0.738 | 0.776    | 0.546 | 0.802        | 0.743  | 0.785       | **0.811**        |


### 3.2 SingeBench-3D for MulSen-AD dataset

The score indicates object-level AUROC ↑. The best result of each category is highlighted in bold.

| Category          | BTF (Raw) | BTF (FPFH) | M3DM (PointMAE) | M3DM (PointBERT) | PatchCore (FPFH) | PatchCore (FPFH+Raw) | PatchCore (PointMAE) | IMRNet | Reg3D-AD |
|-------------------|-----------|------------|-----------------|------------------|------------------|----------------------|----------------------|--------|----------|
| Capsule           | 0.641     | 0.874      | 0.773           | 0.728            | 0.897            | **0.900**            | 0.871                | 0.601  | **0.900** |
| Cotton            | **0.775** | 0.320      | 0.640           | 0.490            | 0.360            | 0.335                | 0.455                | 0.585  | 0.425    |
| Cube              | 0.603     | 0.655      | 0.462           | 0.438            | **0.688**        | 0.657                | 0.557                | 0.432  | 0.449    |
| Spring pad        | 0.764     | 0.872      | 0.698           | 0.517            | 0.986            | **1.000**            | 0.965                | 0.651  | 0.951    |
| Screw             | 0.933     | 0.698      | 0.788           | 0.890            | **1.000**        | **1.000**            | 0.705                | 0.742  | 0.759    |
| Screen            | 0.584     | 0.788      | 0.878           | 0.947            | **0.950**        | **0.950**            | 0.478                | 0.378  | 0.546    |
| Piggy             | 0.818     | 0.831      | 0.030           | 0.040            | **1.000**        | **1.000**            | 0.702                | 0.729  | 0.831    |
| Nut               | 0.789     | 0.883      | 0.806           | 0.783            | 0.980            | **0.989**            | 0.837                | 0.812  | 0.860    |
| Flat pad          | 0.698     | 0.918      | 0.882           | 0.882            | **0.984**        | 0.979                | 0.969                | 0.714  | 0.939    |
| Plastic cylinder  | 0.728     | 0.493      | 0.409           | 0.891            | **0.905**        | 0.692                | 0.590                | 0.621  | 0.720    |
| Zipper            | 0.505     | 0.662      | 0.712           | 0.613            | **0.909**        | 0.871                | 0.747                | 0.630  | 0.695    |
| Button cell       | 0.567     | 0.500      | 0.621           | 0.603            | **0.874**        | 0.754                | 0.782                | 0.702  | 0.715    |
| Toothbrush        | 0.882     | 0.562      | 0.816           | 0.812            | 0.885            | **0.895**            | 0.822                | 0.615  | 0.809    |
| Solar panel       | 0.474     | 0.531      | 0.476           | 0.423            | 0.569            | 0.555                | 0.483                | 0.344  | **0.586** |
| Light             | 0.903     | 0.859      | 0.608           | 0.512            | **1.000**        | **1.000**            | 0.672                | 0.457  | 0.697    |
| **Mean**          | 0.711     | 0.721      | 0.646           | 0.606            | **0.865**        | 0.853                | 0.716                | 0.601  | 0.726    |

The score indicates point-level AUROC ↑. The best result of each category is highlighted in bold.
| Category          | BTF (Raw) | BTF (FPFH) | M3DM (PointMAE) | M3DM (PointBERT) | PatchCore (FPFH) | PatchCore (FPFH+Raw) | PatchCore (PointMAE) | IMRNet | Reg3D-AD |
|-------------------|-----------|------------|-----------------|------------------|------------------|----------------------|----------------------|--------|----------|
| Capsule           | 0.641     | 0.874      | 0.773           | 0.728            | 0.897            | **0.900**            | 0.871                | 0.601  | **0.900** |
| Cotton            | **0.775** | 0.320      | 0.640           | 0.490            | 0.360            | 0.335                | 0.455                | 0.585  | 0.425    |
| Cube              | 0.603     | 0.655      | 0.462           | 0.438            | **0.688**        | 0.657                | 0.557                | 0.432  | 0.449    |
| Spring pad        | 0.764     | 0.872      | 0.698           | 0.517            | 0.986            | **1.000**            | 0.965                | 0.651  | 0.951    |
| Screw             | 0.933     | 0.698      | 0.788           | 0.890            | **1.000**        | **1.000**            | 0.705                | 0.742  | 0.759    |
| Screen            | 0.584     | 0.788      | 0.878           | 0.947            | **0.950**        | **0.950**            | 0.478                | 0.378  | 0.546    |
| Piggy             | 0.818     | 0.831      | 0.030           | 0.040            | **1.000**        | **1.000**            | 0.702                | 0.729  | 0.831    |
| Nut               | 0.789     | 0.883      | 0.806           | 0.783            | 0.980            | **0.989**            | 0.837                | 0.812  | 0.860    |
| Flat pad          | 0.698     | 0.918      | 0.882           | 0.882            | **0.984**        | 0.979                | 0.969                | 0.714  | 0.939    |
| Plastic cylinder  | 0.728     | 0.493      | 0.409           | 0.891            | **0.905**        | 0.692                | 0.590                | 0.621  | 0.720    |
| Zipper            | 0.505     | 0.662      | 0.712           | 0.613            | **0.909**        | 0.871                | 0.747                | 0.630  | 0.695    |
| Button cell       | 0.567     | 0.500      | 0.621           | 0.603            | **0.874**        | 0.754                | 0.782                | 0.702  | 0.715    |
| Toothbrush        | 0.882     | 0.562      | 0.816           | 0.812            | 0.885            | **0.895**            | 0.822                | 0.615  | 0.809    |
| Solar panel       | 0.474     | 0.531      | 0.476           | 0.423            | 0.569            | 0.555                | 0.483                | 0.344  | **0.586** |
| Light             | 0.903     | 0.859      | 0.608           | 0.512            | **1.000**        | **1.000**            | 0.672                | 0.457  | 0.697    |
| **Mean**          | 0.711     | 0.721      | 0.646           | 0.606            | **0.865**        | 0.853                | 0.716                | 0.601  | 0.726    |

## 4. Download

### 4.1 Dataset

To download the `MulSen-AD` dataset, click [MulSen_AD.rar(google drive)](https://drive.google.com/file/d/16peKMQ6KYnPK7v-3rFZB3aIHWdqNtQc5/view?usp=drive_link) or [MulSen_AD.rar(huggingface)](https://huggingface.co/datasets/orgjy314159/MulSen_AD/tree/main).

The dataset can be quickly reviewed on the [website](https://zzzbbbzzz.github.io/MulSen_AD/index.html).

After download, put the dataset in `dataset` folder.

### 4.2 Checkpoint

To download the pre-trained PointMAE model using [this link](https://drive.google.com/file/d/1-wlRIz0GM8o6BuPTJz4kTt6c_z1Gh6LX/view?usp=sharing). 

After download, put the checkpoint file in `checkpoints` folder.




## 5. Getting Started


### 5.1 Installation
To start, I recommend to create an environment using conda:
```
conda create -n MulSen_AD python=3.8
conda activate MulSen_AD
```

Clone the repository and install dependencies:
```
$ git clone https://github.com/ZZZBBBZZZ/MulSen-AD.git
$ cd MulSen-AD
$ pip install -r requirements.txt
```  


## Train and Test
Firstly, please ensure that the dataset and checkpoints have been downloaded and placed in the corresponding folders. The file format is like this:
```
checkpoints
 └ pointmae_pretrain.pth
data
 └ model
```


Train and test with the following command:
```
$ sh start.sh
```

## Thanks

Our code is built on [Real3D-AD](https://github.com/eliahuhorwitz/3D-ADS) and [M3DM](https://github.com/nomewang/M3DM), thanks for their excellent works!

## License
The dataset is released under the CC BY 4.0 license.
