## Key Point-aware Occlusion Suppression and Semantic Alignment for Occluded Person Re-identification
## 
* **Introduction**: We proposed a Key-Point-aware Occlusion suppression and Semantic alignment (POS)
method for occluded person re-ID.


## Properties
* **Dataset**: Support multiple Datasets
  Please obtain the data set according to [PGFA](https://ieeexplore.ieee.org/document/9010704)
* **Key Points Detection**: You can download the Hrnet pertrain model from [here](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC) for pose estimation.
# Requirements:
    CUDA  10.2
    Python  3.8
    Pytorch  1.6.0
    torchvision  0.2.2
    numpy  1.19.0


## Train and Test
#### Train on Occluded DukeMTMC/Market-1501/DukeMTMC-reID
```
python3 main.py --mode train \
    --train_dataset your_train_dataset_name --test_dataset your_test_dataset_name \
    --market(or_duke_or_occduke)_path /path/to/market/dataset/ \
    --output_path ./results/your_dataset_name

```
```
python3 main.py --mode test \
    --train_dataset your_train_dataset_name --test_dataset your_test_dataset_name \
    --market(or_duke_or_occduke)_path /path/to/market/dataset/ \
    --output_path ./results/your_dataset_name

```
## Cite

```

Corresponding papers will appear in Information Science，2022.

```
If you have any questions, please contact us with 1216246628@qq.com

