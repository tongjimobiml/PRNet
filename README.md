# PRNet

PRNet propose a deep neural network (DNN)-based position recovery framework, which can ensemble the power of CNN, sequence model LSTM, and two attention mechanisms to learn local, short- and long-term spatio-temporal dependencies from input MR samples.

```
@inproceedings{10.1145/3357384.3357908,
author = {Zhang, Yige and Rao, Weixiong and Zhang, Kun and Yuan, Mingxuan and Zeng, Jia},
title = {PRNet: Outdoor Position Recovery for Heterogenous Telco Data by Deep Neural Network},
year = {2019},
isbn = {9781450369763},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3357384.3357908},
doi = {10.1145/3357384.3357908},
booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
pages = {1933–1942},
numpages = {10},
keywords = {deep neurual network, outdoor position recovery, telecommunication big data},
location = {Beijing, China},
series = {CIKM '19}
}
```

## Usage

How to run the code:

Start an experiment by:

``python PRNet_xx_xx.ipynb``

## Dataset

The dataset contains 2G and 4G telco data from two district: Jiading and Siping. All these data can be found in ``data`` file.
