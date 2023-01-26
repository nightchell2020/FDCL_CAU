# Integrated Diagnostic Model of High Speed and Low Speed Bearing of Large Wind Power Generators

### Injae Lee


## About Code
Collaborative Research of Doosan&CAU

You have to modify `config.py`to select mode as Auto-Encoder or WDCNN
### 1. Environment setup
This code has been tested on Ubuntu 20.04, Python 3.8.3, Pytorch 1.6.0, CUDA 11.3.

Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```
**Note** Also check your data-path in `config.py` 

### 2. Test & Evaluation


```bash 
cd tools
python test.py                                
	--dataset A                 #dataset_name
	--snapshot snapshot/best_model.pth  # tracker_name
```
The testing result will be saved in the `results/dataset_name/tracker_name` directory.

### 3. Train

#### Prepare training datasets

Used datasets：
* [A]
* [B]
* [C]



**Note:** `train_dataset/dataset_name/readme.md` has listed detailed operations about how to generate training datasets.


#### Train a model
To train the SiamAPN model, run `train.py` with the desired configs:

```bash
cd tools
python train.py
```

### 4. Contact
If you have any questions, please contact me.

IPIS

Injae Lee

Email: [injea0908@gmail.com](injea0908@gmail.com)


## Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot). We would like to express our sincere thanks to the contributors.