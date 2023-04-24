# FloCo
**Official implementation of the Towards Making Flowchart Images Machine Interpretable paper (ICDAR 2023)**

[Paper]() | [Project Page]() | [Poster]() | [Short Talk]() | [Slides]()

## Requirements
* Use **python >= 3.10.8**. Conda recommended : [https://docs.anaconda.com/anaconda/install/linux/](https://docs.anaconda.com/anaconda/install/linux/)

* Use **pytorch 1.13.1 CUDA 11.6**

* Other requirements from 'requirements.txt'

**To setup environment**
```
# create new env flow
$ conda create -n flow python=3.10.8

# activate flow
$ conda activate flow

# install pytorch, torchvision
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# install other dependencies
$ pip install -r requirements.txt
```

## Training

### Preparing dataset
- Download the dataset [here](https://drive.google.com/file/d/155hBKldZijCCK8tAOQyT8OaG7KXOvbl2/view?usp=share_link) and unzip it.
- The dataset directory should have the following structure:
```
[FloCo]
├── Train
│   ├── codes
│   ├── flowchart
│   ├── png
│   └── svg
├── Validation
│   ├── codes
│   ├── flowchart
│   ├── png
│   └── svg
└── Test
    ├── codes
    ├── flowchart
    ├── png
    └── svg
```

### Generating sequence embeddings
- Encode flowchart images into sequential embeddings for each of train, validation and test sets separately
```
# Set path to the folder containing the png flowchart images
# Set path to text file to save the encodings
$ python generate_encodings.py
```
- The dataset directory should now look like:
```
[FloCo]
├── Train
│   ├── codes
│   ├── flowchart
│   ├── png
│   ├── svg
│   └── encodings.txt
├── Validation
│   ├── codes
│   ├── flowchart
│   ├── png
│   ├── svg
│   └── encodings.txt
└── Test
    ├── codes
    ├── flowchart
    ├── png
    ├── svg
    └── encodings.txt
```

### Pre-train the model architecture
```
# Set path to augmented python codes and train set flowchart encodings 
# Set path to save model checkpoints and train logs
$ python pre-training.py
```
### Fine-tune the pre-trained model
```
# Set path to training and validation flowchart encodings and python codes respectively 
# Set path to save model checkpoints and train logs
$ python fine-tuning.py
```

## Inference
- Generate python codes for unseen flowchart images using best checkpoints of the trained model
```
# Set path to flowchart encodings and python codes belonging to the test dataset 
# Define path to the best checkpoint saved above
# Set path to save the generated codes
$ python inference.py
```

## Cite us
- If you find this work useful for your research, please consider citing.
```
@inproceedings{shukla2023floco,
  author    = "Shukla, Shreya and 
              Gatti, Prajwal and 
              Kumar, Yogesh and
              Yadav, Vikash and
              Mishra, Anand",
  title     = "Towards Making Flowchart Images Machine Interpretable",
  booktitle = "ICDAR",
  year      = "2023",
}
```

## Acknowledgements
This repo uses scripts from https://github.com/salesforce/CodeT5/tree/main/evaluator/CodeBLEU to compute BLEU and CodeBLEU scores. 

Code provided by https://huggingface.co/Salesforce/codet5-small helped in implementing FloCo-T5.
