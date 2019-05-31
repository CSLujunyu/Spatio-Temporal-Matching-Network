# STM
## Overview
Implementation of the model described in the ACL'19 short paper:
- Interpretive Spatio-Temporal Features for Multi-TurnResponses Selection by Junyu Lu, Chenbin Zhang, Zeying Xie, Guang Ling, Chao Zhou, Zenglin Xu
```
(update soon)
```

Files in this repository:

* ```dataset``` folder: the dataset.
* ```tmp``` folder: path to save log and models.
* ```cache``` folder: path to save pre-processed data after running at the first time.
* ```model.py``` folder: code of STM.
* ```main.py```: entrance.
* ```processor.py```: code for processing the Ubunutu Corpus and the Advising Dataset that release by DSTC7 Track1.
* ```STMDataLoader.py```: dataloader.

## Dataset
```dataset/ubuntu_train_subtask_1.json```, ```dataset/ubuntu_dev_subtask_1.json``` and ```dataset/ubuntu_test_subtask_1.json``` are the training, development and test sets, respectively. The format of them is as follows:

```
[
  {
        "data-split": "dev",
        "example-id": 1100001,
        "messages-so-far": [
            {
                "speaker": "participant_1",
                "utterance": "hey guys, does your livecd have chroot installed? and bash?"
            },
            {
                "speaker": "participant_2",
                "utterance": "sure"
            },
            {
                "speaker": "participant_1",
                "utterance": "does it have everything I need to format a partition ext2?. and ext3?"
            },
            {
                "speaker": "participant_2",
                "utterance": "yep"
            },
            {
                "speaker": "participant_1",
                "utterance": "yay I can use it to install gentoo. !"
            },
            {
                "speaker": "participant_2",
                "utterance": "lol. LOL"
            },
            {
                "speaker": "participant_1",
                "utterance": "=-). brb rebooting into ubuntu"
            },
            {
                "speaker": "participant_2",
                "utterance": "form last week:. 04:21:47] <findme> this is a big crowd here. [04:21:53] <findme> have all gentoo users moved here ?"
            },
            {
                "speaker": "participant_1",
                "utterance": "to bad its still using apt I would switch in a heart beat if it had its own package manager"
            }
        ],
       "options-for-correct-answers": [
            {
                "candidate-id": "TLSHF16Y4J4L",
                "utterance": "what are you missing in apt ?"
            }
        ],
        "options-for-next": [
            {
                "candidate-id": "YWOA49156J9P",
                "utterance": "issues with msn?. I'm experiencing them on windows atm, current msn version"
            },
            {
                "candidate-id": "RYBI7QRD9QZN",
                "utterance": "<> AmaroqWolf: alias='sudo admincommand'.  <AmaroqWolf>  aw, can't make myself type sudo? I like it better that way."
            },
            {
                "candidate-id": "TLSHF16Y4J4L",
                "utterance": "what are you missing in apt ?"
            },
            ...
       ],
       ...
]
```

## Usage
1. Download the subtask1 dataset from  and put it into dataset/
2. Train a model using default hyperparameters (Excuting at the first time may need a little time to prepare the data and save in ```cache/```).

```python main.py --save_path 'tmp/' --encoder_type 'GRU' --pretrain_embedding glove.42B.300d.txt```

3. Evaluated results are saved to ```tmp/eval_result.txt``` and ```tmp/test_result.txt``` respectively.

Note: You can change hyperparameters in ```main.py``` 
