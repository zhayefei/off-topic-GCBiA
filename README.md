# Off-Topic-GCBiA

Resources (model & inference code) of our paper at ACL2020: 
> [Gated Convolutional Bidirectional Attention-based Model for Off-topic Spoken Response Detection](https://www.aclweb.org/anthology/2020.acl-main.56/)

## Installation
The following command installs all necessary packages:
```.bash
pip install -r requirements.txt
```
The project was tested using Python 3.7.

## Released Model
The released GCBiA model is in the package `model.tar.gz`, please to extract it at first:
```
tar zxvf model.tar.gz
```
- `off-topic-gcbia` GCBiA model 
- `tokenizer` tokenizer of our model

## Model inference
To predict results on the input file using our model, please use the following command:
```
python predict.py --load-model MODEL_PATH \
                  --load-tokenizer TOKENIZER_PATH \
                  --input_file INPUT_FILE \
                  --output_file OUTPUT_FILE
```
- `THLD` in config.py the threshold of predict score, you can adjust it according to your application

## Citation
If you find this work is useful for your research, please cite our paper:
```
@inproceedings{zha-etal-2020-gated,
    title = "Gated Convolutional Bidirectional Attention-based Model for Off-topic Spoken Response Detection",
    author = "Zha, Yefei  and
      Li, Ruobing  and
      Lin, Hui",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.56",
    doi = "10.18653/v1/2020.acl-main.56",
    pages = "600--608",
    abstract = "Off-topic spoken response detection, the task aiming at predicting whether a response is off-topic for the corresponding prompt, is important for an automated speaking assessment system. In many real-world educational applications, off-topic spoken response detectors are required to achieve high recall for off-topic responses not only on seen prompts but also on prompts that are unseen during training. In this paper, we propose a novel approach for off-topic spoken response detection with high off-topic recall on both seen and unseen prompts. We introduce a new model, Gated Convolutional Bidirectional Attention-based Model (GCBiA), which applies bi-attention mechanism and convolutions to extract topic words of prompts and key-phrases of responses, and introduces gated unit and residual connections between major layers to better represent the relevance of responses and prompts. Moreover, a new negative sampling method is proposed to augment training data. Experiment results demonstrate that our novel approach can achieve significant improvements in detecting off-topic responses with extremely high on-topic recall, for both seen and unseen prompts.",
T}
