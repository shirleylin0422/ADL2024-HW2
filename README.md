
# ADL2024-HW2
https://www.csie.ntu.edu.tw/~miulab/f113-adl/

## Chinese News Summarization (Title Generation)

### Task Description
In this homework, we need to fine-tune mT5 model for Summarization task.

https://docs.google.com/presentation/d/1VRpu0DPlIldkoAtCf64AbttEf6Z6gq3q/edit#slide=id.p1


### Before training or prediction, please ensure that you have run download.sh for everything what you need
```shell
bash ./download.sh
```
This command will download training/validation data and model.

### Before training, 
please clone https://github.com/deankuo/ADL24-HW2.
Then put the 'tw_rouge' folder under current folder and install it.

### How to reproduce model training  

```shell
bash ./train.sh
```

Model will be saved under folder 'model'.
A figure 'rouge_scores.png' which record rouge during validation will be saved under current folder.


### How to use these model for prediction
```shell
bash ./run.sh ./data/test/sample_test.jsonl ./data/test/output_sample_test.jsonl
```
### How to evaluate rouge score for model output file 
clone https://github.com/deankuo/ADL24-HW2 and use eval.py