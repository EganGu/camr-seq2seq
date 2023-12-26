# SUDA at CAMRP2023

This repository contains the system we submitted at [CAMRP2023](https://github.com/GoThereGit/Chinese-AMR). 


## Intro

We transform the parsing task into a sequence generation task by linearizing CAMR graph and propose a CAMR parsing model based on the transformer architecture. To run our codes, you should do the following things:

1. Preparations (python env installation and data preprocess)
2. Train our model
3. Prediction with our model
4. (Optional) Use Graphene to ensemble the result. 


## Preparations

Install our env. 

    conda env create -f environment.yaml

Converte raw data to PENMAN format.

    inp_cate=input.txt
    oup_cate=output.pre
    echo "Convert ${inp_cate} to ${oup_cate}"
    python -u preprocess.py convert2penman --inp ${inp_cate} --oup ${oup_cate}

(Optional) Add dependency and POS to the data for syntax-enhanced model.

    # aligning Dependencies and AMR
    amr_cate=input.pre
    dep_cate=input.dep
    python -u preprocess.py cat_wid --amr ${amr_cate} --dep ${dep_cate} -o ${dep_cate}.ww

    # converte the depedencies from word-level to token-level
    echo "Tokenizer: ${tkz_bart:=bart-path}"
    echo "Tokenizer-type: ${tkz_bart_type:=bart}"
    echo "start converting..."
    inp_cate=input.dep.ww
    oup_cate=input.dep.token
    echo "Convert $cate to ${oup_cate}"
    python -u preprocess.py word2token --inp ${inp_cate} --oup ${oup_cate} --tkz ${tkz_bart}

    # add the token-level dependencies/POS to AMR
    amr_cate=input.pre
    dep_cate=input.dep.token
    python -u preprocess.py cat_syn --amr ${amr_cate} --dep ${dep_cate} -o ${amr_cate}.wsyn.${tkz_bert_type}

The cmds above can be seen in **preprocess.sh**.

Note that we repartition the train and dev sets according to **split_cat_data.py** to speed up the efficiency.

## Train

Run the following cmd to train the model. 

    python -u seq2seq.py train -b -d 0 -c config-path -p exp/bart/model --cache --amp --encoder bart --bart bart-path --eval-tgt \
        --train data/ccl2023/train.pre \
        --dev data/ccl2023/dev.pre

You can see the config templates under **configs/**. The usage of training params can be viewed at **seq2seq.py**. 

## Predict

Run the following cmd to predict. 

    python -u seq2seq.py predict -d 0 -c config-path -p exp/bart/model --beam-size 5 \
        --data data/ccl2023/test.pre \
        --pred data/ccl2023/test.pred


## Evaluate

Due to our predicted results are PENMAN-format AMR graphs, we need to convert them to tuple-format and use the tool from [CAMRP2023](https://github.com/GoThereGit/Chinese-AMR) (which we put them at **tools/**) to evaluate. 

    from tools.cal_asmatch import cal_align_smatch
    
    cal_align_smatch(pred-path, gold-path, max-len-path)

In some cases, correcting the alignment results predicted by the model will improve performance a little, and you can run the **postprocess.py** to do this.

## Graphene

Graph ensemble method can significantly improve performance. We improved the integration speed by multi-process approach based on [Graphene](https://github.com/IBM/graph_ensemble_learning). 

    python tools/graphene_fast.py \
        --gold (optional)data/ccl2023/test.pre \
        --data ensemble-dir

## Citation
If you find this repo helpful, please cite the following paper:

    @inproceedings{gu-etal-2023-ccl23,
        title = "System Report for {CCL}23-Eval Task 2: Autoregressive and Non-autoregressive {C}hinese {AMR} Semantic Parsing based on Graph Ensembling",
        author = "Gu, Yanggan  and
        Zhou, Shilin  and
        Li, Zhenghua",
        booktitle = "Proceedings of the 22nd Chinese National Conference on Computational Linguistics (Volume 3: Evaluations)",
        year = "2023",
        publisher = "Chinese Information Processing Society of China",
    }

