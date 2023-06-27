
echo "AMRdir: ${amrdir:=data/ccl2023}"
echo "DEPdir: ${depdir:=data/ccl2023/dep}"

# 将camr处理为penman格式
echo "========="
echo "start converting to penman..."
for cate in test dev train 
do
	echo "Processing ${cate}..."
    inp_cate=${amrdir}/camr_${cate}.txt
    oup_cate=${amrdir}/${cate}.pre
    echo "Convert ${inp_cate} to ${oup_cate}"
    python -u preprocess.py convert2penman --inp ${inp_cate} --oup ${oup_cate}
done

# 在依存数据中加入amr的wid 即分词信息
echo "========="
echo "start cating wid..."
for cate in ctrain sdev test
do
	echo "Processing ${cate}..."
    amr_cate="${amrdir}/${cate}.pre"
    dep_cate="${depdir}/camr_${cate}.txt.out.conllu"
    echo "Cat $cate to ${depdir}/${cate}.ww.conllu"
    python -u preprocess.py cat_wid --amr ${amr_cate} --dep ${dep_cate} -o ${depdir}/${cate}.ww.conllu
done

# 将依存从word-level转化为token-level
echo "========="
echo "Tokenizer: ${tkz_bart:=/data/yggu/prj/huggingface/bart-large-chinese}"
echo "Tokenizer-type: ${tkz_bart_type:=bart-large-chinese}"
echo "start converting..."
for cate in ctrain sdev test
do
	echo "Processing ${cate}..."
    inp_cate=${depdir}/${cate}.ww.conllu
    oup_cate=${depdir}/${cate}.${tkz_bart_type}.token.conllu
    echo "Convert $cate to ${oup_cate}"
    python -u preprocess.py word2token --inp ${inp_cate} --oup ${oup_cate} --tkz ${tkz_bart}
done

# 在amr中加入token-level 依存和词性
echo "========="
echo "start cating syn..."
for cate in ctrain sdev test
do
	echo "Processing ${cate}..."
    amr_cate="${amrdir}/${cate}.pre"
    dep_cate="${depdir}/${cate}.${tkz_bart_type}.token.conllu"
    echo "Cat $cate to ${amrdir}/${cate}.wsyn.${tkz_bart_type}"
    python -u preprocess.py cat_syn --amr ${amr_cate} --dep ${dep_cate} -o ${amrdir}/${cate}.wsyn.${tkz_bart_type}
done


# 将依存从word-level转化为token-level
echo "========="
echo "Tokenizer: ${tkz_bert:=/data/yggu/prj/huggingface/chinese-roberta-wwm-ext-large}"
echo "Tokenizer-type: ${tkz_bert_type:=chinese-roberta-wwm-ext-large}"
echo "start converting..."
for cate in ctrain sdev test
do
	echo "Processing ${cate}..."
    inp_cate=${depdir}/${cate}.ww.conllu
    oup_cate=${depdir}/${cate}.${tkz_bert_type}.token.conllu
    echo "Convert $cate to ${oup_cate}"
    python -u preprocess.py word2token --inp ${inp_cate} --oup ${oup_cate} --tkz ${tkz_bert}
done

# 在amr中加入token-level 依存和词性
echo "========="
echo "start cating syn..."
for cate in ctrain sdev test
do
	echo "Processing ${cate}..."
    amr_cate="${amrdir}/${cate}.pre"
    dep_cate="${depdir}/${cate}.${tkz_bert_type}.token.conllu"
    echo "Cat $cate to ${amrdir}/${cate}.wsyn.${tkz_bert_type}"
    python -u preprocess.py cat_syn --amr ${amr_cate} --dep ${dep_cate} -o ${amrdir}/${cate}.wsyn.${tkz_bert_type}
done