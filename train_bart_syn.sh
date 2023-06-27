# nohup bash train.sh devices=0 update=5 path=exp/bart/model.lr3e-5.batch40960.epochs60.warmup2000        encoder=bart        config=configs/bart.ini     > exp/bart/model.lr3e-5.batch40960.epochs60.warmup2000.train.log.verbose 2>&1 &
args=$@
for arg in $args; do
    eval "$arg"
done

echo "devices: ${devices:=5}"
echo "path:    ${path:=exp/syn-bart/model.big}"
echo "encoder: ${encoder:=bart}"
echo "config:  ${config:=configs/syn_bart.ini}"
echo "bart:  ${bart:=/data/yggu/prj/huggingface/bart-large-chinese}"
echo "seed:  ${seed:=1}"


python -u seq2seq.py train -b -d $devices -c $config -p $path.$seed -s $seed --cache --amp --encoder $encoder --bart $bart \
    --eval-tgt --start-eval 30 --use-syn \
    --train data/ccl2023/ctrain.wsyn.bart-large-chinese \
    --dev data/ccl2023/sdev.wsyn.bart-large-chinese

