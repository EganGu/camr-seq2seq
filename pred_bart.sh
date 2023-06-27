args=$@
for arg in $args; do
    eval "$arg"
done

echo "devices: ${devices:=4}"
echo "path:    ${path:=exp/syn-bart/model.big}"
echo "encoder: ${encoder:=bart}"
echo "config:  ${config:=configs/syn_bart.ini}"
echo "bart:  ${bart:=/data/yggu/prj/huggingface/bart-large-chinese}"
echo "seed:  ${seed:=1}"

python -u seq2seq.py predict -d $devices -c $config -p $path.$seed --beam-size 5 --use-syn \
    --data data/ccl2023/test.wsyn.bart-large-chinese \
    --pred data/ccl2023/test.pred.syn.$seed
