args=$@
for arg in $args; do
    eval "$arg"
done

echo "devices: ${devices:=4}"
echo "path:    ${path:=exp/bart/model.big}"
echo "encoder: ${encoder:=bart}"
echo "config:  ${config:=configs/bart.ini}"
echo "bart:  ${bart:=/data/yggu/prj/huggingface/bart-large-chinese}"
echo "seed:  ${seed:=1}"

python -u seq2seq.py train -b -d $devices -c $config -p $path.$seed -s $seed --cache --amp --encoder $encoder --bart $bart --eval-tgt --start-eval 22 \
    --train data/ccl2023/ctrain.pre \
    --dev data/ccl2023/sdev.pre



