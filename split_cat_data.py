# 为了压缩训练时推理评估的压力，将开发集规模调整到原来的1/3
import penman
import random


random.seed(42)
full_amr_path = 'data/ccl2023/dev.pre'
full_tuple_path = 'data/ccl2023/tuple/tuples_dev.txt'
full_dep_path = 'data/ccl2023/dep/camr_dev.txt.out.conllu'
sub_amr_path = 'data/ccl2023/sdev.pre'
sub_tuple_path = 'data/ccl2023/tuple/tuples_sdev.txt'
sub_dep_path = 'data/ccl2023/dep/camr_sdev.txt.out.conllu'

cat_amr_path = 'data/ccl2023/train.pre'
cat_tuple_path = 'data/ccl2023/tuple/tuples_train.txt'
cat_dep_path = 'data/ccl2023/dep/camr_train.txt.out.conllu'
cat_o_amr_path = 'data/ccl2023/ctrain.pre'
cat_o_tuple_path = 'data/ccl2023/tuple/tuples_ctrain.txt'
cat_o_dep_path = 'data/ccl2023/dep/camr_ctrain.txt.out.conllu'


split_rate = .33

def read_data(p):
    data = []
    with open(p) as fp:
        sent = []
        for line in fp.readlines():
            line = line.strip()
            if len(line):
                sent.append(line)
            else:
                data.append(sent)
                sent = []
    return data

def write_data(p, data):
    with open(p, 'w') as fp:
        for sent in data:
            fp.write('\n'.join(sent)+'\n\n')

graphs = penman.load(full_amr_path)
tuples = read_data(full_tuple_path)
text = tuples[0]
tuples = tuples[1:]
deps = read_data(full_dep_path)

assert len(graphs) == len(tuples) == len(deps), f"{len(graphs)} {len(tuples)} {len(deps)}"

sub_indexs = random.sample(range(len(graphs)), int(len(graphs) * split_rate) + 1)
remain_indexs = [x for x in range(len(graphs)) if x not in sub_indexs]

# 写入sub dataset
penman.dump([graphs[i] for i in sub_indexs], sub_amr_path)
with open(sub_amr_path, 'a') as fp:
    fp.write('\n')
write_data(sub_tuple_path, [text] + [tuples[i] for i in sub_indexs])
write_data(sub_dep_path, [deps[i] for i in sub_indexs])

# 读取cat dataset
cat_graphs = penman.load(cat_amr_path) + [graphs[i] for i in remain_indexs]
cat_tuples = read_data(cat_tuple_path)
cat_text = cat_tuples[0]
cat_tuples = cat_tuples[1:] + [tuples[i] for i in remain_indexs]
cat_deps = read_data(cat_dep_path) + [deps[i] for i in remain_indexs]

# 写入cat dataset
penman.dump(cat_graphs, cat_o_amr_path)
with open(cat_o_amr_path, 'a') as fp:
    fp.write('\n')
write_data(cat_o_tuple_path, [cat_text] + cat_tuples)
write_data(cat_o_dep_path, cat_deps)
