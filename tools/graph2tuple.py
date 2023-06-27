import penman
from penman.models.noop import NoOpModel


class TupleLine:
    def __init__(
        self,
        sid: str = '-',
        nid1: str = '-',
        concept1: str = '-',
        coref1: str = '-',
        rel: str = '-',
        rid: str = '-',
        ralign: str = '-',
        nid2: str = '-',
        concept2: str = '-',
        coref2: str = '-'
    ):
        self.sid = sid
        self.nid1 = nid1
        self.concept1 = concept1
        self.coref1 = coref1
        self.rel = rel
        self.rid = rid
        self.ralign = ralign
        self.nid2 = nid2
        self.concept2 = concept2
        self.coref2 = coref2

    def __str__(self):
        return '\t'.join([self.sid, self.nid1, self.concept1, self.coref1, self.rel,
                          self.rid, self.ralign, self.nid2, self.concept2, self.coref2])


class AMRTuple:
    def __init__(self, sid='-'):
        self.lines = []
        # map {appeared sort: [corresponding lines]}
        self.line_map = {}
        # map {node nid: appeared sort}
        self.node_map = {}
        self.sid = sid

    def add_tuple_line(
        self,
        nid1: str = '-',
        concept1: str = '-',
        coref1: str = '-',
        rel: str = '-',
        rid: str = '-',
        ralign: str = '-',
        nid2: str = '-',
        concept2: str = '-',
        coref2: str = '-'
    ) -> str:
        line = TupleLine(self.sid, nid1, concept1, coref1, rel,
                         rid, ralign, nid2, concept2, coref2)
        self.lines.append(line)
        if nid1 not in self.node_map.keys():
            self.node_map[nid1] = len(self.node_map) + 1
        if self.node_map[nid1] not in self.line_map.keys():
            self.line_map[self.node_map[nid1]] = []
        self.line_map[self.node_map[nid1]].append(line)

    def rel_align(self, ins, aligns):
        for x, rel, y, alg in aligns:
            for line in self.lines:
                if line.nid1 == x and line.rel == rel and line.nid2 == y:
                    assert alg.startswith('x')
                    line.rid = alg
                    if '_' in alg:                        
                        ins_ = ''
                        for w in alg.split('x')[1:]:
                            if w.endswith('_'):
                                w = w[:-1]
                            splits = w.split('_')
                            if len(splits) > 1:
                                for s in splits[1:]:
                                    ins_ += ins[f"x{splits[0]}"][int(s)-1]
                                # seems not that case exsit
                            elif len(splits) == 1:
                                ins_ += ins[f"x{splits[0]}"]
                            else:
                                raise ValueError
                        line.ralign = ins_
                    else:
                        line.ralign = ins[alg]

    def coref_align(self, aligns):
        for x, _, y in aligns:
            for line in self.lines:
                if line.nid1 == x:
                    line.coref1 = y
                if line.nid2 == x:
                    line.coref2 = y

    def deal_with_coref(self, lines):
        return lines

    def get_sorted_tuple(self):
        result = []
        for k in sorted(self.line_map.keys()):
            result += self.line_map[k]
        return [str(x) for x in result]

    def __str__(self) -> str:
        return '\n'.join(self.get_sorted_tuple())


def graph2tuple(input_file, output_file):
    data = []
    with open(input_file, 'r') as fr:
        t = []
        for line in fr.readlines():
            if line != '\n':
                t.append(line)
            else:
                data.append(t)
                t = []

    tuples = []
    for i, g in enumerate(data):
        tuples.append(to_tuple(penman.decode(g, model=NoOpModel())))

    with open(output_file, 'w') as fw:
        fw.write('句子编号\t节点编号1\t概念1\t同指节点1\t关系\t关系编号\t关系对齐词\t节点编号2\t概念2\t同指节点2\n')
        fw.write(
            'sid\tnid1\tconcept1\tcoref1\trel\trid\tralign\tnid2\tconcept2\tcoref2\n\n')
        fw.write('\n\n'.join([str(t) for t in tuples]))
        fw.write('\n\n')


def to_tuple(graph):
    _use_wid = 'wid' in graph.metadata
    
    triples = graph.triples
    sid = graph.metadata['id'].split('.')[1]
    nodes = {}
    if _use_wid:
        for p in graph.metadata['wid'].split('x'):
            p_ = [x.strip() for x in p.split('_')]
            if len(p_) < 2 or not p_[0].isdigit():
                continue
            id = f"x{p_[0]}"
            word = ''.join(p_[1].split(' '))
            nodes[id] = word
    # classify the triples
    tri_classes = {
        'ins': {**nodes},
        'rel': [],
        'ralign': [],
        'coref': []
    }
    
    for tri in triples:
        x, rel, y = tri
        if y is None:
            if x in tri_classes['ins']:
                continue
            elif rel == ':instance':
                y = 'thing'
                # print(sid)
                # print(triples)
                # import pdb; pdb.set_trace()
                # 需要debug，有时需要调整
                # continue
            else:
                continue
            
        if rel == ':instance':
            if y.startswith("\"") and y.endswith("\""):
                y = y[1:-1]
            tri_classes['ins'][x] = y
        elif rel == ':ralign':
            if len(tri_classes['rel']):
                if _use_wid and y not in nodes:
                    continue
                x_, rel_align, y_ = tri_classes['rel'][-1]
                tri_classes['ralign'].append((x_, rel_align, y_, y))
        elif rel == ':coref':
            # _, rel_align, _ = tri_classes['rel'][-1]
            tri_classes['coref'].append((x, '-', y))
            
        # 针对不准确的-of节点
        elif rel == ':ralign-of':
            if len(tri_classes['rel']):
                if _use_wid and x not in nodes:
                    continue
                x_, rel_align, y_ = tri_classes['rel'][-1]
                tri_classes['ralign'].append((x_, rel_align, y_, x))   
        elif rel == ':coref-of':
            tri_classes['coref'].append((y, '-', x))
            
        else:
            tri_classes['rel'].append(tri)

    tuple = AMRTuple(sid)
    # print(penman.encode(graph))
    tuple.add_tuple_line(
        nid1='x0', concept1='root', rel=':top',
        nid2=graph.top,
        concept2=tri_classes['ins'][graph.top])
    for r_tri in tri_classes['rel']:
        x, rel, y = r_tri
        tuple.add_tuple_line(
            nid1=x, concept1=tri_classes['ins'][x],
            nid2=y, concept2=tri_classes['ins'][y],
            rel=rel
        )
    if _use_wid:
        # 只采用原始词进行关系对齐
        tuple.rel_align(nodes, tri_classes['ralign'])
    else:
        tuple.rel_align(tri_classes['ins'], tri_classes['ralign'])
    tuple.coref_align(tri_classes['coref'])

    return tuple


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--i", type=str, default=None, help="path to the input file")
    # parser.add_argument("--o", type=str, default=None, help="path to the output file")
    # parser.add_argument("--m", type=str, default=None, help="transform mode")
    # args = parser.parse_args()

    print('starting transform.')
    # preprocess(args.i, args.o, args.m)
    graph2tuple(
        'AMR-Process-main/data/CCL2023/camr/camr_dev.pre.txt',
        'AMR-Process-main/data/CCL2023/camr/camr_dev.pre.tuple',
        'AMR-Process-main/data/CCL2023/camr/camr_dev.pre.txt.id'
    )
    graph2tuple(
        'AMR-Process-main/data/CCL2023/camr/camr_train.pre.txt',
        'AMR-Process-main/data/CCL2023/camr/camr_train.pre.tuple',
        'AMR-Process-main/data/CCL2023/camr/camr_train.pre.txt.id'
    )
    print('transform finished.')
# python ccl2023/augmentation/preprocess.py --i ccl2023/camr/camr_train.txt --o ccl2023/augmentation/v3/camr_train.pre.txt
# python ccl2023/augmentation/preprocess.py --i ccl2023/camr/camr_train.txt --o ccl2023/augmentation/v3/camr_train.json --m chat
# python ccl2023/augmentation/preprocess.py --i ccl2023/camr/camr_dev.txt --o ccl2023/augmentation/v3/camr_dev.json --m chat
# python ccl2023/augmentation/preprocess.py --i ccl2023/camr/camr_train.txt --o ccl2023/augmentation/v4/camr_train.json --m chat
# python CCL2023/augmentation/preprocess_1.py --i CCL2023/camr/camr_dev.txt --o CCL2023/augmentation/v4/dev.txt
