# 预测结果中，对预测的重复节点采用虚构对齐，后处理中需要将对齐进行校准，同时去除不必要的概念结点等等

import re
import penman
from amrp.utils import noop_model


def sim_post_process(pred_path, out_path):
    graphs = penman.load(pred_path, model=noop_model)
    n_graphs = []
    for graph in graphs:
        nodes = {}
        for p in graph.metadata['wid'].split('x'):
            p_ = [x.strip() for x in p.split('_')]
            if len(p_) < 2 or not p_[0].isdigit():
                continue
            id = f"x{p_[0]}"
            word = ''.join(p_[1].split(' '))
            if word not in nodes:
                nodes[word] = [id]
            else:
                # 重复词进行记录
                nodes[word].append(id)

        inss_map = {}
        remove_node = []
        for x, rel, y in graph.triples:
            if rel == ':instance':
                inss_map[x] = y
                if re.match(r'x20\d\d', x) is not None and re.match(r'x20\d\d', x).group() == x and \
                    y == 'thing':
                    print(f'remove fictional {(x, rel, y)}')
                    remove_node.append(x)

        relocated_inss_map = {}
        for i, w in inss_map.items():
            pred = False
            w_ = w
            if re.match(r'\S+-\d+', w) is not None and re.match(r'\S+-\d+', w).group() == w:
                w = w.rsplit('-', 1)[0]
                pred = True
            if w in nodes and i not in nodes[w]:
                # 重新生成对齐
                print(f'relocated {w_}')
                relocated_inss_map[i] = nodes[w][0]
            elif len(w) == 1 and pred and w+w in nodes and i not in nodes[w+w]:
                # 某些概念节点的原型可能是叠词，如"扫扫"->"扫-01"
                print(f'force relocated {w_}')
                relocated_inss_map[i] = nodes[w+w][0]
            else:
                relocated_inss_map[i] = i
                # TODO: 对于连续的情况，如x1_x2_x3，如何处理比较好？
        
        n_triples = []
        inss_set = []
        for x, rel, y in graph.triples:
            if rel == ':instance':
                # 对于parsed 虚构的x2000 thing节点，进行丢弃
                if relocated_inss_map[x] not in inss_set and x not in remove_node:
                    inss_set.append(relocated_inss_map[x])
                    n_triples.append((relocated_inss_map[x], ':instance', y))
            else:
                # 对关系进行重新对齐
                try:
                    if x not in remove_node and y not in remove_node:
                        if x == y:
                            print('ignore the self ref')
                            continue
                        
                        if rel == 'ralign-of' or rel == 'coref-of':
                            print('inverse the special rel')
                            n_triples.append((relocated_inss_map[y], rel[:-3], relocated_inss_map[x]))
                        else:
                            n_triples.append((relocated_inss_map[x], rel, relocated_inss_map[y]))
                except Exception as e:
                    print(e)
                    import pdb; pdb.set_trace()
        
        try:
            n_graph = penman.Graph(n_triples, metadata=graph.metadata)
            penman.encode(n_graph)
            n_graphs.append(n_graph)
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
        

    with open(out_path, 'w') as fp:
        fp.write('\n\n'.join([penman.encode(g) for g in n_graphs])+'\n\n')



if __name__ == '__main__':
    sim_post_process('/data/yggu/prj/amr-seq2seq/data/graphene_smatch.txt', 
                     '/data/yggu/prj/amr-seq2seq/data/graphene_smatch.txt.out')