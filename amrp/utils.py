from typing import List
import penman
import re
import networkx as nx
from supar.models.dep.biaffine.transform import CoNLL
from supar.utils.field import Field, RawField
from supar.utils.common import BOS, PAD, UNK
from tqdm import tqdm
from .tokenization_bert import AMRBertTokenizer, DEFAULT
from penman.models.noop import NoOpModel

noop_model = NoOpModel()

ERROR = 'error'
CONNECTED = 'connected'
DISCONNECTED = 'disconnected'


def tokenize_encoded_graph(encoded):
    # 将encoded的amr图进行线性化，token之间以' '隔开
    linearized = re.sub(r"(\".+?\")", r" \1 ", encoded)
    pieces = []
    for piece in linearized.split():
        if piece.startswith('"') and piece.endswith('"'):
            pieces.append(piece)
        else:
            piece = piece.replace("(", " ( ")
            piece = piece.replace(")", " ) ")
            piece = piece.replace(":", " :")
            piece = piece.replace("/", " / ")
            piece = piece.strip()
            pieces.append(piece)
    linearized = re.sub(r"\s+", " ", " ".join(pieces)).strip()
    return linearized


def get_entry(e: str):
    # 去除encoded的amr图中不需要的部分，例如metadata，用于评估
    lines = [l.strip() for l in e.splitlines()]
    lines = [l for l in lines if (l and not l.startswith('#'))]
    string = ' '.join(lines)
    string = string.replace('\t', ' ')  # replace tabs with a space
    string = re.sub(r'\s+', ' ', string)  # squeeze multiple spaces into a single
    return string


def decode_amr(tokens: List[str], get_raw: bool = False):
    # 将解码到的tokens组织成amr图
    def multi_slash_check(line):
        cnt = 0
        for c in line:
            if c == '/':
                cnt += 1
        return cnt > 1

    def tokens_linearized(tokens):
        toks = []
        for t in tokens:
            if len(toks) and t.startswith('##'):
                toks[-1] += t[2:]
            elif len(toks) and toks[-1].startswith(':') and t.startswith("x"):
                toks.append(f" {t}")
            else:
                toks.append(t)
        return ''.join(toks)

    raw = str(tokens)
    # with open('/data/yggu/prj/amr-seq2seq/data/raw_cases.txt', 'a') as f:
    #     f.write(raw+'\n')
    pieces = tokens_linearized(tokens)
    raw += f"\n{pieces}"

    # match the whole ()
    start, end = 0, len(pieces)-1
    while start < len(pieces):
        if pieces[start] != '(':
            start += 1
        else:
            break
    while end > -1:
        if pieces[end] != ')':
            end -= 1
        else:
            break
    pieces = pieces[start:end+1]

    # deal with the problem that parse multi '/xx/xx/xx/' in concept
    for wp in re.findall(r'[^:\(\)]+', pieces):
        if multi_slash_check(wp):
            ps = wp.split('/')
            wp_n = '/'.join(ps[1:])
            concept = '/'.join(ps[1:]).strip()
            if not concept.startswith('\"'):
                concept = '\"' + concept
            if not concept.endswith('\"'):
                concept = concept + '\"'
            wp_n = ps[0] + f"/{concept}"
            pieces = pieces.replace(wp, wp_n)

    # add the ) if necessary
    n_lp = re.findall(r'\(', pieces)
    n_rp = re.findall(r'\)', pieces)
    if len(n_lp) > len(n_rp):
        pieces += ')'*(len(n_lp)-len(n_rp))

    try:
        # 尝试在缺失边的情况下添加边
        pieces_ = ''
        edge_exist = True
        for c in pieces:
            if c == '(':
                if edge_exist == True:
                    edge_exist = False
                    pieces_ += c
                else:
                    pieces_ += ':arg0('
            elif c == ':':
                edge_exist = True
                pieces_ += c
            else:
                pieces_ += c
        pieces = pieces_
        
        g = penman.decode(pieces)
        g, state = _fix_graph(g)
        if len(g.triples) == 0:
            g = DEFAULT
    except Exception:
        g = DEFAULT
        state = ERROR

    if get_raw:
        return raw, g, state
    return g, state

def _fix_graph(graph: penman.Graph):
    # 修复amr图
    triples = []
    node_dict = {}
    raign_node = []
    error_coref = []
    newvars = 2000
    for triple in graph.triples:
        x, rel, y = triple
        if rel == ':instance':
            # 记录概念节点
            if x not in node_dict.keys() and x not in error_coref:
                y = 'thing' if y is None else y
                node_dict[x] = y
        elif rel == ':ralign':
            # 记录虚词
            # remove x in node_dict
            raign_node.append(y)
        elif rel == ':coref' and y not in node_dict.keys():
            # 记录前文不存在的同指
            error_coref.append(y)
        elif x in error_coref:
            # 同指所关联的节点也标记错误
            error_coref.append(y)
    
    concept_set = []
    for triple in graph.triples:
        x, rel, y = triple
        if x is None or re.match(r'x\d+(_x*\d+)*$', x) is None or x not in node_dict.keys():
            pass
        elif rel == ':instance':
            if x not in concept_set:
                # 不能重复实例化相同节点
                concept_set.append(x)
                triples.append(penman.Triple(x, rel, node_dict[x]))
            elif y is not None:
                var = f'x{newvars}'
                newvars += 1
                triples.append(penman.Triple(var, ':instance', y))       
        elif x == y or y is None or \
                re.match(r'x\d+(_x*\d+)*$', y) is None or \
                y not in node_dict.keys():
            # y 不符合规范
            if rel != ':coref':
                # 非同指情况 new y
                var = f'x{newvars}'
                newvars += 1
                triples.append(penman.Triple(x, rel, var))
                triples.append(penman.Triple(var, ':instance', 'thing'))
        elif rel == ':coref':
            if y not in raign_node and y in node_dict.keys():
                triples.append(triple)
        else:
            triples.append(triple)
    graph = penman.Graph(triples)

    try:
        state = CONNECTED
        penman.encode(graph)
    except Exception:
        # if graph is not connected, use 'and' and 'op' to make it connected.
        state = DISCONNECTED
        graph = connect_graph(graph)

    return graph, state


def connect_graph(graph):
    # import pdb; pdb.set_trace()
    nxgraph = nx.MultiGraph()
    variables = graph.variables()
    for v1, _, v2 in graph.triples:
        if v1 in variables and v2 in variables:
            nxgraph.add_edge(v1, v2)
        elif v1 in variables:
            nxgraph.add_edge(v1, v1)

    triples = graph.triples.copy()
    new_triples = []
    addition = f"x{len(variables) + 100}"
    triples.append(penman.Triple(addition, ":instance", "and"))
        
    for i, conn_set in enumerate(nx.connected_components(nxgraph), start=1):
        edge = f":op{i}"
        # for 'x14_x15', the key of it is 14.
        conn_set = sorted(conn_set, key=lambda x: int(x[1:]) if '_' not in x else int(x.split('_')[0][1:]))
        conn_set = [c for c in conn_set if c in variables]
        node = conn_set[0]
        new_triples.append(penman.Triple(addition, edge, node))
    triples = triples + new_triples
    metadata = graph.metadata
    graph = penman.Graph(triples)
    graph.metadata.update(metadata)
    
    
    return graph


def get_dep_trans():
    TAG, CHAR, ELMO, BERT = None, None, None, None
    WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, lower=True)
    TEXT = RawField('texts')
    ARC = Field('arcs', bos=BOS, use_vocab=False, fn=CoNLL.get_arcs)
    REL = Field('rels', bos=BOS)
    transform = CoNLL(FORM=(WORD, TEXT, CHAR, ELMO, BERT), CPOS=TAG, HEAD=ARC, DEPREL=REL)
    return transform    


def cat_wid(amr_path, dep_path, dep_o_path):
    # 将wid加入到依存中
    amrs = penman.load(amr_path)
    id2wid = {amr.metadata['id']: tuple(amr.metadata['wid'].split()) for amr in amrs}

    deps = list(get_dep_trans().load(dep_path))

    assert len(deps) == len(amrs)

    for dep in tqdm(deps):
          if dep.annotations[-1][1:] in id2wid.keys():
                dep.values[1] = id2wid[dep.annotations[-1][1:]]
          else:
                raise KeyError(f"{dep.annotations[-1][1:]} not match")
    with open(dep_o_path, 'w') as fw:
          for dep in deps:
                fw.write(str(dep) + '\n')

def _convert_word_to_token_conll(conll_chunk, tokenize):
    li = conll_chunk.split("\n")
    amr_id = ''
    word_li = []
    label_li = []
    head_li = []
    pos_li = []
    for line in li:
        if line.startswith('#'):
            amr_id = line
            continue
        meta_info = line.split("\t")
        word_li.append(meta_info[1])
        pos_li.append(meta_info[3])
        # assert meta_info[3] == meta_info[4]
        head_li.append(int(meta_info[6]))
        label_li.append(meta_info[7])
    token_li = []
    token_pos_li = []
    for word, pos in zip(word_li, pos_li):
        assert '_' in word, f"Sir, this way -> {amr_id}"
        # 报错需要手动调整
        tokens = tokenize(word.replace('_', ' '))
        token_li.append(tokens)
        token_pos_li.extend([pos] * len(tokens))

    word_len_li = [len(word) for word in token_li]
    res_li = []
    now = 0
    for i, word in enumerate(token_li):
        for j, char in enumerate(word[:-1]):
            res_li.append((char, now + j + 2, "app"))
        dis = 0
        if head_li[i] == 0:
            res_li.append((word[-1], 0, label_li[i]))
        elif i + 1 < head_li[i]:
            for k in range(i+1, head_li[i]):  # 计算两个词的结尾字符之间相距多少个字
                dis += word_len_li[k]
            res_li.append((word[-1], now + len(word) + dis, label_li[i]))
        else:
            for k in range(head_li[i],i+1):  # 计算两个词的结尾字符之间相距多少个字
                dis += word_len_li[k]
            res_li.append((word[-1], now + len(word) - dis, label_li[i]))
        now += len(word)
    res_str = f"{amr_id}\n"
    for i, res in enumerate(res_li):
        # 检测弧是否合法
        # assert i+1 != res[1]
        # assert 0 <= res[1] <= len(res_li)
        res_str += f"{i+1}\t{res[0]}\t_\t{token_pos_li[i]}\t{token_pos_li[i]}\t_\t{res[1]}\t{res[2]}\t_\t_\n"
    res_str += "\n"
    return res_str

def convert_word_to_token_conll(input_file, output_file, tokenizer_path):
    # 将依存从word-level转换成token(subword)-level
    t = AMRBertTokenizer.from_pretrained(tokenizer_path)
    input_chunks = [chunk for chunk in open(input_file, "r", encoding="utf-8").read().split("\n\n") if len(chunk)>0]
    with open(output_file, "w", encoding="utf-8") as o:
        for chunk in tqdm(input_chunks):
            res = _convert_word_to_token_conll(chunk, t.tokenize)
            o.write(res)

def cat_syn(amr_path, dep_path, amr_o_path):
    # 把依存、词性信息加入到amr中
    amrs = penman.load(amr_path, model=noop_model)
    deps = list(get_dep_trans().load(dep_path))
    
    for amr, dep in tqdm(zip(amrs, deps)):
      assert amr.metadata['id'] == dep.annotations[-1][1:]
      amr.metadata['pos'] = ' '.join(dep.values[3])
      amr.metadata['arc'] = ' '.join(dep.values[6])
      amr.metadata['rel'] = ' '.join(dep.values[7])

    with open(amr_o_path, 'w') as fw:
        for amr in amrs:
            fw.write(penman.encode(amr)+'\n\n')
    # penman.dump(amrs, amr_o_path)
    

def camr2penman(input_file, output_file, indent=6):
    # 将camr转化为penman规范
    def complete_graph(lines, anchors, indent):
        assert '# ::wid' in lines[2]
        n_lines = [*lines[:3]]
        wids = lines[2].replace('# ::wid ', '').strip().split()
        for wid in wids:
            anc, ins = wid.split('_')
            if anc not in anchors:
                anchors[anc] = ins
            # else:
            #     assert anchors[anc] == ins, f"{ins} not match {anc}: {anchors[anc]} in anchors!"

        for line in lines[3:]:
            # fix the ignore type like :arg1 x171)
            # to the full type like :arg1 (x171 / xx))
            co_line = None
            # if re.search(r'\([\S\s]+', line) is None:
            #     anc = re.search(r'x\d+[_x\d]*\s*', line)
            #     assert anc is not None
            #     anc_f = anc.group()
            #     anc = anc_f.strip()
            #     if anc not in anchors.keys():
            #         print(line)
            #         input()
            #     else:
            #         ins = f"({anc} / {anchors[anc]})"
            #     line = line.replace(anc_f, ins)
            
            # 同指情况
            ancs = re.findall(r'x\d+[_x\d]*', line)
            if len(ancs) > 1 and re.search(r'\sx\d+[_x\d]*', line) is not None:
                assert len(ancs) == 2
                # map the anc0 to nomal font
                ins0 = ''
                for w in ancs[0].split('x')[1:]:
                    if w.endswith('_'):
                        w = w[:-1]
                    splits = w.split('_')
                    if len(splits) > 1:
                        for s in splits[1:]:
                            ins0 += anchors[f"x{splits[0]}"][int(s)-1]
                        # seems not that case exsit
                    elif len(splits) == 1:
                        ins0 += anchors[f"x{splits[0]}"]
                    else:
                        raise ValueError
                # from tail to top, sub once.
                line = re.sub(ancs[1][::-1], ins0[::-1], line[::-1], 1)[::-1]
                # line = line.replace(ancs[1], ins0)
                co_line = indent * ' ' + \
                    f":coref {ancs[1]}"
                # 补充括号
                if re.search(r'\)+$', line) is not None:
                    co_line += re.search(r'\)+$', line).group()
                    line = re.sub(r'\)+$', "", line)
                # 补充前面的indent
                if re.search(r'^\s*', line) is not None:
                    co_line = re.search(r'^\s*', line).group() + co_line
                # print(line)
                # print(co_line)
                # print(n_lines[0])
                # input()
            if not line.endswith('\n'):
                line += '\n'
            n_lines.append(line)
            if co_line is not None:
                n_lines.append(co_line + '\n')

        return n_lines

    data = []
    with open(input_file, 'r') as fr:
        sent = []
        anchors = {}
        for line in fr.readlines():
            if line != '\n':
                # if line.startswith('# ::id'):
                #     print(line)
                # integrate the opx in :name()
                if 'op1' in line and re.search(r':op\d+[-of]*\(', line) is None:
                    role_wf = re.findall(r':name\(x\d+[_x\d+]*/\S+\)', line)
                    if len(role_wf):
                        line = line.replace(role_wf[0], ':name()')
                    splits = line.split('/')
                    pref = splits[0]
                    op = ''
                    for i in range(2, len(splits)):
                        op += splits[i].split(':op')[0].strip().replace(' ', '')
                    # print(line)
                    # print(pref, op)
                    line = pref+'/ '+op+'\n'
                    # print(line)
                    # input()
                    if len(role_wf):
                        line = line.replace(':name()', role_wf[0])

                # map :role(fictional word) => :role \n :ralign (fictional word)
                sp_line = None
                fw = re.search(r':\S+\(x\d+[_x\d+]*/\S+\)', line)
                if fw is not None:
                    fw = re.search(r'\(x\d+[_x\d+]*/\S+\)',
                                   fw.group()).group()[1:-1]
                    line = line.replace(fw, '')
                    # print(fw)
                    fw_align = f":ralign ({fw.replace('/', ' / ')})"
                    sp_line = indent * ' ' + fw_align
                    if re.search(r'\)+$', line) is not None:
                        sp_line += re.search(r'\)+$', line).group()
                        line = re.sub(r'\)+$', "", line)
                    if re.search(r'^\s*', line) is not None:
                        sp_line = re.search(r'^\s*', line).group() + sp_line

                # remove '()'
                rel = re.search(r':\S+\(\)', line)
                if rel is not None:
                    n_rel = rel.group().replace('()', '')
                    line = line.replace(rel.group(), n_rel)

                # fix the ignore type like :arg1 x171)
                # to the full type like :arg1 (x171 / xx))
                # if '# ::' not in line:
                #     ins = re.search(r'\([\S\s]+', line)
                #     if ins is not None:
                #         ins = ins.group().strip().replace('(', '').replace(')', '')
                #         anc, ins = [s.strip() for s in ins.split(' / ')]
                #         if anc not in anchors.keys() and len(re.findall(r'x\d+[_x\d]*', line)) == 1:
                #             anchors[anc] = ins
                sent.append(line)
                if sp_line is not None:
                    sent.append(sp_line+'\n')
            else:
                sent = complete_graph(sent, anchors, indent=indent)
                data.append(sent)
                sent = []
                anchors = {}

    graphs = []
    for sent in tqdm(data):
        amr = ''.join(sent)
        g = penman.decode(amr, model=noop_model)
        # if g.metadata['id'].split('.')[1] == '1850':
        #     import pdb;pdb.set_trace()
        triples = []
        inss = {}
        ralign_inss = []
        # 去掉重复的triple
        for x, rel, y in g.triples:
            if rel == ':instance':
                if x not in inss:
                    inss[x] = y
                    triples.append((x, rel, y))
                elif y != inss[x] and y is not None:
                    # if x in ralign_inss: continue 存在与关系对齐相同的节点，以其他节点优先
                    if x not in ralign_inss:
                        import pdb; pdb.set_trace()  
            elif (x, rel, y) not in triples:
                if rel == ':ralign':
                    ralign_inss.append(y)
                triples.append((x, rel, y))  
        g.triples = triples
        graphs.append(g)
        
        try:
            penman.decode(penman.encode(g))
        except Exception as e:
            print(e)
            print(penman.encode(g))
            print(g.triples)
            import pdb; pdb.set_trace()
    
    with open(output_file, 'w') as fw:
        for g in graphs:
            fw.write(penman.encode(g)+'\n\n')
    # penman.dump(graphs, output_file)