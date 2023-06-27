from .graph2tuple import graph2tuple
from .align_smatch import main


class Parser:
    def __init__(
            self,
            f, lf, r=4,
            significant=4,
            v=False, vv=False, ms=False,
            pr=False, justinstance=False,
            justattribute=False, justrelation=False) -> None:
        self.f, self.lf = f, lf
        self.r = r
        self.significant = significant
        self.v, self.vv, self.ms, self.pr = v, vv, ms, pr
        self.justinstance = justinstance
        self.justattribute = justattribute
        self.justrelation = justrelation


def cal_align_smatch(predict_file, gold_file, max_len_file):
    def file_pointer(*ps):
        return [open(p, 'r', encoding='utf-8') for p in ps]
    predict_tuple_file = predict_file + '.tuple'
    graph2tuple(predict_file, predict_tuple_file)
    # -f parsed_amr.txt gold_amr.txt -lf max_len.txt
    predict_tuple_file, gold_file, max_len_file = file_pointer(
        predict_tuple_file, gold_file, max_len_file
    )
    args = Parser(f=[predict_tuple_file, gold_file], lf=max_len_file, pr=True)
    return main(args)

def cal_align_smatch_tuple(predict_file, gold_file, max_len_file):
    def file_pointer(*ps):
        return [open(p, 'r', encoding='utf-8') for p in ps]
    # -f parsed_amr.txt gold_amr.txt -lf max_len.txt
    predict_tuple_file, gold_file, max_len_file = file_pointer(
        predict_file, gold_file, max_len_file
    )
    args = Parser(f=[predict_tuple_file, gold_file], lf=max_len_file, pr=True)
    return main(args)


if __name__ == '__main__':
    print(cal_align_smatch(
        '/public/home/zhli13/yggu/prj/AMRBART-main/fine-tune/outputs/CCL2023-bart-large-chinese-AMRParing-bsz16-lr-1e-5/val_outputs/val_generated_predictions_2072.txt',
        '/public/home/zhli13/yggu/prj/AMR-Process-main/outputs/CCL2023/val-id.txt',
        '/public/home/zhli13/yggu/prj/AMR-Process-main/outputs/CCL2023/val-gold.tuple',
        '/public/home/zhli13/yggu/prj/AMR-Process-main/outputs/CCL2023/max_len.txt'
    ))
    # print(cal_align_smatch(
    #     'AMR-Process-main/data/CCL2023/camr/camr_train.pre.txt',
    #     'AMR-Process-main/data/CCL2023/camr/camr_train.pre.txt.id',
    #     'AMR-Process-main/data/CCL2023/camr_tuples/tuples_train.txt',
    #     'AMR-Process-main/data/CCL2023/camr_tuples/max_len.txt'
    # ))
