import argparse
from amrp.utils import cat_wid, cat_syn, convert_word_to_token_conll, camr2penman


def main():
    parser = argparse.ArgumentParser(description='Create Seq2Seq AMRC Parser.')
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # camr2penman
    subparser = subparsers.add_parser('convert2penman', help='convert camr to penman format')
    subparser.add_argument('--inp', default=None, help='path to input amr file')
    subparser.add_argument('--oup', default=None, help='path to output amr file')
    # cat_wid
    subparser = subparsers.add_parser('cat_wid', help='cat wid to the dep file')
    subparser.add_argument('--amr', default=None, help='path to amr file')
    subparser.add_argument('--dep', default=None, help='path to dep file')
    subparser.add_argument('--output', '-o', default=None, help='path to output dep file')
    # convert_word_to_token_conll
    subparser = subparsers.add_parser('word2token', help='convert dep from word-level to token-level')
    subparser.add_argument('--inp', default=None, help='path to input dep file')
    subparser.add_argument('--oup', default=None, help='path to output dep file')
    subparser.add_argument('--tkz', default=None, help='path to tokenizer file')    
    # cat_syn
    subparser = subparsers.add_parser('cat_syn', help='cat syntax and pos to amr file')
    subparser.add_argument('--amr', default=None, help='path to amr file')
    subparser.add_argument('--dep', default=None, help='path to dep file')
    subparser.add_argument('--output', '-o', default=None, help='path to output amr file')
    
    args, unknown = parser.parse_known_args()
    args, unknown = parser.parse_known_args(unknown, args)
    
    if args.mode == 'convert2penman':
        camr2penman(args.inp, args.oup)
    
    if args.mode == 'cat_wid':
        cat_wid(args.amr, args.dep, args.output)
    
    if args.mode == 'word2token':
        convert_word_to_token_conll(args.inp, args.oup, args.tkz)
    
    if args.mode == 'cat_syn':
        cat_syn(args.amr, args.dep, args.output)


if __name__ == "__main__":
    main()
