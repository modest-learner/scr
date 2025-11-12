import argparse

from src import process_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default='rmanluo')
    parser.add_argument('--save-dir', type=str, default='data')
    parser.add_argument('--plm', type=str, default='bert-base-uncased')
    parser.add_argument('--datasets', type=str, default='webqsp,cwq')
    parser.add_argument('--splits', type=str, default='train,validation,test')
    parser.add_argument('--max-seq-len', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--num-processes', type=int, default=20)

    process_data(parser.parse_args())