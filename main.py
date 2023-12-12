import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
from argparse import ArgumentParser
import numpy as np
import torch
from datasets import load_dataset, Features, Value
from trainer import Trainer
from transformers import AutoTokenizer

parser = ArgumentParser()
parser.add_argument('-model_path', type=str, default='/storage/ssd1/b04902120/aicup/model.pt')
parser.add_argument('-dataset', type=str, default='opendid_test.tsv')
parser.add_argument('-output_file', type=str, default='answer.txt')
parser.add_argument('-test', action='store_true')
parser.add_argument('-date', action='store_true')

parser.add_argument('-gpuid', type=str, default='0', help='GPUid')
parser.add_argument('-plm', type=str, default='t5-large')
parser.add_argument('-logging_step', type=int, default=1000)
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-early_stop', type=int, default=3)
parser.add_argument('-gradient_accumulation_steps', type=int, default=4)

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Fix random seed for reproducibility
def set_torch_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benckmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]

def set_tokenizer():
    plm = args.plm #"EleutherAI/pythia-70m-deduped"

    # pad = '<|pad|>'
    sep ='\\n'

    special_tokens_dict = {'sep_token': sep}
    tokenizer = AutoTokenizer.from_pretrained(plm)
    # tokenizer.padding_side = 'left'
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"{tokenizer.pad_token}: {tokenizer.pad_token_id}")
    return tokenizer

def main():
    set_torch_seed(1110)
    tokenizer = set_tokenizer()
    trainer = Trainer(device, args, tokenizer)
    if args.test == False:
        dataset = load_dataset("csv", data_files=args.dataset, delimiter='\t',
                                features = Features({'fid': Value('string'), 'idx': Value('int64'),
                                                    'content': Value('string'), 'label': Value('string')}),
                                column_names=['fid', 'idx', 'content', 'label'], keep_default_na=False)
        trainer.train(dataset)
    else:
        dataset = load_dataset("csv", data_files=args.dataset, delimiter='\t',
                                    features = Features({'fid': Value('string'), 'idx': Value('int64'),
                                                        'content': Value('string'), 'label': Value('string')}),
                                    column_names=['fid', 'idx', 'content', 'label'])
        trainer.test(dataset)

    
if __name__ == "__main__":
    main()
    