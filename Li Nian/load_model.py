import argparse
import torch
from model.model import BiDAF
from model.data import SQuAD
from time import gmtime, strftime

parser = argparse.ArgumentParser()
parser.add_argument('--char-dim', default=8, type=int)
parser.add_argument('--char-channel-width', default=5, type=int)
parser.add_argument('--char-channel-size', default=100, type=int)
parser.add_argument('--context-threshold', default=400, type=int)
parser.add_argument('--dev-batch-size', default=100, type=int)
parser.add_argument('--dev-file', default='dev-v1.1.json')
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--epoch', default=12, type=int)
parser.add_argument('--exp-decay-rate', default=0.999, type=float)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--hidden-size', default=100, type=int)
parser.add_argument('--learning-rate', default=0.5, type=float)
parser.add_argument('--print-freq', default=250, type=int)
parser.add_argument('--train-batch-size', default=60, type=int)
parser.add_argument('--train-file', default='train-v1.1.json')
parser.add_argument('--word-dim', default=100, type=int)
args = parser.parse_args()

print('loading SQuAD data...')
data = SQuAD(args)
setattr(args, 'char_vocab_size', len(data.CHAR.vocab))
setattr(args, 'word_vocab_size', len(data.WORD.vocab))
setattr(args, 'dataset_file', f'.data/squad/{args.dev_file}')
setattr(args, 'prediction_file', f'prediction{args.gpu}.out')
setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
print('data loading complete!')

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
model = BiDAF(args, data.WORD.vocab.vectors).to(device)

# load trained the parameters to the model
# model.load_state_dict(torch.load(r'saved_models/BiDAF_08:03:17.pt'))
model.eval()    # show the model result
print(model)