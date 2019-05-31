# -*- coding: utf-8 -*-

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import time
from tqdm import tqdm, trange
import evaluation as eva
from processor import UbuntuProcessor
from STMDataLoader import build_corpus_dataloader, build_corpus_tokenizer, build_corpus_embedding
from models import STM

import numpy as np
import pickle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
plt.interactive(False)
plt.figure(figsize=(20,30))
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is not None:
                layers.append(n)
                if n != 'embedding.weight':
                    ave_grads.append(p.grad.abs().mean())
                else:
                    ave_grads.append(p.grad.abs().sum() / (p.grad !=0).sum().float())

    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('./tmp/gradient.png')


def main():

    parser = argparse.ArgumentParser(description='Spatio-Temporal Matching Network')
    parser.add_argument('--task', type=str, default='Response Selection',
                        help='task name')
    parser.add_argument('--model', type=str, default='STM',
                        help='model name')
    parser.add_argument('--encoder_type', type=str, default='GRU',
                        help='encoder:[GRU, LSTM, SRU, Transoformer]')
    parser.add_argument('--vocab_size', type=int, default=30000,
                        help='vocabulary size')
    parser.add_argument('--max_turns_num', type=int, default=9,
                        help='the max turn number in dialogue context')
    parser.add_argument('--max_options_num', type=int, default=100,
                        help='the max turn number in dialogue context')
    parser.add_argument('--max_seq_len', type=int, default=50,
                        help='the max length of the input sequence')

    parser.add_argument('--emb_dim', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--rnn_layers', type=int, default=3,
                        help='the number of rnn layers for feature extraction')
    parser.add_argument('--mem_dim', type=int, default=150,
                        help='hidden memory size')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=5,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='batch size')
    parser.add_argument('--dropoutP', type=float, default=0.2,
                        help='dropout ratio')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='use CUDA')

    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help='The initial checkpoint')
    parser.add_argument('--save_path', type=str, default='./tmp/',
                        help='The initial checkpoint')
    parser.add_argument('--cache_path', type=str, default='./cache/',
                        help='The initial checkpoint')
    parser.add_argument('--pretrain_embedding', type=str, default='/hdd/lujunyu/dataset/glove/glove.42B.300d.txt',
                        help='The pretraining embedding')

    parser.add_argument('--do_train', type=bool, default=True,
                        help='training or not')
    parser.add_argument('--do_eval', type=bool, default=True,
                        help='evaluate or not')
    parser.add_argument('--do_test', type=bool, default=True,
                        help='test or not')

    args = parser.parse_args()
    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device %s", device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    ### Build dataloader
    processor = UbuntuProcessor()
    train_examples = processor.get_train_examples()
    eval_examples = processor.get_dev_examples()
    test_examples = processor.get_test_examples()

    if args.cache_path and os.path.exists(os.path.join(args.cache_path, 'cache_tokenizer.pkl')):
        with open(os.path.join(args.cache_path, 'cache_tokenizer.pkl'), 'rb') as ff:
            tokenizer = pickle.load(ff)
    else:
        tokenizer = build_corpus_tokenizer(train_examples, args.vocab_size)
        with open(os.path.join(args.cache_path, 'cache_tokenizer.pkl'), 'wb') as ff:
            pickle.dump(tokenizer, ff)

    if args.cache_path and os.path.exists(os.path.join(args.cache_path, 'cache_dataset.pt')):
        with open(os.path.join(args.cache_path, 'cache_dataset.pt'), 'rb') as f:
            train_dataset, eval_dataset, test_dataset = pickle.load(f)
    else:
        train_dataset = build_corpus_dataloader(train_examples, args.max_turns_num, args.max_seq_len,tokenizer)
        eval_dataset = build_corpus_dataloader(eval_examples, args.max_turns_num, args.max_seq_len, tokenizer)
        test_dataset = build_corpus_dataloader(test_examples, args.max_turns_num, args.max_seq_len, tokenizer)

        with open(os.path.join(args.cache_path, 'cache_dataset.pt'), 'wb') as f:
            pickle.dump((train_dataset, eval_dataset, test_dataset), f)

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size)

    ### Build pretrained embedding
    if args.cache_path and os.path.exists(os.path.join(args.cache_path, 'cache_embedding.pt')):
        with open(os.path.join(args.cache_path, 'cache_embedding.pt'), 'rb') as f:
            pretrained_embedding = pickle.load(f)
    else:
        if args.pretrain_embedding:
            pretrained_embedding, vocab = build_corpus_embedding(args.vocab_size, args.emb_dim, args.pretrain_embedding, tokenizer)
        else:
            pretrained_embedding = None
            vocab = []

        with open(os.path.join(args.cache_path, 'cache_embedding.pt'), 'wb') as f:
            pickle.dump(pretrained_embedding, f)

        with open(os.path.join(args.cache_path, 'cache_vocab.txt'), 'w') as f:
            for word in vocab:
                f.write(word+'\n')

    ### Build model
    model = eval(args.model)(args, pretrained_embedding)
    if args.init_checkpoint is not None:
        model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'), strict=False)
    model.to(device)

    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(name, param.data)
            print(name, param.data.size())

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adamax(parameters, lr=args.lr)


    ### training
    save_interval = len(train_dataset) // args.batch_size
    print_step = max(1., save_interval // 10)
    if args.do_train:
        model.train()
        for epoch_id in range(int(args.epochs)):
            tr_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                contexts_ids, candidate_ids, label_ids = batch
                loss, _ = model(contexts_ids, candidate_ids, label_ids)
                loss.backward()

                if step % 20 == 0 and step != 0:
                    plot_grad_flow(model.named_parameters())

                optimizer.step()
                optimizer.zero_grad()

                tr_loss += float(loss.data)

                if step % print_step == 0:
                    cur_loss = tr_loss / print_step if step != 0 else tr_loss
                    logger.info("processed: [{:3.2f}".format(step / save_interval) +
                          "] lr: [{:3.5f}".format(optimizer.param_groups[0]['lr']) + "] loss: [{:3.5f}".format(cur_loss) + "]")
                    tr_loss = 0.0


            # Evaluate at the end of every epoch
            if args.do_eval:
                model.eval()
                eval_loss = 0
                best_result = [0, 0, 0, 0, 0, 0, 0]
                eval_score_file = open(os.path.join(args.save_path,'eval_score.txt'), 'w')

                for nb_eval_steps, batch in enumerate(eval_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    contexts_ids, candidate_ids, label_ids = batch
                    with torch.no_grad():
                        tmp_eval_loss, logits = model(contexts_ids, candidate_ids, label_ids)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.cpu().numpy()

                    for i in range(args.batch_size):
                        for j in range(args.max_options_num):
                            if label_ids[i] == j:
                                eval_score_file.write('{:d}_{:d}\t{:2.5f}\t{:d}\n'.format(i, j, logits[i][j], 1))
                            else:
                                eval_score_file.write('{:d}_{:d}\t{:2.5f}\t{:d}\n'.format(i, j, logits[i][j], 0))

                    eval_loss += tmp_eval_loss.mean().item()

                eval_score_file.close()

                eval_loss /= (nb_eval_steps + 1)

                # write evaluation result
                eval_result = eva.evaluate(os.path.join(args.save_path,'eval_score.txt'))
                eval_result_file_path = os.path.join(args.save_path, 'eval_result.txt')
                with open(eval_result_file_path, 'w') as out_file:
                    for p_at in eval_result:
                        out_file.write(str(p_at) + '\n')

                if eval_result[0] > best_result[0]:
                    save_path = os.path.join(args.save_path, 'model_best.pt')
                    best_result = eval_result
                    torch.save([model, optimizer], save_path)
                    logger.info('eval loss: %2.4f' % eval_loss)
                    logger.info("best result: %2.4f" % best_result[0])
                    logger.info("succ saving model in %s" % save_path)

                model.train()

    if args.do_test:
        model, _ = torch.load(os.path.join(args.save_path, 'model_best.pt'))
        model.eval()
        test_loss = 0
        test_score_file = open(os.path.join(args.save_path, 'test_score.txt'), 'w')

        for nb_test_steps, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            contexts_ids, candidate_ids, label_ids = batch
            with torch.no_grad():
                tmp_test_loss, logits = model(contexts_ids, candidate_ids, label_ids)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.cpu().numpy()

            for i in range(args.batch_size):
                for j in range(args.max_options_num):
                    if label_ids[i] == j:
                        test_score_file.write('{:d}_{:d}\t{:2.5f}\t{:d}\n'.format(i, j, logits[i][j], 1))
                    else:
                        test_score_file.write('{:d}_{:d}\t{:2.5f}\t{:d}\n'.format(i, j, logits[i][j], 0))

            test_loss += tmp_test_loss.mean().item()

        test_score_file.close()

        test_loss /= (nb_eval_steps + 1)

        # write evaluation result
        test_result = eva.evaluate(os.path.join(args.save_path, 'test_score.txt'))
        test_result_file_path = os.path.join(args.save_path, 'test_result.txt')
        with open(test_result_file_path, 'w') as out_file:
            for p_at in test_result:
                out_file.write(str(p_at) + '\n')



if __name__ == '__main__':
    main()