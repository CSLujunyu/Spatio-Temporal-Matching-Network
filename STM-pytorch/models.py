# -*- coding: utf-8 -*-

import math

import torch
from torch import nn
from torch.nn import LSTM, GRU
from sru import SRU, SRUCell
from torch.autograd import Variable

# class MaskGRU(nn.Module):
#     def __init__(self, in_dim, out_dim, layers=1, batch_first=True, bidirectional=True, dropoutP = 0.5):
#         super(MaskGRU, self).__init__()
#         self.batch_first = batch_first
#         self.gru_module = GRU(in_dim, out_dim, num_layers=layers, bidirectional=bidirectional, dropout=dropoutP)
#         self.input_dropout = nn.Dropout(0.5)
#
#     def forward(self, input, seq_lens):
#         mask_in = input.new(input.size()).zero_()
#         for i in range(seq_lens.size(0)):
#             mask_in[i,:seq_lens[i]] = 1
#         mask_in = Variable(mask_in, requires_grad=False)
#
#         input_drop = self.input_dropout(input*mask_in)
#
#         H, _ = self.gru_module(input_drop)
#
#         mask = H.new(H.size()).zero_()
#         for i in range(seq_lens.size(0)):
#             mask[i,:seq_lens[i]] = 1
#         mask = Variable(mask, requires_grad=False)
#
#         output = H * mask
#
#         return output

class MaskGRU(nn.Module):

    def __init__(self, in_dim, out_dim, layers=1, batch_first=True, bidirectional=True, dropoutP = 0.2):
        super(MaskGRU, self).__init__()
        self.rnn = nn.GRU(input_size=in_dim,
                        hidden_size=out_dim,
                        num_layers=layers,
                        batch_first=batch_first,
                        bidirectional=bidirectional,
                        dropout=dropoutP)

    def forward(self, x, lengths):
        """
        :param x:            [batch* len,  max_seq_len, emb_size]
        :param lengths:         [batch* len]
        :return result:      [batch, output_dim]
        """

        seq_len = x.size(1)

        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        x_sort = x.index_select(0, idx_sort)

        # NOTE : in case pad token.
        for i, _ in enumerate(lengths_sort):
            if lengths_sort[i] == 0:
                lengths_sort[i] = 1
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x_sort, lengths_sort, batch_first=True)
        o_pack, hn = self.rnn(x_pack)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack, batch_first=True)

        # unsorted o
        o_unsort = o.index_select(0, idx_unsort)  # Note that here first dim is batch

        if o_unsort.size(1) < seq_len:
            dummy_tensor = Variable(torch.zeros(o_unsort.size(0), seq_len - o_unsort.size(1), o_unsort.size(2))).to(o_unsort)
            o_unsort = torch.cat([o_unsort, dummy_tensor], 1)

        return o_unsort


class MaskSRU(nn.Module):
    def __init__(self, in_dim, out_dim, layers=1, batch_first=True, bidirectional=True, dropoutP = 0.5):
        super(MaskSRU, self).__init__()
        self.batch_first = batch_first
        self.sru_module = SRU(in_dim, out_dim, num_layers=layers, bidirectional=bidirectional, dropout=dropoutP)
        self.input_dropout = nn.Dropout(0.5)

    def forward(self, input, seq_lens, return_last=False):
        mask_in = input.new(input.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask_in[i,:seq_lens[i]] = 1
        mask_in = Variable(mask_in, requires_grad=False)

        input_drop = self.input_dropout(input*mask_in)

        H, _ = self.sru_module(input_drop)

        mask = H.new(H.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask[i,:seq_lens[i]] = 1
        mask = Variable(mask, requires_grad=False)

        output = H * mask

        return output

class MaskLSTM(nn.Module):
    def __init__(self, in_dim, out_dim, layers=1, batch_first=True, bidirectional=True, dropoutP = 0.5):
        super(MaskLSTM, self).__init__()
        self.batch_first = batch_first
        self.lstm_module = LSTM(in_dim, out_dim, num_layers=layers, bidirectional=bidirectional, dropout=dropoutP)
        self.input_dropout = nn.Dropout(0.5)

    def forward(self, input, seq_lens):
        mask_in = input.new(input.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask_in[i,:seq_lens[i]] = 1
        mask_in = Variable(mask_in, requires_grad=False)

        input_drop = self.input_dropout(input*mask_in)

        H, _ = self.lstm_module(input_drop)

        mask = H.new(H.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask[i,:seq_lens[i]] = 1
        mask = Variable(mask, requires_grad=False)

        output = H * mask

        return output

class Conv3DNet(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(Conv3DNet, self).__init__()

        self.conv_1 = nn.Sequential(torch.nn.Conv3d(in_channels=in_channel, out_channels=12, kernel_size=(3,3,3),bias=False),
                               torch.nn.ReLU(),
                               torch.nn.Dropout(0.2),
                               torch.nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,3,3))
                               )

        self.conv_2 = nn.Sequential(torch.nn.Conv3d(in_channels=12, out_channels=24, kernel_size=(3,3,3),bias=False),
                               torch.nn.ReLU(),
                               torch.nn.Dropout(0.2),
                               torch.nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3))
                               )
        self.conv_3 = nn.Sequential(torch.nn.Conv3d(in_channels=24, out_channels=out_channel, kernel_size=(3, 3, 3), bias=False),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(0.2),
                                torch.nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3))
                                )

        self.fc1 = nn.Sequential(
            nn.Linear(384, 100),  #
            nn.ReLU(),
            nn.Dropout(0.5))
        # init.xavier_normal(self.fc1.state_dict()['weight'])
        self.fc2 = nn.Sequential(
            nn.Linear(100, 1))
        self.classifier = nn.Sequential(
            self.fc1,
            self.fc2
        )

    def forward(self, feature_maps):

        conv1_output = self.conv_1(feature_maps)

        conv2_output = self.conv_2(conv1_output)

        # conv3_output = self.conv_3(conv2_output)

        flatten = conv2_output.view(conv2_output.size(0), -1)

        output = self.classifier(flatten).squeeze()

        return output


class STM(nn.Module):
    def __init__(self, args, pretrain_embedding):
        super(STM, self).__init__()

        self.emb_dim = args.emb_dim
        self.mem_dim = args.mem_dim
        self.max_turns_num = args.max_turns_num
        self.max_options_num = args.max_options_num
        self.max_seq_len = args.max_seq_len
        self.rnn_layers = args.rnn_layers

        self.embedding = nn.Embedding(args.vocab_size, args.emb_dim)
        if pretrain_embedding is not None:
            self.embedding.weight.data.copy_(pretrain_embedding)
            self.embedding.weight.requires_grad = True

        if args.encoder_type == 'LSTM':
            self.context_encoder = MaskLSTM(in_dim=args.emb_dim, out_dim=args.mem_dim,batch_first=True,
                                           dropoutP=args.dropoutP)
            self.candidate_encoder = MaskLSTM(in_dim=args.emb_dim, out_dim=args.mem_dim, batch_first=True,
                                            dropoutP=args.dropoutP)
        elif args.encoder_type == 'SRU':
            self.context_encoder = nn.ModuleList(
                [MaskSRU(in_dim=args.emb_dim, out_dim=args.mem_dim, batch_first=True,dropoutP=args.dropoutP) for _ in range(args.rnn_layers)]
            )
            self.candidate_encoder = nn.ModuleList(
                [MaskSRU(in_dim=args.emb_dim, out_dim=args.mem_dim, batch_first=True,dropoutP=args.dropoutP) for _ in range(args.rnn_layers)]
            )
        else:
            self.context_encoder = nn.ModuleList(
                [MaskGRU(in_dim=args.emb_dim, out_dim=args.mem_dim, batch_first=True,dropoutP=args.dropoutP) for _ in range(args.rnn_layers)]
            )
            self.candidate_encoder = nn.ModuleList(
                [MaskGRU(in_dim=args.emb_dim, out_dim=args.mem_dim, batch_first=True,dropoutP=args.dropoutP) for _ in range(args.rnn_layers)]
            )

        self.extractor = Conv3DNet(args.rnn_layers+1, 36)
        self.extractor.apply(self.weights_init)


        self.dropout_module = nn.Dropout(args.dropoutP)
        self.criterion = nn.CrossEntropyLoss()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and classname != 'Conv3DNet':
            m.weight.data.normal_(0.0, 0.02)

    def forward(self, contexts_ids, candidates_ids, labels_ids):
        """

        :param contexts_ids: (batch_size, turns_length, seq_length)
        :param candidates_ids: (batch_size, candidates_set_size, seq_length)
        :param labels_ids:  (batch_size, )
        :return:
        """

        context_seq_len = (contexts_ids != 0).sum(dim=-1).long()
        context_turn_len = (context_seq_len != 0).sum(dim=-1).long()
        candidate_seq_len = (candidates_ids != 0).sum(dim=-1).long()
        candidate_turn_len = (candidate_seq_len != 0).sum(dim=-1).long()

        contexts_emb = self.dropout_module(self.embedding(contexts_ids))
        candidates_emb = self.dropout_module(self.embedding(candidates_ids))

        ###
        context_seq_len_inputs = context_seq_len.view(-1)
        candidate_seq_len_inputs = candidate_seq_len.view(-1)

        all_context_hidden = [contexts_emb]
        all_candidate_hidden = [candidates_emb]
        for layer_id in range(self.rnn_layers):
            contexts_inputs = all_context_hidden[-1].view(-1, self.max_seq_len, self.emb_dim)
            candidates_inputs = all_candidate_hidden[-1].view(-1, self.max_seq_len, self.emb_dim)

            contexts_hidden = self.context_encoder[layer_id](contexts_inputs, context_seq_len_inputs)
            candidates_hidden = self.candidate_encoder[layer_id](candidates_inputs, candidate_seq_len_inputs)

            all_context_hidden.append(contexts_hidden.view(-1, self.max_turns_num, self.max_seq_len, 2*self.mem_dim))
            all_candidate_hidden.append(candidates_hidden.view(-1, self.max_options_num, self.max_seq_len, 2 * self.mem_dim))

        all_context_hidden = torch.stack(all_context_hidden, dim=1)
        all_candidate_hidden = torch.stack(all_candidate_hidden, dim=2)

        spatio_temproal_features = torch.einsum('bltik, boljk->boltij', (all_context_hidden, all_candidate_hidden)) / math.sqrt(300)

        spatio_temproal_features = spatio_temproal_features.contiguous().view(-1, self.rnn_layers+1, self.max_turns_num, self.max_seq_len, self.max_seq_len)

        logits = self.extractor(spatio_temproal_features)

        logits = logits.view(-1, self.max_options_num)

        loss = self.criterion(logits, labels_ids)

        return loss, logits

