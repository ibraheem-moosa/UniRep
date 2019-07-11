import torch
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import LSTMCell
from torch.nn import Linear
from torch.nn import Embedding
from torch.nn.utils import weight_norm

class mLSTMCell(Module):

    def __init__(self, 
            input_size,
            hidden_size):
        super(mLSTMCell, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._lstm_cell = weight_norm(LSTMCell(input_size, hidden_size), name='weight_ih')
        self._lstm_cell = weight_norm(self._lstm_cell, name='weight_hh')
        self._i_multiplier = weight_norm(Linear(input_size, hidden_size, bias=False))
        self._h_multiplier = weight_norm(Linear(hidden_size, hidden_size, bias=False))

    def forward(self, input, state=None):
        batch_size, input_size = input.shape
        assert(self._input_size == input_size)

        if state is None:
            zeros = torch.zeros(batch_size, self._hidden_size, 
                    dtype=input.dtype, device=input.device) 
            state = (zeros, zeros)

        h, c = state
        m = self._i_multiplier(input) * self._h_multiplier(h)
        h, c = self._lstm_cell(input, (m, c))
        return h, c


class mLSTMCellStacked(Module):

    def __init__(self,
            input_size,
            hidden_size,
            num_layers):
        super(mLSTMCellStacked, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._layers = ModuleList([mLSTMCell(input_size, hidden_size)])
        for i in range(num_layers - 1):
            self._layers.append(mLSTMCell(hidden_size, hidden_size))

    def forward(self, input, state=None):
        if state is None:
            state = [None] * self._num_layers

        next_state = []
        for i in range(self._num_layers):
            h, c = self._layers[i].forward(input, state[i])
            input = h
            next_state.append((h, c))
        return next_state
        

import numpy
def load_1900(weight_dir):
    name_fmt = '/rnn_mlstm_mlstm_{}:0.npy'
    b = numpy.load(weight_dir + name_fmt.format('b'))
    wx = numpy.load(weight_dir + name_fmt.format('wx'))
    gx = numpy.load(weight_dir + name_fmt.format('gx'))
    wh = numpy.load(weight_dir + name_fmt.format('wh'))
    gh = numpy.load(weight_dir + name_fmt.format('gh'))
    wmx = numpy.load(weight_dir + name_fmt.format('wmx'))
    gmx = numpy.load(weight_dir + name_fmt.format('gmx'))
    wmh = numpy.load(weight_dir + name_fmt.format('wmh'))
    gmh = numpy.load(weight_dir + name_fmt.format('gmh'))

    mlstm = mLSTMCell(10, 1900)
    mlstm._lstm_cell.bias_ih.data = torch.from_numpy(b)
    mlstm._lstm_cell.bias_hh.data *= 0.
    mlstm._lstm_cell.weight_ih_v.data = torch.from_numpy(wx.transpose())
    mlstm._lstm_cell.weight_ih_g.data = torch.from_numpy(gx.reshape(-1,1))
    mlstm._lstm_cell.weight_hh_v.data = torch.from_numpy(wh.transpose())
    mlstm._lstm_cell.weight_hh_g.data = torch.from_numpy(gh.reshape(-1,1))
    mlstm._i_multiplier.weight_v.data = torch.from_numpy(wmx.transpose())
    mlstm._i_multiplier.weight_g.data = torch.from_numpy(gmx.reshape(-1,1))
    mlstm._h_multiplier.weight_v.data = torch.from_numpy(wmh.transpose())
    mlstm._h_multiplier.weight_g.data = torch.from_numpy(gmh.reshape(-1,1))

    embedding = Embedding(26, 10)
    emb_weight = numpy.load(weight_dir + '/embed_matrix:0.npy')
    embedding.weight.data = torch.from_numpy(emb_weight)

    return embedding, mlstm


embedding, mlstm = load_1900('./1900_weights')
h,c = mlstm.forward(embedding(torch.LongTensor([1, 2, 3, 4])))
print(h.shape)
print(c.shape)
