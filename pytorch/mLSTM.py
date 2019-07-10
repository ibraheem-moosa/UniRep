from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import LSTMCell
from torch.nn import Linear
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
        
