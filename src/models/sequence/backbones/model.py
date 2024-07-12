"""Isotropic deep sequence model backbone, in the style of ResNets / Transformers.

The SequenceModel class implements a generic (batch, length, d_input) -> (batch, length, d_output) transformation.
"""

# import vis
from functools import partial
from typing import Mapping, Optional

import torch
import torch.nn as nn
from einops import rearrange

from src.utils.config import to_list, to_dict
from src.models.sequence.backbones.block import SequenceResidualBlock
from src.models.sequence.base import SequenceModule
from src.models.nn import Normalization, DropoutNd

from spikingjelly.activation_based import functional
# from src.models.sequence.backbones.snn import SNNBlock
from src.models.sequence.backbones.snn import InputMaskingNet

class SequenceModel(SequenceModule):
    """Flexible isotropic deep neural network backbone.

    Options:
      - d_model: Model dimension. Inputs generally have shape (batch, length, d_model).
      - n_layers: Number of repeating blocks.
      - transposed: Transpose inputs so each layer receives (batch, d_model, length).
      - dropout: Dropout parameter applied on every residual and every layer.
      - tie_dropout: Tie dropout mask across sequence like nn.Dropout1d/nn.Dropout2d.
      - prenorm: Pre-norm vs. post-norm placement of the norm layer.
      - bidirectional: Concatenate two copies of each layer like a bi-LSTM.
      - n_repeat: Each layer is repeated n times per stage before applying (optional) pooling.
      - Layer config, must be specified.
      - residual: Residual config, or None for no residual.
      - norm: Normalization config (e.g. layer vs batch), or None for no norm.
      - pool: Config for pooling layer per stage, or None for no pooling.
      - track_norms: Log norms of each layer output.
      - dropinp: Input dropout.
    """
    def __init__(
        self,
        d_model: int,
        n_layers: int = 1,
        transposed: bool = False,
        dropout: int = 0.0,
        tie_dropout: bool = False,
        prenorm: bool = True,
        bidirectional: bool = False,
        n_repeat: int = 1,
        layer: Optional[Mapping] = None,
        residual: Optional[Mapping] = None,
        norm: Optional[Mapping] = None,
        pool: Optional[Mapping] = None,
        track_norms: bool = True,
        dropinp: int = 0.0,
    ):
        super().__init__()
        # Save arguments needed for forward pass
        self.d_model = d_model
        self.transposed = transposed
        self.track_norms = track_norms

        # Input dropout (not really used)
        dropout_fn = partial(DropoutNd, transposed=self.transposed) if tie_dropout else nn.Dropout
        self.drop = dropout_fn(dropinp) if dropinp > 0.0 else nn.Identity()

        # SNN
        # self.snn_block = SNNBlock(step_mode='s', _layer=layer)
        self.input_masking_net = InputMaskingNet(dim=d_model, mode='m')

        layer = to_list(layer, recursive=False)

        # Some special arguments are passed into each layer
        for _layer in layer:
            # If layers don't specify dropout, add it
            if _layer.get('dropout', None) is None:
                _layer['dropout'] = dropout
            # Ensure all layers are shaped the same way
            _layer['transposed'] = transposed

        # Duplicate layers
        layers = layer * n_layers * n_repeat

        # Instantiate layers
        _layers = []
        d = d_model
        for l, layer in enumerate(layers):
            # Pool at the end of every n_repeat blocks
            pool_cfg = pool if (l+1) % n_repeat == 0 else None
            block = SequenceResidualBlock(
                d,
                l+1,
                prenorm=prenorm,
                bidirectional=bidirectional,
                dropout=dropout,
                tie_dropout=tie_dropout,
                transposed=transposed,
                layer=layer,
                residual=residual,
                norm=norm,
                pool=pool_cfg,
            )
            _layers.append(block)
            d = block.d_output

        self.d_output = d
        self.layers = nn.ModuleList(_layers)
        if prenorm:
            if norm is None:
                self.norm = None
            elif isinstance(norm, str):
                self.norm = Normalization(self.d_output, transposed=self.transposed, _name_=norm)
            else:
                self.norm = Normalization(self.d_output, transposed=self.transposed, **norm)
        else:
            self.norm = nn.Identity()

    def forward(self, inputs, *args, state=None, **kwargs):
        """ Inputs assumed to be (batch, sequence, dim) """
        if self.transposed: inputs = rearrange(inputs, 'b ... d -> b d ...')
        inputs = self.drop(inputs)

        # SNN
        # # functional.reset_net(self.snn_block)            # code for 240625-SNN-masked-output-S4-ListOps
        # # spiking_masks = self.snn_block(inputs, len(self.layers))
        functional.reset_net(self.input_masking_net)
        spiking_mask = self.input_masking_net(inputs.permute(1,0,2))    # [B, L, D] -> [L, B, D] (L == T)
        spiking_mask = spiking_mask.permute(1,0,2)                      # [B, L, D], tensor([0., 1.]
        # # for i in range(256):
        # #     _spk_mask = spiking_mask[0,:,i]
        # #     if _spk_mask.sum().item() != 0:
        # #         print(i, _spk_mask.unique())
        # # vis.save_BLD_as_image(inputs, 0, 'ori_inputs_0.png')
        # # vis.save_summed_BLD_graph(inputs, 0, 'ori_inputs_0_sum.png')
        # # vis.save_BLD_zero_ratio_graph(inputs, 0, 'ori_inputs_0_zero_ratio.png')
        # b_idx=0
        # # vis.save_BLD_path_as_image(inputs, b_idx, 'pathfinder_ori.png')
        # # vis.long_term_vis(inputs, 'long_term_inputs_ori.png')
        # # vis.save_BLD_path_as_image(spiking_mask, b_idx, 'pathfinder_spk_mask.png')
        # # vis.save_BLD_path_spk_frq_as_image(spiking_mask, b_idx, 'pathfinder_spk_mask_spk_frequency.png')
        # # vis.long_term_vis(spiking_mask, 'long_term_spiking_mask.png')
        # # vis.save_BLD_path_as_33grid_image(inputs, b_idx, 'D_imgs_inputs_ori.png')
        inputs = spiking_mask * inputs
        
        # # vis.save_BLD_as_image(inputs, 0, 'inputs_0.png')
        # # vis.save_summed_BLD_graph(inputs, 0, 'inputs_0_sum.png')
        # # vis.save_BLD_zero_ratio_graph(inputs, 0, 'inputs_0_zero_ratio.png')
        # # vis.save_BLD_path_as_image(inputs, b_idx, 'pathfinder.png')
        # # # vis.long_term_vis(inputs, 'long_term_inputs.png')
        # # vis.save_BLD_path_as_33grid_image(spiking_mask, b_idx, 'D_imgs_spiking_mask.png')
        # # vis.save_BLD_path_as_33grid_image(inputs, b_idx, 'D_imgs_inputs_masked.png')

        # Track norms
        if self.track_norms: output_norms = [torch.mean(inputs.detach() ** 2)]

        # Apply layers
        outputs = inputs
        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        # ----- original
        # for layer, prev_state in zip(self.layers, prev_states):
        #     outputs, state = layer(outputs, *args, state=prev_state, **kwargs)
        # ----- Dual SSM; origin, spike input
        # layer_idx = 0
        # for layer, spike_layer, prev_state, spiking_mask in zip(self.layers, self.spike_layers, prev_states, spiking_masks):
        #     assert outputs.shape == spiking_mask.shape
        #     outputs, state = layer(outputs, *args, state=prev_state, layer_idx=layer_idx, **kwargs)
        #     spike_outputs, spike_state = spike_layer(spiking_mask, *args, state=prev_state, layer_idx=layer_idx, **kwargs)
        #     assert outputs.shape == spike_outputs.shape
        #     outputs = outputs * spike_outputs
        #     layer_idx += 1
        # ----- Spike masked SSM outputs; CURRENT BEST
        # layer_idx = 0
        # for layer, prev_state in zip(self.layers, prev_states):
        #     outputs, state = layer(outputs, *args, state=prev_state, layer_idx=layer_idx, **kwargs)
        #     spike_mask = self.snn_block(outputs)        # code for 240625-SNN-masked-output-S4-ListOps
        #     layer_idx += 1
        # ----- Spike masked input SSM; Input -> SNN (through L-dim) -> spike masked input -> SSM
        for layer, prev_state in zip(self.layers, prev_states):
            outputs, state = layer(outputs, *args, state=prev_state, **kwargs)
            '''
            0. from /train.py
            layer: SequenceResidualBlock -> block.py
            '''
            next_states.append(state)
            if self.track_norms: output_norms.append(torch.mean(outputs.detach() ** 2))
        if self.norm is not None: outputs = self.norm(outputs)

        if self.transposed: outputs = rearrange(outputs, 'b d ... -> b ... d')

        if self.track_norms:
            metrics = to_dict(output_norms, recursive=False)
            self.metrics = {f'norm/{i}': v for i, v in metrics.items()}

        # vis.save_BLD_path_as_33grid_image(outputs, b_idx, 'D_imgs_outputs_masked.png')
        # exit()
        return outputs, next_states

    @property
    def d_state(self):
        d_states = [layer.d_state for layer in self.layers]
        return sum([d for d in d_states if d is not None])

    @property
    def state_to_tensor(self):
        # Slightly hacky way to implement this in a curried manner (so that the function can be extracted from an instance)
        # Somewhat more sound may be to turn this into a @staticmethod and grab subclasses using hydra.utils.get_class
        def fn(state):
            x = [_layer.state_to_tensor(_state) for (_layer, _state) in zip(self.layers, state)]
            x = [_x for _x in x if _x is not None]
            return torch.cat( x, dim=-1)
        return fn

    def default_state(self, *batch_shape, device=None):
        return [layer.default_state(*batch_shape, device=device) for layer in self.layers]

    def step(self, x, state, **kwargs):
        print(x.shape, state)
        print('src/models/sequence/backbones/model.py/step')
        exit()
        # Apply layers
        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        for layer, prev_state in zip(self.layers, prev_states):
            x, state = layer.step(x, state=prev_state, **kwargs)
            next_states.append(state)

        x = self.norm(x)

        return x, next_states
