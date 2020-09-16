from typing import List, Tuple

import torch
import torch.nn as nn


class CharCNN(nn.Module):
    def __init__(
        self,
        char_embedding: nn.Embedding = None,
        filters: List[Tuple[int, int]] = None,
    ):
        super().__init__()

        self.char_embedding = char_embedding
        self.filters = filters

        self.convs = nn.ModuleList()
        self.conv_size = self.char_embedding.weight.size(1)

        for kernel_size, num_filters in self.filters:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.conv_size,
                        num_filters,
                        kernel_size=kernel_size,
                        padding=0,
                    ),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),
                )
            )

        for module in self.convs:
            module.apply(self.init_weights)

    @staticmethod
    def init_weights(module):

        if isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight.data)
            if module.bias is not None:
                nn.init.normal_(module.bias.data)

        elif isinstance(module, nn.BatchNorm1d):
            nn.init.normal_(module.weight.data, mean=1, std=0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, input_matrix: torch.FloatTensor = None):

        batch_size = input_matrix.size(0)

        input_matrix = input_matrix.view(
            input_matrix.size(0) * input_matrix.size(1),
            input_matrix.size(2),
            input_matrix.size(3),
        )

        input_matrix = input_matrix.transpose(1, 2)

        conv_output = list()
        for conv in self.convs:
            conv_output.append(conv(input_matrix))

        conv_output = torch.cat(conv_output, 1).squeeze(2)
        conv_output = conv_output.view(batch_size, -1, conv_output.size(1))

        return conv_output
