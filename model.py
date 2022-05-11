from typing import Optional

import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertModel, RobertaModel, AutoModel, BertPreTrainedModel, AutoModelForQuestionAnswering
import torch.nn as nn
from transformers.modeling_outputs import QuestionAnsweringModelOutput
import torch.nn.functional as F

def get_model(model_args, config, ):
    if model_args.use_default:
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )
    else:
        model = Birdirectional_model(config = config)

    return model

class BiLSTMHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=int(input_size), hidden_size=1024, num_layers=1, bidirectional=True, batch_first=True)
        self.final_layer = nn.Linear(1024 * 2, 2)
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        h0 = Variable(torch.zeros(
            2, x.size(0), self.input_size
        )).to('cuda:0')
        c0 = Variable(torch.zeros(
            2, x.size(0), self.input_size
        )).to('cuda:0')

        ula, (hn, cn) = self.lstm(x, (h0, c0))

        self.dropout(ula)

        logits = self.final_layer(ula)

        return logits

class CnnHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv_1 = nn.Conv1d(
            in_channels=input_size, out_channels=2, kernel_size=1, padding=0
        )
        self.conv_3 = nn.Conv1d(
            in_channels=input_size, out_channels=2, kernel_size=3, padding=1
        )
        self.conv_5 = nn.Conv1d(
            in_channels=input_size, out_channels=2, kernel_size=5, padding=2
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        conv1_out = self.relu(self.conv_1(x).transpose(1, 2).contiguous().squeeze(-1))
        conv3_out = self.relu(self.conv_3(x).transpose(1, 2).contiguous().squeeze(-1))
        conv5_out = self.relu(self.conv_5(x).transpose(1, 2).contiguous().squeeze(-1))
        x = (conv1_out + conv3_out + conv5_out)/3

        return x

class Birdirectional_model(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        config.add_pooling_layer=False

        self.roberta = AutoModel.from_pretrained('klue/roberta-large', config=config)
        self.cnnHead = CnnHead(1024)
        self.lstm = BiLSTMHead(1024)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.lstm(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )