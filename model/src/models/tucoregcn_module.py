from typing import Optional, List, Any

import dgl
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import BCEWithLogitsLoss

from .components.rgcn import RelGraphConvLayer
from .components.lstm import TurnLevelLSTM
from .components.attention import MultiHeadAttention
from torchmetrics import Accuracy, F1Score, MetricCollection


class Tucoregcn(pl.LightningModule):
    """
    LightningModule for TUCORE-GCN
    """
    def __init__(
            self,
            roberta: nn.Module,
            optimizer: torch.optim.Optimizer,
            num_labels: int,
            gcn_layers: int = 2,
            activation: str = "relu",
            gcn_dropout: float = 0.6,
            hidden_dropout_prob: float = 0.1,
            hidden_size: int = 768,
            num_attention_heads: int = 12,
            attention_probs_dropout_prob: float = 0.1,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.roberta = roberta
        self.gcn_dim = hidden_size
        self.optimizer = optimizer
        self.num_labels = num_labels
        self.gcn_layers = gcn_layers
        self.gcn_dropout = gcn_dropout
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.ReLU()
        else:
            raise NameError(f"Unexpected activation function {activation}")

        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.classifier = nn.Linear(
            self.hidden_size*3*(self.gcn_layers+1),
            self.num_labels,
        )
        # self.init_weights()

        rel_name_lists = ["speaker", "dialog", "entity"]
        self.GCN_layers = nn.ModuleList([RelGraphConvLayer(self.gcn_dim, self.gcn_dim, rel_name_lists,
                                                           num_bases=len(rel_name_lists), activation=self.activation,
                                                           self_loop=True, dropout=gcn_dropout)
                                         for i in range(self.gcn_layers)])
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.turnAttention = MultiHeadAttention(
            self.num_attention_heads,
            self.hidden_size,
            self.attention_head_size,
            self.attention_head_size,
            self.attention_probs_dropout_prob,
        )
        self.LSTM_layers = nn.ModuleList([
            TurnLevelLSTM(self.hidden_size, 2, 0.2, 0.4) for i in range(self.gcn_layers)
        ])

        metrics = MetricCollection([
            Accuracy(),
            F1Score(num_classes=self.num_labels, average="macro")])
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            speaker_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            graphs=None,
            mention_ids: Optional[torch.Tensor] = None,
            turn_mask: Optional[torch.Tensor] = None,
    ):
        slen = input_ids.size(1)
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # roberta do not use token_type_ids
            token_type_ids=None,
            position_ids=position_ids,
            speaker_ids=speaker_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_outputs = outputs[0]
        pooled_outputs = outputs[1]

        features = None
        sequence_outputs, _ = self.turnAttention(sequence_outputs, sequence_outputs, sequence_outputs, turn_mask)

        num_batch_turn = []

        for i in range(len(graphs)):
            sequence_output = sequence_outputs[i]
            mention_num = torch.max(mention_ids[i])
            num_batch_turn.append(mention_num + 1)
            mention_index = (torch.arange(mention_num) + 1).unsqueeze(1).expand(-1, slen).to(self.device)
            mentions = mention_ids[i].unsqueeze(0).expand(mention_num, -1)
            select_metrix = (mention_index == mentions).float()
            word_total_numbers = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, slen)
            select_metrix = torch.where(word_total_numbers > 0, select_metrix / word_total_numbers, select_metrix)

            x = torch.mm(select_metrix, sequence_output)
            x = torch.cat((pooled_outputs[i].unsqueeze(0), x), dim=0)

            if features is None:
                features = x
            else:
                features = torch.cat((features, x), dim=0)

        graph_big = dgl.batch(graphs)
        output_features = [features]

        for layer_num, GCN_layer in enumerate(self.GCN_layers):
            start = 0
            new_features = []
            for idx in num_batch_turn:
                new_features.append(features[start])
                lstm_out = self.LSTM_layers[layer_num](features[start + 1:start + idx - 2].unsqueeze(0))
                new_features += lstm_out
                new_features.append(features[start + idx - 2])
                new_features.append(features[start + idx - 1])
                start += idx
            features = torch.stack(new_features)
            features = GCN_layer(graph_big, {"node": features})["node"]
            output_features.append(features)

        graphs = dgl.unbatch(graph_big)

        graph_output = list()

        fea_idx = 0
        for i in range(len(graphs)):
            node_num = graphs[i].number_of_nodes('node')
            intergrated_output = None
            for j in range(self.gcn_layers + 1):
                if intergrated_output == None:
                    intergrated_output = output_features[j][fea_idx]
                else:
                    intergrated_output = torch.cat((intergrated_output, output_features[j][fea_idx]), dim=-1)
                intergrated_output = torch.cat((intergrated_output, output_features[j][fea_idx + node_num - 2]), dim=-1)
                intergrated_output = torch.cat((intergrated_output, output_features[j][fea_idx + node_num - 1]), dim=-1)
            fea_idx += node_num
            graph_output.append(intergrated_output)
        graph_output = torch.stack(graph_output)

        pooled_output = self.dropout(graph_output)
        outputs = self.classifier(pooled_output)
        return outputs

    def step(self, batch: Any):
        labels = batch.pop("labels", None)
        outputs = self.forward(**batch)
        logits = outputs.view(-1, self.num_labels)
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.view(-1, self.num_labels)
            loss = loss_fct(logits, labels)
            return loss, logits, labels
        else:
            return logits

    def training_step(self, batch: Any, batch_index: int):
        loss, logits, targets = self.step(batch)
        preds = logits.argmax(dim=-1)
        targets = targets.argmax(dim=-1)
        # calculate train metrics
        self.train_metrics(preds, targets)
        self.log("train/loss", loss)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_index: int):
        loss, logits, targets = self.step(batch)
        preds = logits.argmax(dim=-1)
        targets = targets.argmax(dim=-1)
        # calculate train metrics
        self.val_metrics(preds, targets)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def test_step(self, batch: Any, batch_index: int):
        loss, logits, targets = self.step(batch)
        preds = logits.argmax(dim=-1)
        targets = targets.argmax(dim=-1)

        # calculate train metrics
        self.test_metrics(preds, targets)
        self.log("test/loss", loss, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        self.train_metrics.reset()

    def validation_epoch_end(self, outputs: List[Any]):
        self.val_metrics.reset()

    def test_epoch_end(self, outputs: List[Any]):
        self.test_metrics.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
        }
