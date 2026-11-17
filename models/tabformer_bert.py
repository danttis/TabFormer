import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertConfig

try:  # transformers >= 4.x moved model internals to transformers.models.*
    from transformers.models.bert.modeling_bert import ACT2FN, BertForMaskedLM
except ImportError:  # transformers < 4.0 (e.g. 3.2.0 pinned in setup.yml)
    from transformers.modeling_bert import ACT2FN, BertForMaskedLM

try:
    from transformers.models.bert.modeling_bert import BertLayerNorm
except ImportError:
    try:
        from transformers.modeling_bert import BertLayerNorm
    except ImportError:  # removed in transformers >= 4: use nn.LayerNorm
        BertLayerNorm = nn.LayerNorm

from models.custom_criterion import CustomAdaptiveLogSoftmax


class TabFormerBertConfig(BertConfig):
    def __init__(
        self,
        flatten=True,
        ncols=12,
        vocab_size=30522,
        field_hidden_size=64,
        hidden_size=768,
        num_attention_heads=12,
        pad_token_id=0,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.ncols = ncols
        self.field_hidden_size = field_hidden_size
        self.hidden_size = hidden_size
        self.flatten = flatten
        self.vocab_size = vocab_size
        self.num_attention_heads=num_attention_heads

class TabFormerBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.field_hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class TabFormerBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = TabFormerBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class TabFormerBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = TabFormerBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class TabFormerBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config, vocab):
        super().__init__(config)

        self.vocab = vocab
        self.cls = TabFormerBertOnlyMLMHead(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,  # renamed from masked_lm_labels in transformers >= 3.5
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            masked_lm_labels=None,
            **kwargs,
    ):
        # backward compatibility with the old keyword
        if labels is None:
            labels = masked_lm_labels

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            # head_mask was removed in newer transformers; not forwarded
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]  # [bsz * seqlen * hidden]

        if not self.config.flatten:
            output_sz = list(sequence_output.size())
            expected_sz = [output_sz[0], output_sz[1]*self.config.ncols, -1]
            sequence_output = sequence_output.view(expected_sz)
            labels = labels.view(expected_sz[0], -1)

        prediction_scores = self.cls(sequence_output) # [bsz * seqlen * vocab_sz]

        outputs = (prediction_scores,) + outputs[2:]

        # prediction_scores : [bsz x seqlen x vsz]
        # labels            : [bsz x seqlen]

        total_masked_lm_loss = 0

        seq_len = prediction_scores.size(1)
        # TODO : remove_target is True for card
        field_names = self.vocab.get_field_keys(remove_target=True, ignore_special=False)
        for field_idx, field_name in enumerate(field_names):
            col_ids = list(range(field_idx, seq_len, len(field_names)))

            global_ids_field = self.vocab.get_field_ids(field_name)

            prediction_scores_field = prediction_scores[:, col_ids, :][:, :, global_ids_field]  # bsz * 10 * K
            labels_field = labels[:, col_ids]
            labels_field_local = self.vocab.get_from_global_ids(global_ids=labels_field,
                                                                what_to_get='local_ids')

            # guard: a field with no masked position in the batch would make
            # the mean cross-entropy NaN (all targets are ignore_index -100)
            if (labels_field_local != -100).sum() == 0:
                continue

            nfeas = len(global_ids_field)
            loss_fct = self.get_criterion(field_name, nfeas, prediction_scores.device)

            masked_lm_loss_field = loss_fct(prediction_scores_field.view(-1, len(global_ids_field)),
                                            labels_field_local.view(-1))

            total_masked_lm_loss += masked_lm_loss_field

        return (total_masked_lm_loss,) + outputs

    def get_criterion(self, fname, vs, device, cutoffs=False, div_value=4.0):

        if fname in self.vocab.adap_sm_cols:
            if not cutoffs:
                cutoffs = [int(vs/15), 3*int(vs/15), 6*int(vs/15)]

            criteria = CustomAdaptiveLogSoftmax(in_features=vs, n_classes=vs, cutoffs=cutoffs, div_value=div_value)

            return criteria.to(device)
        else:
            return CrossEntropyLoss()

class TabFormerBertModel(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        self.cls = TabFormerBertOnlyMLMHead(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            # head_mask was removed in newer transformers; not forwarded
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]  # [bsz * seqlen * hidden]

        return sequence_output