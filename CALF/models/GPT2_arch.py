from transformers import GPT2Model


class AccustumGPT2Model(GPT2Model):
    def forward(self, input_ids=None, labels=None, **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        outputs = super().forward(input_ids=input_ids, **kwargs)
        return outputs.last_hidden_state, outputs.hidden_states
