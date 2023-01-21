import torch
from transformers import BertModel, BertTokenizer

class TextProcessor:
    def __init__(
        self,
        model_name: str,
        max_length: int,
        output_hidden_states: bool = True
    ):
        """
        Initialize the TextProcessor class with a pre-trained BERT model

        :param model_name:
            name of the pre-trained BERT model to use.
        :param max_length:
            maximum length of the input text (default=100)
        :param output_hidden_states:
            output hidden states for BERT model
        """
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name, output_hidden_states=output_hidden_states)
        self.max_length = max_length

    def __call__(
        self,
        text: str,
        padding: str = 'max_length',
        truncation: bool = True
    ) -> torch.Tensor:
        """
        Process a text input and return the hidden states of the last layer of BERT

        :param text:
            input text
        :param padding:
            padding option for BERT transformation (default='max length')
        :param truncation:
            truncation option for BERT transformation (default=True)

        :return:
            torch.Tensor: hidden states of the last layer of BERT
        """
        encoded = self.tokenizer.batch_encode_plus([text], max_length=self.max_length, padding=padding, truncation=truncation)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            hidden_states = self.model(**encoded).last_hidden_state.swapaxes(1,2)
        return hidden_states.squeeze(0)[None, :, :]