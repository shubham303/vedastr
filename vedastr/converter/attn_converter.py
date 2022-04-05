# modify from clovaai

import torch
from abfn import abfn
from .base_convert import BaseConverter
from .registry import CONVERTERS


@CONVERTERS.register_module
class AttnConverter(BaseConverter):

    def __init__(self, character, batch_max_length,language_list,go_last=False):
        
        """
        language_list is used for multilingual model. for single language model list contains one element.
        """
        list_character = list(character)
        self.batch_max_length = batch_max_length + 1
        if go_last:
            list_token = ['[s]', '[GO]']
            character = list_character + list_token
        else:
            list_token = ['[GO]', '[s]']
            character = list_token + list_character
        super(AttnConverter, self).__init__(character=character)
        self.ignore_index = self.dict['[GO]']
        self.language_list = language_list

    def train_encode(self, text):
        length = [len(s) + 1 for s in text]
        batch_text = torch.LongTensor(len(text), self.batch_max_length + 1).fill_(self.ignore_index)  # noqa 501
        language_id = torch.LongTensor(len(text))
        for idx, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[idx][1:1 + len(text)] = torch.LongTensor(text)
        
            lang = abfn.detect_language(t)
            if lang not in self.language_list:
                print("word: {} not in language list. check if language codes defined in config files are "
                      "correct".format(t) )
            lang_id = self.language_list.index(lang) if lang in self.language_list else len(self.language_list)
            language_id[idx] = lang_id
                
        batch_text_input = batch_text[:, :-1]
        batch_text_target = batch_text[:, 1:]

        return batch_text_input, torch.IntTensor(length), batch_text_target , language_id

    def decode(self, text_index):
        texts = []
        batch_size = text_index.shape[0]
        for index in range(batch_size):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            text = text[:text.find('[s]')]
            texts.append(text)

        return texts
