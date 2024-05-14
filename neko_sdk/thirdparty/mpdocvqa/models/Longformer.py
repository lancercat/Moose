import re, random
import numpy as np

import torch
import torch.nn as nn
from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering
import models._model_utils as model_utils

""" From https://colab.research.google.com/github/patil-suraj/Notebooks/blob/master/longformer_qa_training.ipynb#scrollTo=ON0le-uD4yiK
Longformer uses sliding-window local attention which scales linearly with sequence length. This is what allows longformer to handle longer sequences. For more details on how the sliding window attention works, please refer to the paper. Along with local attention longformer also allows you to use global attention for certain tokens. For QA task, all question tokens should have global attention.

The attention is configured using the attention_mask paramter of the forward method of LongformerForQuestionAnswering. Mask values are selected in [0, 1, 2]: 0 for no attention (padding tokens), 1 for local attention (a sliding window attention), 2 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).

As stated above all question tokens should be given gloabl attention. The LongformerForQuestionAnswering model handles this automatically for you. To allow it to do that

    The input sequence must have three sep tokens, i.e the sequence should be encoded like this <s> question</s></s> context</s>. If you encode the question and answer as a input pair, then the tokenizer already takes care of that, you shouldn't worry about it.
    input_ids should always be a batch of examples.
"""
class Longformer:

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.tokenizer = LongformerTokenizerFast.from_pretrained(config['model_weights'])
        self.model = LongformerForQuestionAnswering.from_pretrained(config['model_weights'])
        self.page_retrieval = config['page_retrieval'].lower() if 'page_retrieval' in config else None
        self.ignore_index = 9999  # 0

    def parallelize(self):
        self.model = nn.DataParallel(self.model)

    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        context = batch['contexts']
        answers = batch['answers']

        if self.page_retrieval == 'logits':
            outputs = []
            pred_answers = []
            pred_answer_pages = []
            answ_confidence = []
            for batch_idx in range(len(context)):
                document_encoding = self.tokenizer([question[batch_idx]] * len(context[batch_idx]), context[batch_idx], return_tensors="pt", padding=True, truncation=True)

                max_logits = -999999
                answer_page = None
                document_outputs = None
                for page_idx in range(len(document_encoding['input_ids'])):
                    input_ids = document_encoding["input_ids"][page_idx].to(self.model.device)
                    attention_mask = document_encoding["attention_mask"][page_idx].to(self.model.device)

                    # Retrieval with logits is available only during inference and hence, the start and end indices are not used.
                    # start_pos = torch.LongTensor(start_idxs).to(self.model.device) if start_idxs else None
                    # end_pos = torch.LongTensor(end_idxs).to(self.model.device) if end_idxs else None

                    page_outputs = self.model(input_ids.unsqueeze(dim=0), attention_mask=attention_mask.unsqueeze(dim=0))
                    pred_answer, answer_conf = self.get_answer_from_model_output(input_ids.unsqueeze(dim=0), page_outputs)

                    """
                    start_logits_cnf = [page_outputs.start_logits[batch_ix, max_start_logits_idx.item()].item() for batch_ix, max_start_logits_idx in enumerate(page_outputs.start_logits.argmax(-1))][0]
                    end_logits_cnf = [page_outputs.end_logits[batch_ix, max_end_logits_idx.item()].item() for batch_ix, max_end_logits_idx in enumerate(page_outputs.end_logits.argmax(-1))][0]
                    page_logits = np.mean([start_logits_cnf, end_logits_cnf])
                    """

                    if answer_conf[0] > max_logits:
                        answer_page = page_idx
                        document_outputs = page_outputs
                        max_logits = answer_conf[0]

                outputs.append(None)  # outputs.append(document_outputs)  # During inference outputs are not used.
                pred_answers.extend(self.get_answer_from_model_output([document_encoding["input_ids"][answer_page]], document_outputs)[0] if return_pred_answer else None)
                pred_answer_pages.append(answer_page)
                answ_confidence.append(max_logits)

        else:
            encoding = self.tokenizer(question, context, return_tensors="pt", padding=True, truncation=True)
            input_ids = encoding["input_ids"].to(self.model.device)
            attention_mask = encoding["attention_mask"].to(self.model.device)

            context_encoding = self.tokenizer.batch_encode_plus(context, padding=True, truncation=True)
            # start_pos_x, end_pos_x, context_page_token_correspondent_x = self.get_start_end_idx(encoding, context, answers, batch['context_page_corresp'])
            start_pos, end_pos, context_page_token_correspondent = model_utils.get_start_end_idx('Longformer', encoding, context, context_encoding, answers, batch['context_page_corresp'], self.page_retrieval, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id, self.ignore_index, self.model.device)

            outputs = self.model(input_ids, attention_mask=attention_mask, start_positions=start_pos, end_positions=end_pos)
            pred_answers, answ_confidence = self.get_answer_from_model_output(input_ids, outputs) if return_pred_answer else None

            if self.page_retrieval == 'oracle':
                pred_answer_pages = batch['answer_page_idx']

            elif self.page_retrieval == 'concat':
                pred_answer_pages = [context_page_token_correspondent[batch_idx][pred_start_idx] if len(context_page_token_correspondent[batch_idx]) > pred_start_idx else -1 for batch_idx, pred_start_idx in enumerate(outputs.start_logits.argmax(-1).tolist())]

            elif self.page_retrieval == 'none':
                pred_answer_pages = None

        """
        if random.randint(0, 1000) == 0:
            print(batch['question_id'])
            for gt_answer, pred_answer in zip(answers, pred_answers):
                print(gt_answer, pred_answer)

            for start_p, end_p, pred_start_p, pred_end_p in zip(start_pos, end_pos, outputs.start_logits.argmax(-1), outputs.end_logits.argmax(-1)):
                print("GT: {:d}-{:d} \t Pred: {:d}-{:d}".format(start_p.item(), end_p.item(), pred_start_p, pred_end_p))
        """

        # start_pos, end_pos = self.get_start_end_idx(encoding, context, answers)
        return outputs, pred_answers, pred_answer_pages, answ_confidence

    def get_answer_from_model_output(self, input_tokens, outputs):
        start_idxs = torch.argmax(outputs.start_logits, axis=1)
        end_idxs = torch.argmax(outputs.end_logits, axis=1)

        answers = []
        for batch_idx in range(len(input_tokens)):
            context_tokens = self.tokenizer.convert_ids_to_tokens(input_tokens[batch_idx].tolist())

            answer_tokens = context_tokens[start_idxs[batch_idx]: end_idxs[batch_idx] + 1]
            answer = self.tokenizer.decode(
                self.tokenizer.convert_tokens_to_ids(answer_tokens)
            )

            answer = answer.strip()  # remove space prepending space token
            answers.append(answer)

        answ_confidence = model_utils.get_extractive_confidence(outputs)

        return answers, answ_confidence

    # def to(self, device):
    #     self.model.to(device)

