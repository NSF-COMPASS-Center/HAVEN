import torch.nn as nn
import torch
from models.nlp.transformer.transformer import TransformerEncoder
from utils import nn_utils


# only encoder
class MaskedLanguageModel(nn.Module):
    def __init__(self, encoder_model, pad_token_val, mask_token_val, no_mask_token_vals, n_tokens,
                 mask_prob=0.15, random_mask_prob=0.1, no_change_mask_prob=0.1):
        super(MaskedLanguageModel, self).__init__()
        self.encoder_model = encoder_model
        self.pad_token_val = pad_token_val
        self.mask_token_val = mask_token_val
        self.no_mask_token_vals = no_mask_token_vals
        self.n_tokens = n_tokens
        self.mask_prob = mask_prob
        self.random_mask_prob = random_mask_prob
        self.no_change_mask_prob = no_change_mask_prob

    def mask_sequence(self, sequence):
        # create a clone of the original sequence to generate labels for masked positions
        label = sequence.clone()

        # mask <mask_prob> (15%) of the sequence
        init_mask = torch.rand(sequence.shape, device=nn_utils.get_device()) < self.mask_prob

        # exclude the <no_mask_tokens> if selected for masking in mask_pos
        for no_mask_token_val in self.no_mask_token_vals:
            no_mask = sequence != no_mask_token_val # positions WITHOUT the <no_mask_token_val>
            init_mask = init_mask & no_mask # only positions WITHOUT the the <no_mask_token_val> will be retained for final masking

        # mask for positions to be left unchanged (i.e., replace with the original tokens)
        unchanged_token_mask = torch.rand(sequence.shape, device=nn_utils.get_device()) < self.no_change_mask_prob
        unchanged_token_mask = init_mask & unchanged_token_mask

        ## Masking: Replace with Random Tokens
        # mask for positions to be replaced with random tokens
        random_token_mask = torch.rand(sequence.shape, device=nn_utils.get_device()) < self.random_mask_prob
        random_token_mask = init_mask & random_token_mask

        # positions for random masking
        random_mask_pos = torch.nonzero(random_token_mask) # returns indices of all non-zero values in the tensor
        n_random_mask_tokens = len(random_mask_pos) # number of random mask positions

        # random tokens to be used for replacement in each of the selected positions
        random_mask_tokens = torch.randint(low=0, high=self.n_tokens, size=n_random_mask_tokens, device=nn_utils.get_device())
        # replace the random token positions with the generated random tokens
        seq[random_mask_pos] = random_mask_tokens

        ## Final mask
        # generate the final mask
        mask = init_mask & ~random_token_mask & ~unchanged_token_mask

        # fill with the mask position vals
        sequence.masked_fill_(mask, self.mask_token_val)

        ## Replace all the non masked positions (init_mask) in the label with pad_token_val which will be ignored in the Cross Entropy loss calculation  as below
        # this code is in the mlm pipeline: CrossEntropyLoss(ignore_index=pad_token_val)
        label.masked_fill_(~init_mask, self.pad_token_val)

        return sequence, label, init_mask


    def forward(self, X):
        X, label, mask = self.mask_sequence(X)
        masked_seq_logits = self.transformer_encoder(X, mask)
        return masked_seq_logits, label


def get_mlm_model(encoder_model, mlm_model):
    mlm_model = MaskedLanguageModel(encoder_model=encoder_model,
                                    pad_token_val=mlm_model["pad_token_val"],
                                    mask_token_val=mlm_model["mask_token_val"],
                                    no_mask_token_vals=mlm_model["no_mask_token_vals"],
                                    n_tokens=mlm_model["n_tokens"],
                                    mask_prob=mlm_model["mask_prob"],
                                    random_mask_prob=mlm_model["random_mask_prob"],
                                    no_change_mask_prob=mlm_model["no_change_mask_prob"])

    print(mlm_model)
    print("Number of parameters = ", sum(p.numel() for p in mlm_model.parameters() if p.requires_grad))
    return mlm_model.to(nn_utils.get_device())
