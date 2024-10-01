import transformers
from transformers import AutoTokenizer
from transformers import (
    AutoModelForCausalLM,
)
from transformers import pipeline, set_seed, LogitsProcessor

import torch

import arithmeticcoding
import io
import numpy as np
import peft
from peft import LoraConfig, get_peft_model

PRECISION = 32

class TextZipper(object):

    def __init__(self, *args, modelname="facebook/opt-350m", adapter_path = None, **kwargs):
        super().__init__(*args, *kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.model = AutoModelForCausalLM.from_pretrained(modelname, torch_dtype=torch.float32, low_cpu_mem_usage=True).cuda()
        if adapter_path:
            self.model.load_adapter(adapter_path)

        self.model = self.model.eval()

        self.precision = PRECISION-3 # less than quarter range to allow for rounding
    
    def encode(self, bitstream, input_text, prompt="",max_length = None):

        if prompt:
            input_text = prompt + " " + input_text
            prompt_end = self.tokenizer([prompt], return_tensors="pt")["attention_mask"].sum() -1
        else:
            prompt_end = 0

        inputs = self.tokenizer([input_text], return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        if max_length is not None:
            input_ids =input_ids[:,:max_length]
        if max_length is not None and i == max_length-1:
             logits[:,i] = -inf
             logits[:,self.tokenizer.eos_token_id]=0
        else: ...





        # compute logits
        with torch.no_grad():
            outputs = self.model.forward(input_ids, return_dict=True)
        logits = outputs['logits']
    
        seq_len = input_ids.shape[1]
        for i in range(prompt_end, seq_len):
            with torch.no_grad():
                outputs = self.model.forward(input_ids[:,:i+1], return_dict=True)
            
            scores = outputs['logits'][:,-1]
    
            # patch
            logits[:,i] = scores
    
        probs = logits.softmax(dim=-1)

        V = logits.shape[2]
    
        # compute entropy
        pseq = probs[0, torch.arange(start=prompt_end, end=seq_len-1), input_ids[0, prompt_end+1:]]

        bitout = arithmeticcoding.BitOutputStream(bitstream)
        ac_enc = arithmeticcoding.ArithmeticEncoder(PRECISION, bitout)
    
        seq_len = input_ids.shape[1]
        seq = input_ids[0,1:]

        H = 0.0
        for i in range(prompt_end, seq_len):
            # make a frequency table from probs
            p = probs[0,i]
            f = torch.ceil(p.float() * (2**self.precision)).long().cpu().numpy().tolist()
            freqs = arithmeticcoding.SimpleFrequencyTable(f)
    
            if i == seq_len-1: # last symbol is EOS
                symbol = self.tokenizer.eos_token_id
            else:
                symbol = int(seq[i])
            H += -torch.log2(p[symbol])
            ac_enc.write(freqs, symbol)
        padding = ac_enc.finish(randomize=False)

        return H.item(), padding

    def probs_to_freq(self, probs):
        p = probs[0]
        freqs = torch.ceil(p.float() * 2**self.precision).long().cpu().numpy().tolist()
        freqs = arithmeticcoding.SimpleFrequencyTable(freqs)
        return freqs
    
        
    def decode(self, bitstream, prompt = "", max_length=30):
        bitin = arithmeticcoding.BitInputStream(bitstream)
        ac_dec = arithmeticcoding.ArithmeticDecoder(PRECISION, bitin)
    
        # tokenize prompt
        inputs = self.tokenizer([ prompt ], return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
       
        # unroll greedy search loop ourselves
        for ip in range(max_length):

            # compute logits
            with torch.no_grad():
                outputs = self.model.forward(input_ids, return_dict=True)
            scores = outputs['logits'][:,-1]
    
            probs = scores.softmax(dim=-1)
    
            # rebuild freqs
            freqs = self.probs_to_freq(probs)
    
            # decode token
            # our arithmetic decoder is modified to read random bits past the end of file
            # so that it gets back to sampling until it finally picks a end-of-sequence token
            symbol = ac_dec.read(freqs)
            next_tokens = torch.tensor([symbol], device=self.model.device)

            # append to sequence
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                
            if symbol == self.tokenizer.eos_token_id:
                break
            
        decoded_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        decoded_text = decoded_text[0]

        if prompt:
            # drop prompt and space
            decoded_text = decoded_text.replace(prompt, "")[1:]

        return decoded_text
