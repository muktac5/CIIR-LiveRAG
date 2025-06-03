def tokenize_add_cls_token_id_and_padding(tokenizer, texts, max_length):
    assert tokenizer.padding_side == "left", tokenizer.padding_side
    tokenized_texts = tokenizer(texts,
                                truncation=True, 
                                padding=False,
                                max_length=max_length-1,
                                return_attention_mask=False,
                                add_special_tokens=True)
    tokenized_texts["input_ids"] = [ids + [tokenizer.cls_token_id] for ids in tokenized_texts["input_ids"]]
    tokenized_texts = tokenizer.pad(tokenized_texts,
                                    padding=True,
                                    pad_to_multiple_of=8,
                                    return_attention_mask=True,
                                    return_tensors="pt")
    return tokenized_texts

class LlamaSparseCollectionCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch):
        ids, texts = [list(xs) for xs in zip(*batch)]
        tokenized_contexts = self.tokenizer(texts,
                                            max_length=self.max_length,
                                            truncation=True, padding="longest", return_tensors="pt")
        return {
            **{k: v for k, v in tokenized_contexts.items()},
            "ids": ids
        }
        