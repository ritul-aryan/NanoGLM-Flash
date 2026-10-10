import re
from typing import List, Dict, Optional, Union

class ByteTokenizer:
    """
    Byte-level Tokenizer with special tokens for Multi-Modal and Reasoning traces.
    - 0..255: Raw UTF-8 bytes
    - 256..383: Special tokens & control symbols
    """
    SPECIAL_TOKENS: Dict[str, int] = {
        "<pad>": 256,
        "<bos>": 257,
        "<eos>": 258,
        "<img_start>": 259,
        "<img_end>": 260,
        "<thought_start>": 261,
        "<thought_end>": 262,
        "<code_start>": 263,
        "<code_end>": 264,
        "<user_start>": 265,
        "<user_end>": 266,
        "<assistant_start>": 267,
        "<assistant_end>": 268,
        "<test_pass>": 269,
        "<test_fail>": 270,
    }
    
    def __init__(self, vocab_size: int = 384):
        self.vocab_size = vocab_size
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        self.token_to_id: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        
        # Regex pattern for splitting text by special tokens
        escaped_tokens = [re.escape(tok) for tok in self.SPECIAL_TOKENS.keys()]
        self.special_regex = re.compile(f"({'|'.join(escaped_tokens)})")
        
        # Reserved token properties
        self.pad_token_id = self.SPECIAL_TOKENS["<pad>"]
        self.bos_token_id = self.SPECIAL_TOKENS["<bos>"]
        self.eos_token_id = self.SPECIAL_TOKENS["<eos>"]
        self.img_start_id = self.SPECIAL_TOKENS["<img_start>"]
        self.img_end_id = self.SPECIAL_TOKENS["<img_end>"]
        self.thought_start_id = self.SPECIAL_TOKENS["<thought_start>"]
        self.thought_end_id = self.SPECIAL_TOKENS["<thought_end>"]
        self.code_start_id = self.SPECIAL_TOKENS["<code_start>"]
        self.code_end_id = self.SPECIAL_TOKENS["<code_end>"]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encodes text to a list of integer token IDs."""
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.bos_token_id)
            
        parts = self.special_regex.split(text)
        for part in parts:
            if not part:
                continue
            if part in self.token_to_id:
                tokens.append(self.token_to_id[part])
            else:
                raw_bytes = part.encode("utf-8")
                tokens.extend(list(raw_bytes))
                
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: Union[List[int], "torch.Tensor"], skip_special_tokens: bool = False) -> str:
        """Decodes token IDs back to a string."""
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
            
        byte_buffer = bytearray()
        out_str = []
        
        for t in tokens:
            if 0 <= t < 256:
                byte_buffer.append(t)
            elif t in self.id_to_token:
                if byte_buffer:
                    out_str.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer = bytearray()
                if not skip_special_tokens:
                    out_str.append(self.id_to_token[t])
            else:
                # Unknown / unassigned token
                if byte_buffer:
                    out_str.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer = bytearray()
                if not skip_special_tokens:
                    out_str.append(f"<tok_{t}>")
                    
        if byte_buffer:
            out_str.append(byte_buffer.decode("utf-8", errors="replace"))
            
        return "".join(out_str)

    def encode_chat(self, user_msg: str, assistant_response: Optional[str] = None) -> List[int]:
        """Encodes a conversation turn using chat control tokens."""
        prompt = f"<user_start>{user_msg}<user_end><assistant_start>"
        tokens = self.encode(prompt, add_bos=True)
        if assistant_response is not None:
            ans_tokens = self.encode(f"{assistant_response}<assistant_end>", add_eos=True)
            tokens.extend(ans_tokens)
        return tokens
