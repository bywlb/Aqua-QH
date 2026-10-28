

DEFAULT_PROMPT = """请根据用户的信息检索需求，比较以下两段水产领域候选文本的相关程度。
用户查询：{query}
候选文本 A：{doc_a}
候选文本 B：{doc_b}
哪段文本能够更好地支持查询的信息需求？只输出 A 或 B。"""


@dataclass
class FlanT5Judge:
    model_name: str = "google/flan-t5-large"
    prompt: str = DEFAULT_PROMPT
    max_query_tokens: int = 64
    max_passage_tokens: int = 160

    def __post_init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        kwargs = {"torch_dtype": torch.float16} if torch.cuda.is_available() else {}
        if torch.cuda.is_available():
            # Auto placement helps larger FLAN variants; may offload to CPU.
            kwargs["device_map"] = "auto"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **kwargs)
        self.model.eval()
        self.label_ids = {}
        for label in ("A", "B"):
            tokens = self.tokenizer.encode(label, add_special_tokens=False)
            if len(tokens) != 1:
                raise ValueError(f"Token {label} is not a single token for this tokenizer")
            self.label_ids[label] = tokens[0]

    def _truncate(self, text: str, size: int) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=size)
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _probs(self, query: str, doc_a: str, doc_b: str) -> tuple[float, float]:
        text = self.prompt.format(query=query, doc_a=doc_a, doc_b=doc_b)
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        # For model parallelism the first embedding shard is the input device.
        device = self.model.get_input_embeddings().weight.device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        start = self.model.config.decoder_start_token_id
        if start is None:
            start = self.tokenizer.pad_token_id
        decoder = self.torch.tensor([[start]], dtype=self.torch.long, device=device)
        with self.torch.inference_mode():
            logits = self.model(**tokens, decoder_input_ids=decoder).logits[0, -1]
            selected = logits[[self.label_ids["A"], self.label_ids["B"]]].float()
            p = self.torch.softmax(selected, dim=-1).cpu().tolist()
        return float(p[0]), float(p[1])

    def compare(self, query: str, doc_a: str, doc_b: str) -> float:
        q = self._truncate(query, self.max_query_tokens)
        a = self._truncate(doc_a, self.max_passage_tokens)
        b = self._truncate(doc_b, self.max_passage_tokens)
        p_af, p_bf = self._probs(q, a, b)
        p_ar, p_br = self._probs(q, b, a)
        score_a = (p_af + p_br) / 2.0
        score_b = (p_bf + p_ar) / 2.0
        return float(score_a / (score_a + score_b))


class MockJudge:
    """No-LLM test double. Scores are not research results."""
    def compare(self, query: str, doc_a: str, doc_b: str) -> float:
        chars = {char for char in query if not char.isspace()}
        sa = len(chars.intersection(doc_a))
        sb = len(chars.intersection(doc_b))
        return float(1 / (1 + np.exp(-0.5 * (sa - sb))))
