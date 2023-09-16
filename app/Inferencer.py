from copy import deepcopy
from time import perf_counter

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)


class BaseMemoryConstrucor:
    def __init__(self, system_message, human_symbol, ai_symbol):
        self.system_message = system_message
        self.human_symbol = human_symbol
        self.ai_symbol = ai_symbol
        self.memory = []  # [(human_input, bot_response)]

    def get_full_conversation(self):
        conversation = self.system_message
        for human, ai in self.memory:
            conversation += (
                f"{self.human_symbol} {human}\n" + f"{self.ai_symbol} {ai}\n"
            )
        return conversation

    def clear_memory(self):
        self.memory = []

    def add_to_memory(self, human_input, ai_response):
        self.memory.append((human_input, ai_response))

    def pop_from_memory(self):
        if len(self.memory) > 0:
            self.memory.pop()

    def get_used_memory(self):
        pass

    def construct_input_memory(self, human_input):
        return (
            self.get_used_memory()
            + f"{self.human_symbol} {human_input}\n"
            + f"{self.ai_symbol} "
        )


class FixedWindowLengthMemoryConstructor(BaseMemoryConstrucor):
    def __init__(self, window_length, system_message, human_symbol, ai_symbol):
        super().__init__(system_message, human_symbol, ai_symbol)
        self.window_length = window_length

    def get_used_memory(self):
        conversation = self.system_message
        for human, ai in self.memory[-self.window_length :]:
            conversation += (
                f"{self.human_symbol} {human}\n" + f"{self.ai_symbol} {ai}\n"
            )
        return conversation


# default generation config
default_gen_config = {
    "temperature": 0.9,
    "top_p": 0.9,
    "top_k": 30,
    "max_new_tokens": 512,
}


class Inferencer:
    def __init__(
        self,
        adapter_path,
        base_model_path,
        gen_config=default_gen_config,
        tokenizer_max_length=2048,
        human_symbol="[|Con người|]",
    ):
        self.human_symbol = human_symbol
        self.gen_config = deepcopy(gen_config)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.stopping_criteria = None

        # need to run these functions in order
        print("Loading model and tokenizer...")
        self._load_model_and_tokenizer(
            adapter_path, base_model_path, tokenizer_max_length
        )
        print("Setting stopping criteria...")
        self._set_stopping_criteria([self.human_symbol, self.tokenizer.eos_token])
        print("Building generation pipeline...")
        self._build_pipeline()

    def _load_model_and_tokenizer(
        self, adapter_path, base_model_name_or_path, tokenizer_max_length
    ):
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                device_map="auto",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(model, adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = tokenizer_max_length

    def _set_stopping_criteria(self, stop_seq_list=[]):
        stop_token_ids_list = [
            torch.tensor(
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))
            )
            .long()
            .to("cuda")
            for x in stop_seq_list
        ]

        class StopOnTokens(StoppingCriteria):
            def __call__(
                self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
            ) -> bool:
                for stop_ids in stop_token_ids_list:
                    if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                        return True
                return False

        self.stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    def _build_pipeline(self):
        self.pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            stopping_criteria=self.stopping_criteria,
            return_full_text=False,
            task="text-generation",
            **self.gen_config,
        )

    def set_gen_config(self, gen_config):
        self.gen_config = gen_config
        self._build_pipeline()

    def generate(self, prompt):
        start = perf_counter()
        text_output = self.pipeline(prompt)[0]["generated_text"]
        total_time = perf_counter() - start
        print(f"### Generated in {total_time:.6f} seconds ###\n")
        return text_output
