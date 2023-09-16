from time import sleep

import gradio as gr
from Inferencer import Inferencer
from IntentCLS import IntentClassifier
from langchain.embeddings import HuggingFaceEmbeddings
from Memory import FixedWindowLengthMemoryConstructor
from Retriever import Retriever

ADPAPTER_PATH = "ADPAPTER_PATH"
embedder = HuggingFaceEmbeddings("intfloat/multilingual-e5-large")
intent_cls = IntentClassifier.load_local("data/intent_cls", embedder=embedder)
retriever = Retriever.load_local("data/retriever", embedder=embedder)
inferencer = Inferencer(
    adapter_path=ADPAPTER_PATH,
    base_model_path="bigscience/bloom-3b",
    tokenizer_max_length=1024,
    human_symbol="[|Con người|]",
)
inferencer.set_gen_config(
    {
        "temperature": 0.9,
        "top_p": 0.9,
        "top_k": 30,
        "max_new_tokens": 512,
    }
)
memory = FixedWindowLengthMemoryConstructor(
    window_length=1,
    system_message="Cuộc trò chuyện giữa con người và trợ lý AI.\n",
    human_symbol="[|Con người|]",
    ai_symbol="[|AI|]",
)
gen_func = inferencer.generate
intent_cls_func = intent_cls


# app logic
class HCCApp:
    def __init__(self, retriever: Retriever, gen_func, intent_cls_func, memory):
        self.intent_cls_func = intent_cls_func
        self.gen_func = gen_func
        self.retriever = retriever
        self.memory = memory
        self.main_context = None
        self.template_response = {
            "other": "Xin lỗi, tôi không thể trả lời câu hỏi này.",
            "thanks": "Rất vui vì điều đó, tôi có thể hỗ trợ gì thêm cho bạn không?",
            "greeting": "Xin chào, tôi là trợ lý ảo hành chính công. Tôi có thể giúp gì cho bạn?",
            "ability": "Tôi là trợ lý ảo của sếp Hoàng. Tôi có thể giúp bạn trả lời những thắc mắc về lĩnh vực hành chính công. Hãy bắt đầu bằng cách đưa ra những câu hỏi",
            "ending": "Cảm ơn bạn đã sử dụng dịch vụ của tôi. Hẹn gặp lại bạn sau!",
            "unclear": "Xin lỗi, tôi không hiểu câu hỏi của bạn. Bạn hãy đưa ra câu hỏi rõ ràng hơn.",
        }
        self.qa_template = ""

    def _return_answer(self, question, answer):
        self.memory.add_to_memory(question, answer)
        return answer

    def answer(self, question):
        intent = self.intent_cls_func(question)
        if intent in self.template_response:
            return self._return_answer(question, self.template_response[intent])

        if intent == "hcc":
            context = self.retriever.search_main_document(question)
            if context is None:
                return self._return_answer(question, self.template_response["unclear"])
            self.main_context = context

        if intent == "continue":
            if self.main_context is None:
                return self._return_answer(question, self.template_response["unclear"])

        ip = "\n".join(
            self.retriever.search_chunks(self.main_context, question)["chunk_texts"]
        )
        answer = self.gen_func(self.qa_template.format(question=question, context=ip))
        return self._return_answer(question, answer)

    def render(self, share=True):
        def chat(human_input, history=self.memory.memory):
            yield "", history + [(human_input, None)]
            response = ""
            for word in self.answer(human_input).split(" "):
                sleep(self.sleep_time)
                response += word + " "
                yield "", history + [(human_input, response)]

        with gr.Blocks() as demo:
            gr.Markdown("## Chat bot demo")
            with gr.Tabs():
                with gr.TabItem("Chat"):
                    chatbot = gr.Chatbot(height=600)
                    message = gr.Textbox(placeholder="Type your message here...")
                    message.submit(chat, [message, chatbot], [message, chatbot])
        demo.queue().launch(share=share)


hcc_app = HCCApp(retriever, gen_func, intent_cls_func, memory)

# prompt is specific to the model
hcc_app.qa_template = """"""
hcc_app.render()
