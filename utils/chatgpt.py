import time
from openai import OpenAI
from loguru import logger

PROMPT_LLM = """Question:
{question}

Please answer the given question as simply as possible. Please return only the answers, with each answer on a new line."""
PROMPT_RAG = """Reasoning Paths:
{paths}

Question:
{question}

Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and only return answers. Please return each answer in a new line."""


class ChatGPT:
    def __init__(self, args):
        self.api_key = args.api_key
        self.num_retry = args.num_retry
        self.base_url = args.base_url
        self.model_name = args.model_name

    def generate(self, question: str, paths: list=None):
        if not question.endswith("?"):
            question += "?"

        if paths is None:
            content = PROMPT_LLM.format(question=question)
        else:
            content = PROMPT_RAG.format(
                paths="\n".join(paths),
                question=question
            )

        query = [{"role": "user", "content": content}]

        if self.base_url:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif self.model_name.startswith('deepseek'):
            client = OpenAI(api_key=self.api_key, base_url='https://api.deepseek.com/v1')
        elif self.model_name.startswith('qwen'):
            client = OpenAI(api_key=self.api_key, base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')
        else:
            client = OpenAI(api_key=self.api_key)

        for _ in range(self.num_retry):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=query,
                    extra_body={
                        "enable_thinking": False, 
                        "options": {"num_ctx": 16384}
                    },
                    timeout=600,
                    temperature=0.0,
                    max_tokens=512
                )
                result = response.choices[0].message.content.strip()
                return result
            except Exception as e:
                logger.exception(e)
                time.sleep(5)

        # 兜底返回：如果重试了 num_retry 次依然失败，返回空字符串作为错误答案，防止下游崩溃
        logger.error("All retries failed. Returning empty string.")
        return ""
