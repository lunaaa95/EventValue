import requests
from typing import Any, List, Mapping, Optional
from enum import Enum

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.output_parsers.enum import EnumOutputParser
from langchain.schema import OutputParserException
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class ChatGLM(LLM):
    """ChatGLM和ChatGLM2访问接口
    """
    api_base: str
    temperature: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
            self,
            prompt:str,
            stop: Optional[List[str]] = None,
            history:list=[],
            run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        payload = {
            'prompt': prompt,
            'history': history,
            'temperature':self.temperature
        }
        r = requests.post(
            url=self.api_base,
            json=payload
        )
        response = r.json()['response']
        return response


class Sentiment(Enum):
    POSITIVE = "积极"
    NEGATIVE = "消极"
    NEUTRAL  = "未知"

    def __int__(self):
        values = {
            'POSITIVE': 1,
            'NEGATIVE': -1,
            'NEUTRAL': 0
        }
        return values[self.name]
    
    def __float__(self):
        return float(int(self))


class LLMFinSA:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self._build_prompt()
        self._build_sentiment_parser()
    
    def _build_prompt(self):
        # 回答模板
        response_schemas = [
            ResponseSchema(name="分析", description="一句话分析新闻对市场的影响,"),
            ResponseSchema(name="回答", description="回答<积极>或<消极>,")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        # 问题提示模板
        format_instructions = self.output_parser.get_format_instructions()
        self.prompt = PromptTemplate(
            template="阅读新闻,尽可能回答问题.回答部分只能是<积极>或<消极>\n新闻:{news}\n{format_instructions}",
            input_variables=["news"],
            partial_variables={"format_instructions": format_instructions}
        )
    
    def _build_sentiment_parser(self):
        # 情感极性解析器
        self.enum_parser = EnumOutputParser(enum=Sentiment)
    
    def __call__(self, query) -> Any:
        query = query.strip().replace('\n','')
        _input = self.prompt.format_prompt(news=query)
        output = self.llm(_input.to_string())

        # 让llm分析并回答
        try:
            ans = self.output_parser.parse(output)
        except OutputParserException as e:
            output = output.replace('\n\t"回答"', ',\\n\\t"回答"')
            print(output)
            try:
                ans = self.output_parser.parse(output)
            except OutputParserException as e:
                print(e)
                ans = {'回答': '未知'}

        # 提取情感极性
        try:
            sen = self.enum_parser.parse(ans['回答'])
        except OutputParserException as e:
            print(e)
            sen = Sentiment('未知')

        return sen
