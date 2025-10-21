from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import Field, BaseModel
from dotenv import load_dotenv
import os

load_dotenv() 

class ExtratorDeEstudante(BaseModel):
    estudante: str = Field("Nome do estudante informado, sempre em letras minusculas. Exemplo: joao, carlos, joana, carla")

class DadosDeEstudante(BaseTool):
    name: str = "DadosDeEstudante"
    description: str = """Esta ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico"""
    
    def _run(self, input: str) -> str:
                     
        llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))                    
        
        parser = JsonOutputParser(pydantic_object=ExtratorDeEstudante)

        template = PromptTemplate(template="""Você deve analisar a {input} e extrair o nome de usuário informado
                       Formato de saída:
                       {formato_saida}""",
                       input_variables=["input"],
                       partial_variables={"formato_saida" : parser.get_format_instructions()})
        
        cadeia = template | llm | parser
        resposta = cadeia.invoke({"input" : input})
        print(resposta)
        return resposta['estudante']
    

pergunta = "Quais os dados da Ana?"

resposta = DadosDeEstudante().run(pergunta)
print(resposta)