from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import Field, BaseModel
from dotenv import load_dotenv
from typing import List
import pandas as pd
import json
import os

load_dotenv()

def busca_dados_de_estudante(estudante):
    dados = pd.read_csv("documentos/estudantes.csv")
    dados_com_esse_estudante = dados[dados["USUARIO"] == estudante]
    if dados_com_esse_estudante.empty:
        return {}
    return dados_com_esse_estudante.iloc[:1].to_dict()

class ExtratorDeEstudante(BaseModel):
    estudante: str = Field("Nome do estudante informado, sempre em letras minusculas. Exemplo: joao, carlos, joana, carla")

class DadosDeEstudante(BaseTool):
    name: str = "DadosDeEstudante"
    description: str = """Esta ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico
                        Passe para essa ferramenta como argumento o nome do estudante"""
    
    def _run(self, input: str) -> str:                     
        llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))      
        parser = JsonOutputParser(pydantic_object=ExtratorDeEstudante)
        template = PromptTemplate(template="""Você deve analisar a entrada a seguir e extrair o nome informado
                                  Entrada:
                                  -------------------------------
                                  {input}
                                  -------------------------------

                       Formato de saída:
                       {formato_saida}""",
                       input_variables=["input"],
                       partial_variables={"formato_saida" : parser.get_format_instructions()})
        
        cadeia = template | llm | parser
        resposta = cadeia.invoke({"input" : input})        
        estudante = resposta['estudante']
        # estudante = input
        estudante = estudante.lower().strip()
        dados = busca_dados_de_estudante(estudante)
        return json.dumps(dados)

class Nota(BaseModel):
    area: str = Field("Nome da area de conhecimento")
    nota: float = Field("Nota na area de conhecimento")

class PerfilAcademicoDeEstudante(BaseModel):
    nome: str = Field("Nome do estudante")
    ano_de_conclusão: int = Field("Ano de conclusão")
    notas: List[Nota] = Field("Lista de notas das diciplinas e areas de conhecimento")
    resumo: str = Field("Resumo das principais caracteristicas desse estudante de forma a torna-lo unico e um otimo potencial estudante para faculdades. Exemplo: Só esse estudante tem bla bla bla")

class PerfilAcademico(BaseTool):
    name: str = "PerfilAcademico"
    description: str = """Cria um perfil acadêmico de um estudante.    
                Esta ferramenta requer como entrada todos os dados do estudante.
                Eu sou incapaz de buscar dados do estudante.
                Você  tem que buscar os dados do estudante antes de me invocar"""
    
    def _run(self, input: str) -> str:
        llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))      
        parser = JsonOutputParser(pydantic_object=PerfilAcademicoDeEstudante)
        template = PromptTemplate(template="""- Formate o estudante para o seu perfil acadêmico.
                                  - Com os dados identifique as opeções de universidades sugeridas e cursos compativeis com o interesse do aluno
                                  - Destaque o perfil do aluno dando enfase principalmente naquilo que faz sentido para as instituições de interesse do aluno
                                  
                                  Persona: Você é uma consultara de carreira e precisa indicar com detalhes, riqueza, mas direto oa ponto para o estudante as opções e consequencias possiveis
                                  Informações atuais:
                                  
                                  {dados_do_estudante}
                                  {formato_de_saida}""", input_variables=["dados_do_estudante"],
                                  partial_variables={"formato_de_saida": parser.get_format_instructions()})
        cadeia = template | llm | parser
        resposta = cadeia.invoke({"dados_do_estudante": input})
        return resposta