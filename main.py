from langchain.agents import AgentExecutor
from agente import AgenteOpenAIFunctions
from estudante import *
from dotenv import load_dotenv

load_dotenv()   

pergunta = "Quais os dados da Bianca e da Ana?"
pergunta = "Crie um perfil academico para a Ana"
   
agente = AgenteOpenAIFunctions()
executor = AgentExecutor( agent=agente.agente, tools=agente.tools, verbose=True)

resposta = executor.invoke({"input": pergunta})
print(resposta)

# resposta = DadosDeEstudante().run(pergunta)
# print(resposta)