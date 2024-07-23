from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

def generate_random_pet_name(animal_type, color):
  llm = OpenAI(temperature=0.6)

  prompt_pet_name = PromptTemplate(
    input_variables=['animal_type', 'color'],
    template="Can you generate five random {animal_type} names with the color {color}?"
  )

  pet_name_chain = LLMChain(llm=llm, prompt=prompt_pet_name)
  response = pet_name_chain({'animal_type':animal_type, 'color':color})

  return response

if __name__ == "__main__":
  print(generate_random_pet_name("Arara", "Azul"))