import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
the_model = 'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es'
tokenizer = AutoTokenizer.from_pretrained(the_model, do_lower_case=False)
model = AutoModelForQuestionAnswering.from_pretrained(the_model)
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)


class modelo():
    contexto = ""
    pregunta = ""
    respuesta = ""
    def set_context(self,texto):
        self.contexto = texto

    def set_question(self,texto):
        self.pregunta =  texto

    def get_answer(self):
        salida = nlp({'question':self.pregunta, 'context':self.contexto}) #llama al modelo
        self.respuesta = salida['answer']
        return self.respuesta

if __name__ == "__main__":
    mod = modelo()
    st.title('Sistema Inteligente de Preguntas y Respuestas usando el modelo de BETO destilado')
    try:
        contexto  = st.text_area("Ingrese el texto a analizar", "")
    except ValueError:
        st.error("Please enter a valid input")
    try:
        pregunta  = st.text_area("Ingrese pregunta en el texto", "")
    except ValueError:
        st.error("Please enter a valid input")
    mod.set_context(contexto)
    mod.set_question(pregunta)
    respuesta = mod.get_answer()
    print (respuesta)
    st.text("La respuesta del Sistema es:")
    st.write(respuesta)

    
    