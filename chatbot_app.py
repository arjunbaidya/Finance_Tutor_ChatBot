# Building an autoGPT style chatbot finance tutor using LangChain

from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.utilities import WikipediaAPIWrapper
from langchain.chains import LLMChain

def main():

    # Loading the api keys
    load_dotenv()

    # Creating the prompt templates
    topic_template = PromptTemplate(
        input_variables=["topic"], 
        template="""
        You are an experienced finance tutor.
        
        Explain the topic: {topic} to a newbie.
        If the topic to explain is not related to finance then just say that "The topic is not related to finance" 
        and do not explain anything.
        """
    )

    lecture_template = PromptTemplate(
        input_variables=["finance_topic", "wikipedia_research"], 
        template="Write me bullet pointed lecture notes based on the following: {finance_topic}, while leveraging this wikipedia reserch:{wikipedia_research}"
    )

    # Creating Memory objects 
    topic_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history")
    lecture_memory = ConversationBufferMemory(input_key="finance_topic", memory_key="chat_history")

    # Creating LLM object
    llm = OpenAI(temperature=0.1) 
    
    # Creating chains
    topic_chain = LLMChain(
        llm=llm,
        prompt=topic_template,
        verbose=True,
        output_key="finance_topic",
        memory=topic_memory
    )
    
    lecture_chain = LLMChain(
        llm=llm,
        prompt=lecture_template,
        verbose=True,
        output_key="lecture_notes",
        memory=lecture_memory
    )

    wiki = WikipediaAPIWrapper()

    # Setting the streamlit app
    st.title("🦜 Your Friendly Finance Tutor")
    prompt = st.text_input("Which Finance topic would you like to learn:") 

    # Show stuff to the screen if there's a prompt
    if prompt: 
        finance_topic = topic_chain.run(prompt)
        wiki_research = wiki.run(prompt)
        lecture_notes = lecture_chain.run(finance_topic=finance_topic, wikipedia_research=wiki_research)

        # If the notes have bullet points then it needs to be processed properly before sending to streamlit.write
        ln = lecture_notes
        notes = ln.split("•")
        tmp = ""
        for note in notes[1:]:
            tmp = tmp + "\n•  " + note + "\n"
        lecture_notes = tmp
        del(tmp)

        st.write(f"Topic: :red[{prompt.capitalize()}]")
        st.write(finance_topic)
        st.write("### :pencil: :blue[_Useful Lecture Notes_]:")
        st.write(lecture_notes)
        st.write("### :blue[_Wikipedia Notes_]:")
        st.write(wiki_research.replace("Page:", "Term:").replace("Summary:", "\nSummary:"))

        with st.expander("Topic History"): 
            st.info(topic_memory.buffer)

        with st.expander("Lecture Notes"): 
            st.info(lecture_memory.buffer)

        with st.expander("Wikipedia Research"): 
            st.info(wiki_research)

if __name__ == "__main__":
    main()