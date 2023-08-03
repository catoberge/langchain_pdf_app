from dotenv import load_dotenv
import streamlit as st
from langchain import OpenAI

# This allows me to extract text from the PDF-file
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

# Enables search in text
from langchain.vectorstores import FAISS

# Answer questioning model
from langchain.chains.question_answering import load_qa_chain

# Monitor how much money each prompt in OpenAI costs
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    st.set_page_config(page_title="Prompt your PDF")
    st.header("Prompt your PDF 💬")

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # Extract the text from the file
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )

        chunks = text_splitter.split_text(text)

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Show user input
        user_input = st.text_input("Ask a question about your PDF:")
        if user_input:
            docs = knowledge_base.similarity_search(user_input)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")

            # Monitor cost of prompt
            with get_openai_callback() as cb:
                response = chain(
                    {"input_documents": docs, "question": user_input},
                    return_only_outputs=True,
                )
                print(cb)

            st.write(response)


if __name__ == "__main__":
    main()
