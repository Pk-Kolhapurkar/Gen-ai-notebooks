import streamlit as st
from transformers import pipeline
from serpapi import GoogleSearch

def get_latest_info(query):
    serpapi_key = st.secrets["serpapi_key"]
    search = GoogleSearch({
        "q": query,
        "api_key": serpapi_key
    })
    results = search.get_dict()
    if 'organic_results' in results and len(results['organic_results']) > 0:
        return results['organic_results'][0]['snippet']
    else:
        return "No relevant information found."

def generate_response(prompt):
    model_name = "facebook/bart-large-cnn"
    generator = pipeline("text2text-generation", model=model_name)
    response = generator(prompt, max_length=100, num_return_sequences=1, truncation=True)[0]['generated_text']
    return response

# Streamlit UI
def main():
    st.title("Generative AI Project Showcase")
    st.write("This app uses SerpAPI to fetch the latest information and a Hugging Face model to generate a response.")

    query = st.text_input("Enter your query:", "Ex. who won the cricket worldcup recently")
    if st.button("Get Latest Information"):
        with st.spinner("Fetching information..."):
            info = get_latest_info(query)
        
        with st.spinner("Generating response..."):
            response = generate_response(info)
        
        st.success("Response generated successfully!")
        st.write("Generated Response:", response)

if __name__ == "__main__":
    main()
