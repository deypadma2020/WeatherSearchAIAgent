import os
import requests
import streamlit as st
from dotenv import load_dotenv
# from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.tools import tool as make_tool
from langchain.agents import create_react_agent, AgentExecutor
# from langchain import hub

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define the weather fetching tool
def get_weather_data(city: str) -> str:
    """
    Fetches the current weather data for a given city.
    """
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    url = f"https://api.weatherstack.com/current?access_key={api_key}&query={city}"
    try:
        response = requests.get(url)
        data = response.json()
        if "error" in data:
            return f"Unable to fetch weather data: {data['error'].get('info', 'Unknown error')}"
        return (
            f"Weather in {city}:\n"
            f"{data['current']['weather_descriptions'][0]}, "
            f"{data['current']['temperature']}°C, "
            f"Humidity: {data['current']['humidity']}%"
        )
    except Exception as e:
        return f"Failed to fetch weather data due to: {str(e)}"

# Wrap as LangChain tool
get_weather_data_tool = make_tool(get_weather_data)

# Set up other tools
# search_tool = DuckDuckGoSearchRun()
search_tool = TavilySearchResults(max_results=3)

# LLM setup
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# Cache agent setup
@st.cache_resource
def initialize_agent():

    from langchain_core.prompts import PromptTemplate

    template = """
    Answer the following questions as best you can.
    
    You have access to the following tools:
    
    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Question: {input}
    
    Thought:{agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template)

    agent = create_react_agent(
        llm=llm,
        tools=[search_tool, get_weather_data_tool],
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=[search_tool, get_weather_data_tool],
        verbose=True
    )

# Streamlit app UI
st.set_page_config(page_title="WeatherSearchAI", page_icon="🌤️")
st.title("🌤️ WeatherSearchAI")

st.markdown("Ask about any place and get current weather details along with search insights.")
st.info("ℹ️ For smaller towns, try entering a nearby well-known city if weather data is unavailable.")

place = st.text_input("Enter a place name:", placeholder="e.g., Bethuadahari, Kolkata, Delhi")

if place:
    query = f"Tell me about {place}, and provide its current weather conditions."
    agent_executor = initialize_agent()
    with st.spinner("Thinking..."):
        try:
            response = agent_executor.invoke({"input": query})
            st.success("Here's what I found:")
            st.markdown(response["output"])
        except Exception as e:
            st.error(f"Something went wrong: {e}")
