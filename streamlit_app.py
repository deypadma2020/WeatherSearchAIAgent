import os
import requests
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate

from langchain.agents import (
    create_react_agent,
    AgentExecutor
)

from langchain_community.tools.tavily_search import (
    TavilySearchResults
)

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ---------------------------------------------
# Weather Tool
# ---------------------------------------------
@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"https://api.weatherstack.com/current"
        f"?access_key={WEATHERSTACK_API_KEY}"
        f"&query={city}"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if "error" in data:
            return "I don't know the current weather for this location."

        current = data["current"]

        return (
            f"Weather in {city}: "
            f"{current['weather_descriptions'][0]}, "
            f"Temperature: {current['temperature']}°C, "
            f"Humidity: {current['humidity']}%, "
            f"Wind Speed: {current['wind_speed']} km/h"
        )

    except Exception:
        return "I don't know the current weather for this location."


# ---------------------------------------------
# Search Tool
# ---------------------------------------------
search_tool = TavilySearchResults(
    max_results=3
)

# ---------------------------------------------
# LLM Setup
# ---------------------------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# ---------------------------------------------
# Agent Initialization
# ---------------------------------------------
@st.cache_resource
def initialize_agent():

    template = """
You are WeatherSearchAI.

You ONLY answer weather and location-related questions.

You have access to the following tools:

{tools}

STRICT RULES:

1. ONLY answer questions related to:
- weather
- temperature
- humidity
- rainfall
- climate
- wind
- locations
- cities
- countries

2. If the user asks anything unrelated, respond EXACTLY with:
"I'm doing well, thank you for asking. I don't know. Please ask only any location-weather related questions."

3. If weather information is unavailable, respond EXACTLY with:
"I don't know the current weather for this location."

4. Keep answers concise and user-friendly.

Use the following format:

Question: the user question
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: input to the action
Observation: result of the action
... (repeat as needed)
Thought: I now know the final answer
Final Answer: final response to the user

Begin!

Question: {input}

Thought:{agent_scratchpad}
"""

    prompt = PromptTemplate.from_template(template)

    agent = create_react_agent(
        llm=llm,
        tools=[search_tool, get_weather_data],
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=[search_tool, get_weather_data],
        verbose=False,
        handle_parsing_errors=True
    )

    return agent_executor

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------
st.set_page_config(
    page_title="WeatherSearchAI",
    page_icon="🌤️",
    layout="centered"
)

st.title("🌤️ WeatherSearchAI")

st.markdown(
    """
Ask about any city or location to get:
- Current weather
- Temperature
- Humidity
- Climate-related insights
"""
)

st.info(
    "ℹ️ Ask only weather or location-related questions."
)

# ---------------------------------------------
# User Input
# ---------------------------------------------
user_input = st.text_input(
    "Enter your question:",
    placeholder="e.g., Weather in Kolkata"
)

# ---------------------------------------------
# Generate Response
# ---------------------------------------------
if user_input:

    agent_executor = initialize_agent()

    with st.spinner("Fetching weather insights..."):

        try:

            response = agent_executor.invoke(
                {
                    "input": user_input
                }
            )

            st.success("Result")

            st.markdown(response["output"])

        except Exception as e:

            st.error(
                f"Something went wrong: {str(e)}"
            )