import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from .product_tool import product_recommendation_tool
from .common_info_tool import common_information_tool


load_dotenv()

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash",
)

root_agent = Agent(
    name="smartshopper_agent",
    model=Gemini(
        model=MODEL_NAME,
        retry_options=types.HttpRetryOptions(
            attempts=3
        ),
    ),
    description=(
        "AI shopping assistant for personalized product "
        "recommendations and general e-commerce information."
    ),
    instruction="""
You are Personalized SmartShopper Assistant.

You have TWO capabilities.

1. PRODUCT RECOMMENDATION
Use get_product_recommendations when the user asks for:
- recommended products
- suitable products
- personalized products
- recommendations for a user_id such as U001

Product recommendations MUST come from the Product Recommendation tool.
Never invent products yourself.

2. COMMON INFORMATION
Use retrieve_common_information when the user asks about:
- shipping
- delivery
- tracking
- purchase process
- checkout
- payment
- cancellation
- return
- refund

Common Information MUST come from the Common Information tool.
Never invent store policies yourself.

3. MIXED QUESTIONS
If a user asks both a product recommendation question and a
common information question, use BOTH relevant tools before answering.

4. OUT OF SCOPE
If the question is unrelated to SmartShopper, shopping, products,
or e-commerce processes, explain that it is outside your scope.

5. RESPONSE STYLE
Respond in Bahasa Indonesia unless the user uses another language.
Keep answers clear, concise, contextual, and structured.
""",
    tools=[
        product_recommendation_tool,
        common_information_tool,
    ],
)

app = App(
    root_agent=root_agent,
    name="smartshopper_agent",
)