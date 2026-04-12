from dotenv import load_dotenv
import os
from pathlib import Path
import base64
from typing import Dict, Any
from tavily import TavilyClient
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()

def load_image_as_base64(image_path: str) -> str:
    """Load a local PNG image and convert it to base64 encoded string."""
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if image_path.suffix.lower() != ".png":
        raise ValueError(f"Expected a .png file, got: {image_path.suffix}")
    
    with open(image_path, "rb") as image_file:
        base64_encoded = base64.b64encode(image_file.read()).decode("utf-8")
    
    return base64_encoded


def create_human_message_with_image(image_path: str, prompt: str = "Describe this image.") -> HumanMessage:
    """Create a LangChain HumanMessage with a base64-encoded image."""
    base64_image = load_image_as_base64(image_path=image_path)
    message = HumanMessage(
        content=[
            { "type": "text", "text": prompt },
            { "type": "image", "base64": base64_image, "mime_type": "image/png" }
        ]
    )
    return message

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return tavily_client.search(query=query, search_depth="basic", max_results=4, include_answer='basic')


# Create the personal chef agent
chef_agent = create_agent(
    name = "personal_chef_agent",
    model = "gpt-5-nano",
    system_prompt = SystemMessage(content="""
You are a personal chef. The user will give you an image. Your job is to describe the ingredients in the image and suggest recipes that can be made with those ingredients.
Using the web search tool, search the web for recipes that can be made with the ingredients they have.
Return recipe suggestions and eventually the recipe instructions to the user, if requested.
"""),
    tools = [web_search]
)


def run_demo() -> None:
    image_path = Path("notebooks/module-1/resources/fridge.png")
    if not image_path.exists():
        image_path = Path("resources/fridge.png")

    response = chef_agent.invoke(
        input={
            "messages": [
                create_human_message_with_image(
                    image_path=str(image_path),
                    prompt="Describe the ingredients in this image. Then suggest some recipes I can make with them.",
                )
            ]
        }
    )

    print(response['messages'][-1].content)


if __name__ == "__main__":
    run_demo()
