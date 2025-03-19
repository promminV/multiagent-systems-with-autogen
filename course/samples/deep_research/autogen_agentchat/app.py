# create a virtual env e.g conda create -n autogen python=3.12
# pip install -U autogen-agentchat autogen-ext[openai]  
# This snippet uses the Google Search API. You need to set your google search engine id and api key
# os.environ["GOOGLE_CSE_ID"] = "your_google_cse_id"
# os.environ["GOOGLE_API_KEY"] = "your_google_api_key"


import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from typing import List, Dict, Optional

# Import the google_search_tool from your implementation
from google_search import google_search, google_search_tool
from fetch_webpage import fetch_webpage, fetch_webpage_tool
 
import argparse
import os
import logging

from autogen_core import TRACE_LOGGER_NAME

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(TRACE_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

def str_to_bool(value):
    return value.lower() in ("true", "1", "yes")

parser = argparse.ArgumentParser(description="Configuration")
parser.add_argument("--forced_google_search_params", type=bool, default=str_to_bool(os.environ.get("forced_google_search_params", "false")))
parser.add_argument("--forced_fetch_webpage_params", type=bool, default=str_to_bool(os.environ.get("forced_fetch_webpage", "false")))

parser.add_argument("--GOOGLE_SEARCH_NUM_RESULTS", type=int, default=int(os.environ.get("GOOGLE_SEARCH_NUM_RESULTS", 5)))
parser.add_argument("--GOOGLE_SEARCH_INCLUDE_SNIPPETS", type=bool, default=str_to_bool(os.environ.get("GOOGLE_SEARCH_INCLUDE_SNIPPETS", "true")))
parser.add_argument("--GOOGLE_SEARCH_INCLUDE_CONTENT", type=bool, default=str_to_bool(os.environ.get("GOOGLE_SEARCH_INCLUDE_CONTENT", "true")))
parser.add_argument("--GOOGLE_SEARCH_CONTENT_MAX_LENGTH", type=int, default=int(os.environ.get("GOOGLE_SEARCH_CONTENT_MAX_LENGTH", 15000)))
parser.add_argument("--GOOGLE_SEARCH_LANGUAGE", type=str, default=os.environ.get("GOOGLE_SEARCH_LANGUAGE", "en"))
parser.add_argument("--GOOGLE_SEARCH_COUNTRY", type=str, default=os.environ.get("GOOGLE_SEARCH_COUNTRY", None))
parser.add_argument("--GOOGLE_SEARCH_SAFE_SEARCH", type=bool, default=str_to_bool(os.environ.get("GOOGLE_SEARCH_SAFE_SEARCH", "true")))

parser.add_argument("--FETCH_INCLUDE_IMAGES", type=bool, default=True)
parser.add_argument("--FETCH_MAX_LENGTH", type=int, default=None)

parser.add_argument("--model", type=str, default="gpt-4o")
parser.add_argument("--question", type=str)

args = parser.parse_args()

async def main() -> None:
    print("main() -> started...")  # Track when main starts

    # Initialize the model client


    ### Model Selection Confirmation
    print(f"main() -> Using model: {args.model}")
    ###

    if args.model == "deepseek-chat":
        model_client = OpenAIChatCompletionClient(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DeepSeek_API_KEY"),
            model_capabilities={
                "vision": True,
                "function_calling": True,
                "json_output": True,
            },
        )
    elif args.model == "gpt-4o":
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o", 
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    ### Research Question Input
    print(f"main() -> Research Question: {args.question}")
    ###

    google_search_tool = google_search_tool  
    fetch_webpage_tool = fetch_webpage_tool 

    # (Prommin - Modify) Add
    if args.forced_google_search_params:
        async def forced_google_search(
            query: str,
            include_snippets: bool = True,  # LLM can decide
            include_content: bool = True,  # LLM can decide
            country: Optional[str] = None,
            language: str = "en",
            safe_search = True
        ) -> List[Dict[str, str]]:
            """Google Search with num_results=2 and content_max_length=5000 forced on every execution."""

            # Explicitly enforce values before calling google_search
            forced_params = {
                "num_results": args.GOOGLE_SEARCH_NUM_RESULTS,
                "content_max_length": args.GOOGLE_SEARCH_CONTENT_MAX_LENGTH,
            }

            return await google_search(
                query=query,
                include_snippets=include_snippets,  # LLM-controlled
                include_content=include_content,  # LLM-controlled
                language=language,
                country=country,
                safe_search=safe_search,
                **forced_params  # Ensuring these parameters are applied every time
            )

            # Register Forced FunctionTool
        google_search_tool = FunctionTool(
            func=forced_google_search,
            description="""
            Perform Google searches with enforced num_results=2 and content_max_length=5000.
            LLM can choose whether to include snippets and webpage content.
            """,
            global_imports=[
                ImportFromModule("typing", ["List", "Dict", "Optional"]),
                "os",
                "requests",
                "html2text",
                ImportFromModule("bs4", ["BeautifulSoup"]),
                ImportFromModule("urllib.parse", ["urljoin"])
            ]
        )
        google_search_tool = forced_google_search_tool

        
    if args.forced_fetch_webpage_params:
            async def forced_fetch_webpage(
                url: str,
                include_images: bool = args.FETCH_INCLUDE_IMAGES,
                max_length: Optional[int] = args.FETCH_MAX_LENGTH,
                headers: Optional[Dict[str, str]] = None
            ) -> str:
                # Explicitly enforce values before calling google_search
                fetch_forced_params = {
                    "include_images": args.FETCH_INCLUDE_IMAGES,
                    "max_length": args.FETCH_MAX_LENGTH,
                }

                return await fetch_webpage(
                    url=url,
                    headers=headers,  # LLM-controlled
                    **fetch_forced_params  # Ensuring these parameters are applied every time
                )

            fetch_webpage_tool = FunctionTool(
                func=forced_fetch_webpage,
                description="Fetch a webpage and convert it to markdown format, with options for including images and limiting length",
                global_imports=[
                    "os",
                    "html2text",
                    ImportFromModule("typing", ("Optional", "Dict")),
                    "requests",
                    ImportFromModule("bs4", ("BeautifulSoup",)),
                    ImportFromModule("html2text", ("HTML2Text",)),
                    ImportFromModule("urllib.parse", ("urljoin",))
                ]
            )
            fetch_webpage_tool = forced_fetch_webpage_tool
            #####



    # Create the Research Assistant agent
    research_assistant = AssistantAgent(
        name="research_assistant",
        description="A research assistant that performs web searches and analyzes information",
        model_client=model_client,
        tools=[google_search_tool, fetch_webpage_tool],
        system_message="""You are a research assistant focused on finding accurate information.
        Use the google_search tool to find relevant information.
        Break down complex queries into specific search terms.
        Always verify information across multiple sources when possible.
        When you find relevant information, explain why it's relevant and how it connects to the query. When you get feedback from the a verifier agent, use your tools to act on the feedback and make progress."""
    )

    # Create the Verifier agent
    verifier = AssistantAgent(
        name="verifier",
        description="A verification specialist who ensures research quality and completeness",
        model_client=model_client,
        system_message="""You are a research verification specialist.
        Your role is to:
        1. Verify that search queries are effective and suggest improvements if needed
        2. Explore drill downs where needed e.g, if the answer is likely in a link in the returned search results, suggest clicking on the link
        3. Suggest additional angles or perspectives to explore. Be judicious in suggesting new paths to avoid scope creep or wasting resources, if the task appears to be addressed and we can provide a report, do this and respond with "TERMINATE".
        4. Track progress toward answering the original question
        5. When the research is complete, provide a detailed summary in markdown format
        
        For incomplete research, end your message with "CONTINUE RESEARCH". 
        For complete research, end your message with APPROVED.
        
        Your responses should be structured as:
        - Progress Assessment
        - Gaps/Issues (if any)
        - Suggestions (if needed)
        - Next Steps or Final Summary"""
    )

    summary_agent = AssistantAgent(
        name="summary_agent",
        description="A summary agent that provides a detailed markdown summary of the research as a report to the user.",
        model_client=model_client,
        system_message="""You are a summary agent. Your role is to provide a detailed markdown summary of the research as a report to the user. Your report should have a reasonable title that matches the research question and should summarize the key details in the results found in natural an actionable manner. The main results/answer should be in the first paragraph.
        Your report should end with the word "TERMINATE" to signal the end of the conversation."""
    )

    # Set up termination conditions
    text_termination = TextMentionTermination("TERMINATE")
    max_messages = MaxMessageTermination(max_messages=30)
    termination = text_termination | max_messages

    # Create the selector prompt
    selector_prompt = """You are coordinating a research team by selecting the team member to speak/act next. The following team member roles are available:
    {roles}.
    The research_assistant performs searches and analyzes information.
    The verifier evaluates progress and ensures completeness.
    The summary_agent provides a detailed markdown summary of the research as a report to the user.

    Given the current context, select the most appropriate next speaker.
    The research_assistant should search and analyze.
    The verifier should evaluate progress and guide the research (select this role is there is a need to verify/evaluate progress). You should ONLY select the summary_agent role if the research is complete and it is time to generate a report.

    Base your selection on:
    1. Current stage of research
    2. Last speaker's findings or suggestions
    3. Need for verification vs need for new information
        
    Read the following conversation. Then select the next role from {participants} to play. Only return the role.

    {history}

    Read the above conversation. Then select the next role from {participants} to play. ONLY RETURN THE ROLE."""

    # Create the team
    team = SelectorGroupChat(
        participants=[research_assistant, verifier, summary_agent],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=True
    )

    task = args.question


    print("main() -> Running team chat stream...")
    #await Console(team.run_stream(task=task))
    await Console(team.run_stream(task=task))
    print("main() -> Finished team chat stream.")

asyncio.run(main())