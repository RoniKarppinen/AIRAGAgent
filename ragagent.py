from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


load_dotenv()

# Define the global default name
SUBJECT_NAME = "Bill Gates"

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

current_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(current_dir, "chroma_db")
collection_name = "bill_gates_docs"

if not os.path.exists(persist_directory):
    raise FileNotFoundError("Vector store not found. Please run 'ingest.py' first.")

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name=collection_name
)

# Create a retriever from the vector store and tavily search engine for web search.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":4}) 
tavily_engine = TavilySearchResults(max_results=2)

@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns the information from the pdf documents."""

    try:
        documents = retriever.invoke(query)

        if not documents:
            return "No relevant information found in the documents."
        
        results = []
        for i, doc in enumerate(documents):
            results.append(f"Document {i+1}:\n{doc.page_content}\n")
        
        return "\n".join(results)
    except Exception as e:
        return f"Error during document retrieval: {str(e)}"

@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for information not found in the PDF.
    Use this for recent news, current net worth, or general facts about Bill Gates 
    that might not be in the static document.
    """

    try:
        results = tavily_engine.invoke(query)

        if not results:
            return "No relevant information found on the web."
        
        formatted_results = []
        for result in results:
            formatted_results.append(
                f"Source: {result.get('url', 'Unknown URL')}\n"
                f"Content: {result.get('content', 'No content available')}\n"
            )
        return "\n---\n".join(formatted_results)
    except Exception as e:
        return f"Error during web search: {str(e)}"


tools = [retriever_tool, web_search_tool]

llm = llm.bind_tools(tools)

class RagAgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: RagAgentState):
    """Decide whether to continue based on tool calls in the last message."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


system_prompt_template = """
You are an intelligent AI assistant answering questions about {NAME}.

1. PRIMARY SOURCE: Use 'retriever_tool' to check the PDF knowledge base first.
2. SECONDARY SOURCE: Use 'web_search_tool' if the answer is not in the PDF or requires up-to-date information.

Always cite your source (For example, "According to the PDF..." or "Search results indicate...").
"""

tools_dict = {tool.name: tool for tool in tools}

def call_llm(state: RagAgentState) -> RagAgentState:
    """Function to call the LLM with the current state."""
    formatted_prompt = system_prompt_template.format(NAME=SUBJECT_NAME)
    messages = list(state["messages"])
    messages = [SystemMessage(content=formatted_prompt)] + messages
    messages = llm.invoke(messages)
    return {"messages": [messages]}

def take_action(state: RagAgentState) -> RagAgentState:
    """Function to execute the tool calls made by the LLM."""
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for tool in tool_calls:
        print(f"Calling tool: {tool['name']} with input: {tool['args']}")

        if tool["name"] not in tools_dict:
            print(f"\nTool: {tool['name']} not found.")
            result = "Error: Tool not found."
        try:
            result = tools_dict[tool["name"]].invoke(tool["args"].get("query", ""))
            print(f"Tool result length: {len(str(result))} characters")
        except Exception as e:
            print(f"Error invoking tool {tool['name']}: {str(e)}")
            result = f"Error: {str(e)}"
        # Append the tool result as a ToolMessage.
        results.append(ToolMessage(tool_call_id=tool["id"], name=tool["name"], content=str(result)))
    print(f"Total tool calls executed: {len(results)}")
    return {"messages": results}

graph = StateGraph(RagAgentState)
graph.add_node("llm_call", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges("llm_call", should_continue, {True: "retriever_agent", False: END})
graph.add_edge("retriever_agent", "llm_call")
graph.set_entry_point("llm_call")

rag_agent = graph.compile()


def run_document_agent():
    global SUBJECT_NAME
    
    print("\nRWelcome to the RAG Agent!")
    user_name = input(f"Who is the subject? (Press Enter for default: \"{SUBJECT_NAME}\"): ").strip()
    
    if user_name.lower() in ["exit", "quit"]: # Edge case
        print("Exiting RAG Agent. Goodbye!")
        return
    elif user_name:
        SUBJECT_NAME = user_name
        print(f"Subject set to: {SUBJECT_NAME}")
    else:
        print(f"Using default subject: {SUBJECT_NAME}")

    print(f"\nWelcome to the {SUBJECT_NAME} RAG Agent!")

    while True:
        user_input = input(f"\nPlease enter your question about {SUBJECT_NAME} (or 'exit'): ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting RAG Agent. Goodbye!")
            break
        
        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})
        print(f"\nResponse:\n{result['messages'][-1].content}")

# For visualization purposes only
# def visualize_graph(agent_app, filename="agent_graph.png"):
#     """
#     Visualizes the LangGraph agent and saves it as a PNG image in the 'images' folder.
#     """
#     images_dir = "images"

#     if not os.path.exists(images_dir):
#         os.makedirs(images_dir)
#         print(f"Created directory: {images_dir}")

#     # Create the full path (images/agent_graph.png)
#     filepath = os.path.join(images_dir, filename)

#     print(f"\nGenerating graph image as '{filepath}'...")
#     try:
#         # Get the graph structure and render it as a PNG
#         graph_image = agent_app.get_graph().draw_mermaid_png()
        
#         with open(filepath, "wb") as f:
#             f.write(graph_image)
            
#         print(f"Success! Graph saved to {os.path.abspath(filepath)}")
        
#     except Exception as e:
#         print(f"Could not generate image: {e}")



if __name__ == "__main__":
    # visualize_graph(rag_agent)
    run_document_agent()