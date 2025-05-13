
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


def get_prompt(user_prompt):
    human_input = HumanMessagePromptTemplate.from_template("{user_prompt}")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
        """
        You are a helpful assistant name ReActOr who uses tools to access and retrieve information.
        You should carefully consider the available tools and use them to help the user with their questions.
        You should always explain your reasoning and your actions to the user.
        Your final answer should be the full answer to the user's question.
        
        Example of a tool usage:
        Action: Search
        Action Input: "what is the capital of France?"
        Observation: "The capital of France is Paris."

        Now answer the user's question:
        """),
        human_input
    ])
    # Create prompt template
    prompt_tmpl = prompt_template.format_messages(user_prompt=user_prompt)
    return prompt_tmpl

def prompt_agent(agent, user_prompt, session_id):
    """
    Function to generate a response from the agent using the provided user prompt and Session ID.
    """
    prompt_template = get_prompt(user_prompt)
    config = {"configurable": {"thread_id": session_id}}
    response = agent.invoke({"messages": prompt_template}, config)  
    return response