import os, time, logging
from llm.tools import prompt_utils as pu
from llm.tools import toolchain as tc
from langchain_community.chat_models import ChatOpenAI
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import streamlit as st
from streamlit.runtime import get_instance
from streamlit.runtime.scriptrunner import get_script_run_ctx

sb_initial_state = "expanded"
openai_api_key = st.secrets.openai_api_key
models = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"]
selected_model = models[0]
log = st.logger.get_logger(__name__)

def app_setup():
    st.set_page_config(
        page_title="LLM Agent",
        page_icon=":earth_americas:",
        layout="wide",
        initial_sidebar_state=sb_initial_state,
    )
    init_sidebar()

def init_sidebar():
    with st.sidebar.expander("Agent Settings", expanded=True):
        selected_model = st.selectbox("Choose a GPT Model:", models, key="active_model", on_change=gpt_model_change)
        st.write(f"Using GPT Model:  ***{selected_model}***")


def gpt_model_change():    
    """
    Callback function to handle model change in the sidebar.
    """
    log.info(f"gpt_model_change to => {st.session_state.active_model}")
    # Reinitialize agent with new model
    if valid_openai_api_key():
        st.session_state.agent = init_agent(st.session_state.active_model)  


def _get_session():
    runtime = get_instance()
    session_id = get_script_run_ctx().session_id
    session_info = runtime._session_mgr.get_session_info(session_id)
    if session_info is None:
        raise RuntimeError("Could not get your Streamlit Session object!")
        log.error(f"Could not get Streamlit Session object from runtime: {runtime.session_mgr}")
    return session_info.session

def valid_openai_api_key():
    """
    Checks if the OpenAI API key is valid. openai_api_key gets read at startup from secrets.toml.
    It is set in the session state only when passes validity check.
    """
    log.info(f"Validating API key: {openai_api_key}")
    # Check if the API key starts with 'sk-'
    if openai_api_key.startswith('sk-'):
        return True
    else:
        return False

def init_agent(model):
    log.info(f"Creating agent with model: {model}")
    model = init_chat_model(temperature=0.0, model=model, model_provider="openai")
    memory = MemorySaver()
    # bind tools
    tooled_model = model.bind_tools(tc.toolkit)
    agent = create_react_agent(tooled_model, tc.toolkit, checkpointer=memory)
    return agent

# def generate_response(input_text):
#     llm = ChatOpenAI(temperature=0.0, model="gpt-4.1-mini", openai_api_key=st.session_state.openai_api_key)
#     #st.info(llm.invoke(input_text).content)
#     return llm.invoke(input_text)

def agent_response(agent, user_prompt, session_id):
    """
    Generate response from the LLM powered Agent using the provided user prompt and session ID.
    The use of the durable session ID maintains conversational memory.
    """
    output = pu.prompt_agent(agent, user_prompt, session_id)
    # Using memory: Get proper messages
    agent_messages = output["messages"]
    return agent_messages[-1]


# ------------------------
# 
#       Main App
# 
# ------------------------
def main():
    global openai_api_key
    app_setup()
    user_session = _get_session()
    log.info(f"Created Session ID: {user_session.id}")

    if "messages" not in st.session_state:
        st.session_state.messages = []
   
    # API key not in session state
    if "openai_api_key" not in st.session_state:
        if valid_openai_api_key():
            log.info(f"Valid API key found: {openai_api_key}")
            st.session_state.openai_api_key = openai_api_key
        else:
            placeholder = st.empty()
            with placeholder.form("api_key_form"):
                openai_api_key = st.text_input('OpenAI API Key', type='password', 
                                               placeholder='sk-...', help='Enter your OpenAI API key here.')
                do_submit = st.form_submit_button("Submit")
                if do_submit:
                    if valid_openai_api_key():
                        log.info(f"Valid API key entered from api_key_form: {openai_api_key}")
                        st.session_state.openai_api_key = openai_api_key
                        log.info(f"(1) Setting ENV Var after api_key_form entry => OPENAI_API_KEY")
                        os.environ["OPENAI_API_KEY"] = openai_api_key
                        placeholder.empty()
                    else:
                        st.error('Invalid API key format! Please enter a valid OpenAI API key.', icon='🚫')
                        time.sleep(3)
                        st.rerun() # rerun script          
    # API key in session state
    else:
        openai_api_key = st.session_state.openai_api_key
        log.info(f"(2) Setting ENV Var because valid key in session => OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = openai_api_key

    # Setup initial agent
    if valid_openai_api_key() and "agent" not in st.session_state:
        st.session_state.agent = init_agent(st.session_state.active_model)

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if valid_openai_api_key():
        if prompt := st.chat_input("Ask me anything?"):
            # Add to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get agent response
            response = agent_response(st.session_state.agent, prompt, user_session.id).content
            # Add to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
    else:
        st.warning(f'OpenAI API key required to prompt ChatGPT', icon='⚠')
        log.info(f"API key blank : {openai_api_key}")

# end main()



if __name__ == "__main__":
    main()