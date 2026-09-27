import os
import streamlit as st
from dotenv import load_dotenv

# ==================================================
# IMPORTANT FIX FOR CREWAI + GROQ
# ==================================================
import crewai.llms.cache as crewai_cache

# Disable cache_breakpoint because Groq does not support it
crewai_cache.mark_cache_breakpoint = lambda msg: msg

from crewai import Agent, Task, Crew, Process, LLM


# ==================================================
# STREAMLIT PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Healthcare AI Research Crew",
    page_icon="🤖",
    layout="wide"
)


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY is missing from your .env file.")
    st.stop()


# ==================================================
# GROQ LLM CONFIGURATION
# ==================================================

llm = LLM(
    model="groq/openai/gpt-oss-20b",
    api_key=groq_api_key,
    temperature=0.7
)


# ==================================================
# RESEARCHER AGENT
# ==================================================

researcher = Agent(
    role="Healthcare AI Researcher",

    goal=(
        "Research and analyze questions related to artificial "
        "intelligence, healthcare, medical technology, and "
        "emerging AI trends."
    ),

    backstory=(
        "You are an experienced AI and healthcare technology "
        "researcher. You analyze questions carefully, identify "
        "important information, and provide accurate explanations "
        "in simple language."
    ),

    llm=llm,

    verbose=True,

    allow_delegation=False
)


# ==================================================
# ANSWER AGENT
# ==================================================

answer_agent = Agent(
    role="Healthcare AI Expert",

    goal=(
        "Use the research provided by the researcher to answer "
        "the user's question clearly, accurately, and professionally."
    ),

    backstory=(
        "You are an expert technology communicator specializing "
        "in artificial intelligence and healthcare. You transform "
        "research into useful answers that are easy for a general "
        "audience to understand."
    ),

    llm=llm,

    verbose=True,

    allow_delegation=False
)


# ==================================================
# STREAMLIT SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==================================================
# STREAMLIT HEADER
# ==================================================

st.title("🤖 Healthcare AI Research Crew")

st.write(
    "Ask a question about Artificial Intelligence in Healthcare. "
    "The Researcher Agent analyzes your question and the Expert "
    "Agent produces the final answer."
)

st.divider()


# ==================================================
# DISPLAY CHAT HISTORY
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==================================================
# USER QUESTION PLACEHOLDER
# ==================================================

user_question = st.chat_input(
    "💬 Ask your healthcare AI question..."
)


# ==================================================
# PROCESS USER QUESTION
# ==================================================

if user_question:

    # ----------------------------------------------
    # Display user's question
    # ----------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)


    # ----------------------------------------------
    # AI response
    # ----------------------------------------------

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "🔎 Researcher is analyzing your question..."
            ):

                # ==========================================
                # RESEARCH TASK
                # ==========================================

                research_task = Task(
                    description=f"""
                    The user has asked the following question:

                    "{user_question}"

                    Analyze this question carefully.

                    Research and explain the relevant information
                    related to artificial intelligence and healthcare.

                    Consider:

                    1. What the question is asking
                    2. Important concepts
                    3. Current applications
                    4. Benefits
                    5. Risks or limitations
                    6. Practical examples
                    7. Future developments

                    Provide useful research that another AI expert
                    can use to formulate the final answer.

                    Do not unnecessarily discuss unrelated topics.
                    """,

                    expected_output="""
                    A clear and detailed research analysis directly
                    related to the user's question.
                    """,

                    agent=researcher
                )


                # ==========================================
                # ANSWER TASK
                # ==========================================

                answer_task = Task(
                    description=f"""
                    The user asked:

                    "{user_question}"

                    Use the research produced by the Researcher Agent
                    to answer the user's question.

                    Requirements:

                    - Directly answer the question.
                    - Use clear professional English.
                    - Explain technical concepts simply.
                    - Include examples when useful.
                    - Mention important benefits and limitations
                      when relevant.
                    - Do not add unnecessary information.
                    - Do not say that you are an AI agent.
                    - Structure the answer using headings or bullet
                      points when appropriate.

                    The answer should be useful to a business or
                    general audience.
                    """,

                    expected_output="""
                    A clear, accurate, professional answer to the
                    user's question based on the research.
                    """,

                    agent=answer_agent,

                    context=[research_task]
                )


                # ==========================================
                # CREATE CREW
                # ==========================================

                question_crew = Crew(
                    agents=[
                        researcher,
                        answer_agent
                    ],

                    tasks=[
                        research_task,
                        answer_task
                    ],

                    process=Process.sequential,

                    verbose=True
                )


                # ==========================================
                # RUN CREW
                # ==========================================

                result = question_crew.kickoff()


            # ----------------------------------------------
            # Display final answer
            # ----------------------------------------------

            final_answer = str(result)

            st.markdown(final_answer)


            # ----------------------------------------------
            # Save assistant response
            # ----------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer
                }
            )


        except Exception as error:

            error_message = (
                "❌ An error occurred while processing "
                "your question."
            )

            st.error(error_message)

            st.exception(error)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("🤖 Healthcare AI Crew")

    st.write(
        "This application uses two CrewAI agents:"
    )

    st.markdown(
        """
        **🔎 Researcher Agent**

        Analyzes the user's question and produces relevant research.

        **✍️ Healthcare AI Expert**

        Uses the research to create the final answer.
        """
    )

    st.divider()

    st.subheader("💡 Example Questions")

    st.markdown(
        """
        - How is AI being used to detect cancer?

        - What is generative AI in healthcare?

        - How can AI help doctors?

        - What are the risks of AI diagnosis?

        - How is AI being used in drug discovery?

        - Can AI replace doctors?

        - How does AI medical imaging work?

        - What is the future of AI in healthcare?
        """
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()