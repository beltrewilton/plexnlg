import os
from typing import Dict, List, Optional

import dspy
from dotenv import dotenv_values
from pydantic import BaseModel, Field


print("CWD: ", os.getcwd())

# from .configs import retriever_model
from config import retriever_model

secret = dotenv_values('.secret')

if secret['LLM_MODEL'].startswith("llama"):
    llm = dspy.GROQ(
        model=secret['LLM_MODEL'],
        api_key=secret['GROQ_API_KEY'],
        max_tokens=int(secret['LLM_MAX_TOKEN']),
        temperature=float(secret['LLM_TEMPERATURE'])
    )
else:
    llm  = dspy.OpenAI(
        model=secret['LLM_MODEL'],
        api_key=secret['OPEN_AI_API_KEY'],
        max_tokens=int(secret['LLM_MAX_TOKEN']),
        temperature=float(secret['LLM_TEMPERATURE'])
    )

print(llm.kwargs)




import os
from typing import Dict, List, Optional

import dspy
from dotenv import dotenv_values
from pydantic import BaseModel, Field

import requests
from rich import print

class FInput(BaseModel):
    utterance: str = Field(description="User utterance input")
    current_task: str = Field(description="User current task to be completed")
    previous_state: str = Field(description="User previous state")
    current_state: str = Field(description="User current state")

class FOutput(BaseModel):
    share_link: bool = Field(description="True if: (1) you are asking the user to fill out a form or you are sharing some form, (2) the user is requesting to share the form.")
    schedule: bool = Field(description="True if: (1) If the user for some reason cannot continue with the task, ask them to schedule them and continue later, (2) The user decides to abandon the process.")
    company_question: bool = Field(description="True if: the user's intent is related to a company-related question. Respond with 'True' if the question pertains to a company, its operations, products, services, or related topics.")
    abort_scheduled_state: bool = Field(description="True if: the user agrees to continue with the current task even if it is in 'Scheduled' state.")

class FlagsSignature(dspy.Signature):
    user_input: FInput = dspy.InputField()
    output: FOutput = dspy.OutputField()

class Flags(dspy.Module):
    def __init__(self, signature: dspy.Signature, node: str):
        super().__init__()
        self.predict = dspy.TypedChainOfThought(signature=signature)

    def forward(self, user_input: FInput) -> FOutput:
        prediction: FOutput  = self.predict(user_input=user_input).output
        return prediction



class Output(BaseModel):
    response: str
    share_link: bool
    schedule: bool
    company_question: bool
    abort_scheduled_state: bool


class NRequest(BaseModel):
    utterance: str
    states: Dict[str, str]
    current_state: str
    previous_state: str
    tasks: Dict[int, str]
    current_task: str
    previous_conversation_history: List[str]
    node: str 

class NResponse(BaseModel):
    output: Output
    previous_conversation_history: Optional[List[str]] = None



def convert_structure(utterance, conversations, system_prompt):
    """
    Converts a list of conversations from a specific format to a dictionary format.

    Args:
        conversations (list): A list of conversations where each conversation is a string
                             in the format "User/Role: message" or "AI/Assistant: message".

    Returns:
        list: A list of dictionaries where each dictionary represents a conversation
              with keys 'role' and 'content'.
    """
    converted_conversations = []
    for conversation in conversations:
        role, content = conversation.split(":", 1)
        role = role.strip()
        content = content.strip()
        
        if role == "User":
            role = "user"
        elif role == "AI":
            role = "assistant"
        
        conversation_dict = {"role": role, "content": content}
        converted_conversations.append(conversation_dict)

    converted_conversations.insert(len(converted_conversations), system_prompt)
    converted_conversations.insert(len(converted_conversations), {'role': 'user', 'content': utterance})
    
    return converted_conversations


def task_tracking(current_task, guide):
    tasks = guide[0: -1]
    incompleted_tasks = tasks[guide.index(current_task): len(tasks)]
    completed_tasks = tasks[0: guide.index(current_task)]

    return (
        current_task,
        ", ".join(tasks),
        ", ".join(completed_tasks),
        ", ".join(incompleted_tasks)
    )


def generate(
        utterance: str,
        states: Dict[str, str],
        current_state: str,
        previous_state: str,
        tasks: Dict[int, str],
        current_task: str,
        previous_conversation_history: List[str],
        node: str
):
    api_endpoint = "http://localhost:11434/api/chat"
    model = "unsloth_model_llama3_3B:latest"
    
    guide = ["Talent entry form", "Grammar assessment form",  "Scripted text", "Open question", "End of task"]
    _, tasks, completed_tasks, incompleted_tasks = task_tracking(current_task, guide)

    rephrases = ""
    if current_task == "Talent entry form":
        rephrases = "Hi there! I'm excited to help you land your dream job 🎯! I'll guide you through the following steps. Let's start by filling out a quick profile to get you closer to success. Ready? Let's go! 🚀"
    if current_task == "Grammar assessment form":
        rephrases = "Just a few steps to your job! 🙌🏼\nYour next step is to fill in the assessment."
    if current_task == "Scripted text":        
        rephrases = "You've made great progress—well done! 🚀 Next, read the text aloud and send it as a voice note: `PLACEHOLDER_1`"
    if current_task == "Open question":
        rephrases = "Got your voice note! ✅  You've made substantial progress—fantastic job! 🚀 The last task involves recording a voice note (1+ minute) that thoughtfully addresses the following prompt: `PLACEHOLDER_2`"


    system_prompt = {"role": "system", "content": f"You are a task-oriented dialogue assistant. Your goal is to help users complete a sequence of tasks: [{tasks}] in a structured manner. Generate clear, polite, and contextually relevant responses based on the current task, user input, and progress. The current task is `{current_task}`. Completed tasks: [{completed_tasks}]. Incompleted tasks: [{incompleted_tasks}]."}

    user_input = f"\nuser input: {utterance}\noutput:"

    messages = convert_structure(user_input, previous_conversation_history, system_prompt)

    print(messages)

    # Define the input data
    input_data = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0}
    }

    # Send the request to the API endpoint
    response = requests.post(api_endpoint, json=input_data)

    # Print the response
    r = response.json()

    message = r['message']['content']

    nlg = Flags(FlagsSignature, "node")
    input = FInput(
        utterance=message,
        current_state= "In progress",
        previous_state= "In progress",
        current_task= current_task
    )

    with dspy.context(lm=llm):
        output = nlg(input)


    output = Output(
        response=message,
        share_link=output.share_link,
        schedule=output.schedule,
        company_question=output.company_question,
        abort_scheduled_state=output.abort_scheduled_state
    )

    previous_conversation_history.extend([f"User: {utterance}", f"AI: {output.response}"])

    return NResponse(
        output=output,
        previous_conversation_history=previous_conversation_history
    )



# messages = [
#     "User:I'm writing to inquire about the opportunity to join your team",
#     "AI:Hello! I'm thrilled to assist you in joining our team. Let's start by answering a few basic questions to create your profile and move you closer to your career goals. Are you ready to begin?",
#     "User:Talent entry form completed.",
#     "AI:You're just a few steps away from your job! 🙌🏼 Your next step is to complete the assessment.",
#     "User:I've finished the assessment.",
#     "AI:Great job on completing the assessment! You're making excellent progress towards your new job. Next, we'll move on to the scripted text task.",
#     "User:?",
#     "AI:You're just a few steps away from your job! 🙌🏼 Your next step is to fill out the assessment."
#  ]

# current_task = "Grammar assessment form"
# utterance = "Thanks"
# response = x_generate(utterance=utterance, previous_conversation_history=messages, current_task=current_task, tasks=None, states=None, current_state=None, previous_state=None, node=None)

# print(response)