import os
from typing import Dict, List, Optional

import dspy
from dotenv import dotenv_values
from pydantic import BaseModel, Field

# os.chdir("/Users/beltre.wilton/apps/tars")
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

class RelevanceInput(BaseModel):
    previous_conversation_history: str = Field(description="Previous conversation history")
    question: str = Field(description="User question")
    answer: str = Field("Answer to user question")

class RelevanceOutput(BaseModel):
    response: str = Field(description="If relevant is False: Answer that unfortunately you do not have the answer while keeping track of tasks")
    relevance: bool = Field(description="True if: confidently answers the user's question")

class RelevanceSignature(dspy.Signature):
    """to evaluate whether the answer satisfies the user's question"""
    input: RelevanceInput = dspy.InputField()
    output: RelevanceOutput = dspy.OutputField()

class Input(BaseModel):
    previous_conversation_history: str = Field(description="Previous conversation history")
    utterance: str = Field(description="User utterance input")
    previous_task: str = Field(description="User already completed task")
    current_task: str = Field(description="User current task to be completed")
    next_task: str = Field(description="User next task to be completed")
    tasks_completed: str = Field(description="Useful for validate Tasks already completed")
    previous_state: str = Field(description="User previous state")
    current_state: str = Field(description="User current state")

class Output(BaseModel):
    response: str = Field(description=f"AI response")
    share_link: bool = Field(description="True if: (1) you are asking the user to fill out a form for the first time, (2) the user is requesting to share the form.")
    schedule: bool = Field(description="True if: (1) If the user for some reason cannot continue with the task, ask them to schedule them and continue later, (2) The user decides to abandon the process.")
    company_question: bool = Field(description="True if: the user's intent is related to a company-related question. Respond with 'True' if the question pertains to a company, its operations, products, services, or related topics.")
    abort_scheduled_state: bool = Field(description="True if: the user agrees to continue with the current task even if it is in 'Scheduled' state.")

class NLGSignature(dspy.Signature):
    context: List[str] = dspy.InputField(desc="may contain relevant facts")
    user_input: Input = dspy.InputField()
    output: Output = dspy.OutputField()

class NLG(dspy.Module):
    def __init__(self, signature: dspy.Signature, node: str):
        super().__init__()
        self.predict = dspy.TypedChainOfThought(signature=signature)
        self.relevance = dspy.TypedChainOfThought(RelevanceSignature)
        self.retriever = retriever_model(node=node, table_name="company_info")

    def forward(self, user_input: Input) -> Output:
        prediction: Output  = self.predict(context=["N/A"], user_input=user_input).output
        if prediction.company_question:
            context = self.retriever(user_input.utterance)
            context = [ctx['text'] for ctx in context]
            prediction: Output = self.predict(context=context, user_input=user_input).output
            relevance_info = self.relevance(
                input=RelevanceInput(
                    previous_conversation_history=user_input.previous_conversation_history,
                    question=user_input.utterance,
                    answer=prediction.response
                )
            ).output
            if not relevance_info.relevance:
                prediction.response = relevance_info.response

        return prediction
    

def main_signature(
        index: int,
        states: list,
        current_state: str,
        previous_state: str
    ) -> str:
    task_instruct = ""
    if index == 1:
        # task_instruct = "Rephrase the following message: Welcome! the purpose here is to get to know you better. I'll guide you through a quick assessment to check your grammar and English fluency. It only takes about 10 minutes to complete! Instead of spending weeks going to an office, this assessment happens right here, right now.\n\nReady to start?"
        # task_instruct = "Rephrase the following message: Hi there! Thank you for your interest in the opportunities we have available. We’re excited to help you on your journey to landing your next great job and achieving success! 🎯 To get started, we’d like to ask you a few basic questions to create your profile and see if you’re the ideal candidate for our openings. Let’s get you closer to your dream job! Ready? Let’s go! 🚀"
        task_instruct = "Rephrase the following message: Hi there, I'm excited to help you land your dream job 🎯! I'll guide you through the following steps. Let's answer a few basic questions to create your profile and get you closer to success - ready, let's go! 🚀"
    if index == 2:
        task_instruct = "Rephrase the following message: Just a few steps to your job! 🙌🏼\nYour next step is to fill in the assessment."
    if index == 3:
        task_instruct = "Rephrase the following message: You've made great progress—well done! 🚀 Next, read the text aloud and send it as a voice note: `PLACEHOLDER_1`"
    if index == 4:
        task_instruct = "Rephrase the following message: Got your voice note! ✅  You've made substantial progress—fantastic job! 🚀 The last task involves recording a voice note (1+ minute) that thoughtfully addresses the following prompt: `PLACEHOLDER_2`"
    # if index == 5:
    #   task_instruct = "Rephrase the following message: Your voice note has landed! Well done on completing all the steps, thanks!"

    state_instruct = ""
    main_body_instruct = ""
    if previous_state == "Scheduled" and current_state != "Scheduled":
        state_instruct = "Welcome back to the user, as the previous status was 'Scheduled'."
    elif current_state == "Scheduled" and previous_state == "Scheduled":
        main_body_instruct = "Ask the user if they want to continue with the current task."
        task_instruct = ""
    elif current_state == "Scheduled":
        main_body_instruct = "Thanks the user for scheduling, see you later."
        task_instruct = ""
    elif current_state == "In progress":
        main_body_instruct = """Ask the user to complete the following sequence tasks:
- Talent entry form
Fields: Profile
Delivery: Share in this chat
IMPORTANT: The form is self-contained. You are not informed about its content.

- Grammar Assessment form
Fields: Two questions
Delivery:  Share in this chat
IMPORTANT: The form is self-contained. You are not informed about its content.

- Scripted text
Fields: read aloud the text `PLACEHOLDER_1` and share as a voice note
Delivery:  Share in this chat

- Open question
Fields: answer the question `PLACEHOLDER_2` aloud and share as a voice note
Delivery:  Share in this chat

- End_of_Task


Your task is to validate that the sequence of tasks are completed by the user, If current task is NOT completed, ask again.
Respond to any concerns while keeping track of tasks.
If the user decides to abandon the process, politely remind them of the excellent job opportunity at hand. Highlight the career growth, supportive team, and exciting challenges that align with their skills. Reassure them that continuing could be a significant step forward in their career. Offer to address any concerns they may have and emphasize that opportunities like this are rare.
Ask the user to schedule if: (1) the user for some reason cannot continue with the task, ask them to schedule them and continue later, (2) The user decides to abandon the process.
    """
        
    if current_state == "Completed" and previous_state != "Completed":
        task_instruct = "Rephrase the following message: Your voice note has landed! Well done on completing all the steps, thanks!"
        task_instruct = f"{task_instruct}. OPTIONALLY: Only if you haven't received the video yet, Ask the user if they want to send a final video with their expectations, the video should not be longer than 15 seconds."
    elif current_state == "Completed" and previous_state == "Completed":
        main_body_instruct = "At this point the user has completed the task sequence, If the user asks for additional information about the process, respond shortly and politely and provide the necessary details. If no further information is needed, kindly say goodbye."
        main_body_instruct = f"{main_body_instruct} OPTIONALLY: Only if you haven't received the video yet, Ask the user if they want to send a final video with their expectations, the video should not be longer than 15 seconds."
        task_instruct = ""

    signature = f"""You are Maria, a virtual assistant at a call center recruiting company.
You are only able to answer in English.
If the user uses a language different from English, ask politely to switch to English.

{main_body_instruct}

{task_instruct}

{state_instruct}
            """
    return signature


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

def generate(
        utterance: str,
        states: Dict[str, str],
        current_state: str,
        previous_state: str,
        tasks: Dict[int, str],
        current_task: str,
        previous_conversation_history: List[str],
        node: str
    ) -> NResponse:

    from rich import print 

    print("utterance:", utterance)
    print("states:", states)
    print("current_state:", current_state)
    print("previous_state:", previous_state)
    print("tasks:", tasks)
    print("current_task:", current_task)
    print("previous_conversation_history:", previous_conversation_history)
    print("node:", node)

    index = next(key for key, value in tasks.items() if value == current_task)
    user_input = Input(
        utterance=utterance,
        previous_task=tasks.get(index - 1) if index > 1 else "",
        current_task=current_task,
        next_task=tasks.get(index + 1) if index < len(tasks) else "",
        tasks_completed="\n ".join([tasks.get(i) for i in range(1, index)]),
        previous_conversation_history="\n".join(previous_conversation_history),
        current_state=current_state,
        previous_state=previous_state
    )
    NLGSignature.__doc__ = main_signature(
        index=index,
        states=states,
        current_state=current_state,
        previous_state=previous_state,
    )
    
    nlg = NLG(signature=NLGSignature, node=node)
    # nlg.load("nlg_miprov2_optimized_20241125_163853")

    with dspy.context(lm=llm):
        output = nlg(user_input=user_input)
    previous_conversation_history.extend(
        [f"User: {utterance}", f"AI: {output.response}"]
    )
    
    resp = NResponse(
        output=output,
        previous_conversation_history=previous_conversation_history
    )

    print("Response:", resp)

    return resp
