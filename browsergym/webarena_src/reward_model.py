
from langchain.schema import SystemMessage, HumanMessage

################# Evaluation for Action Matching ##############################
import sys
sys.path.append('./')
from demo_agent.agents.legacy.utils.chat_api import ChatModelArgs


def format_eval_prompt(pred_action, ref_action):
    eval_prompt = f'''
You will be given
1). **reference action** which indicates an correct action.
2). **predicted action** which is predicted by assistant agent

Your task is to assess whether the message in **predicted action** is semantically aligned with message in the **reference action**.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Alignment = 1: the predicted action is semantically aligned with the reference action.
    send_msg_to_user('30%') and send_msg_to_user('The percentage of amount of pending orders among entire orders is 30%') are semantically aligned.
    click('a34') and click('a34', button='left') is semantically aligned.

Alignment = 0: the predicted action is semantically not aligned with the reference action.
    send_msg_to_user('$25') and send_msg_to_user('The requested value is $29') are not semantically aligned.
    click('a34') and click('a34', button='left') is semantically aligned.

Evaluation Steps:
1. Write a simple feedback that assess whether the predicted action is semantically aligned with the reference action. 
2. After writing a feedback, write a score that is 0 or 1. You should refer to the Evaluation Criteria.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number among 0 or 1)"
4. Please do not generate any other opening, closing, and explanations.

**reference action**: {ref_action}
**predicted action**: {pred_action}

Feedback:
'''
    eval_system_prompt = 'Your task is to evaluate whether the given two action commands are semantically aligned.'
    return eval_prompt, eval_system_prompt


def action_alignment(pred_action, ref_action):
    eval_model_args = ChatModelArgs(
                model_name='openai/gpt-4o-2024-08-06',#'googleai/gemini-1.5-flash-latest',
                max_total_tokens=128_000,  # "Maximum total tokens for the chat model."
                max_input_tokens=126_488,  # "Maximum tokens for the input to the chat model."
                max_new_tokens=2_000,  # "Maximum total tokens for the chat model."
                temperature = 0.0,
                top_p = 1e-8
            )
    eval_llm = eval_model_args.make_chat_model()
    eval_prompt, eval_system_prompt = format_eval_prompt(pred_action, ref_action)
    chat_messages = [
                SystemMessage(content=eval_system_prompt),
                HumanMessage(content=eval_prompt)
            ]
    # sample candidate 
    output = eval_llm.invoke(chat_messages).content
    feedback = output.split('[RESULT]')[0].strip()
    score = output.split('[RESULT]')[-1].strip()
    try:
        reward = int(score)
    except:
        reward = 1.0
    
    if reward == 1.0:
        return True
    else:
        return False
    





