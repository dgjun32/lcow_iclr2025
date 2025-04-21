
def format_rephrase_prompt(domain_info, goal, history, observation):
    rephrase_system_prompt = f'''You are an agent tasked with extracting and refine a subset of the webpage's observations based on the content of the page and user instructions. 
Perform the following tasks based on the provided [Information source], including user instructions, interaction history, and the AXTree observation at the current time step. 
First, provide high-level reasoning for the next action by analyzing the provided information. 
Second, extract relevant elements based on your high-level reasoning.
'''
    rephrase_prompt = f'''[General instructions]
You are currently on the {domain_info} website.
Your task is to generate a "Reasoning" and a "Refined observation" based on the provided inputs.

First, review the "User instruction" and "History of interactions" and, then, generate the "Reasoning".
Analyze the progress made so far, and provide a rationale for the next steps needed to efficiently accomplish the user instruction on the {domain_info} website.

Second, refine the "AXTree observation at the current time step" into a "Refined observation".
Extract a subset of the AXTree observation (e.g., chart, table, menu items) that contains necessary information for completing the user instruction, and explain the extracted elements.
Ensure that the information on the elements (e.g., numeric element ID) is correctly included.

Please follow the format in the [Reasoning & Refinement example] carefully.

[Information source]
# User instruction
{goal}

# History of interactions
{history}

# AXTree observation at the current time step
{observation}

[Reasoning & Refinement example]
# Abstract example
Here is an abstract version of the answer, describing the content of each tag. 
Make sure you follow this structure, but replace the content with your own answer:

<reasoning>  
Think step by step. Based on the "User instruction," "History of interaction," and "AXTree observation at the current time step":  
1. Provide a high-level description of the "AXTree observation at the current time step."  
2. Based on the "User instruction" and "History of interaction," track your progress and provide your reasoning on the next action needed to accomplish the "User instruction."  
</reasoning>  
  
<extraction>
Based on your reasoning, identify the elements (e.g., links, buttons, static text, table row, chart) to focus on.
Then, explain the semantics and functionalities of each extracted element.
Ensure that:
You do not alter the structure of the AXTree observation.
You extract the element ID (id in [ ]) accurately without any errors.
When extracting chart or table, you must extract the entire chart or table to avoid any confusion or loss of information.
</extraction>
'''
    return rephrase_prompt, rephrase_system_prompt


def format_rephrase_prompt_retry(domain_info, goal, history, observation, action):
    rephrase_system_prompt = f'''You are an agent tasked with extracting and refine a subset of the webpage's observations based on the content of the page and user instructions. 
Perform the following tasks based on the provided [Information source], including user instructions, interaction history, and the AXTree observation at the current time step. 
First, provide high-level reasoning for the next action by analyzing the provided information. 
Second, extract relevant elements based on your high-level reasoning.
'''
    rephrase_prompt = f'''[General instructions]
You are currently on the {domain_info} website.
Your task is to generate a "Reasoning" and a "Refined observation" based on the provided inputs.

First, review the "User instruction" and "History of interactions" and, then, generate the "Reasoning".
Analyze the progress made so far, and provide a rationale for the next steps needed to efficiently accomplish the user instruction on the {domain_info} website.

Second, refine the "AXTree observation at the current time step" into a "Refined observation".
Extract a subset of the AXTree observation (e.g., chart, table, menu items) that contains necessary information for completing the user instruction, and explain the extracted elements.
Ensure that the information on the elements (e.g., numeric element ID) is correctly included.

You may refer to the Hints, which consists of the ground truth next action, but do not explicitly mention these hints in your output.

Please follow the format in the [Reasoning & Rephrase Example] carefully, and avoid leaving any empty strings.

[Information source]
# User instruction
{goal}

# History of interactions
{history}

# AXTree observation at the current time step
{observation}

# Hint
## Ground-truth next action
{action}

[Reasoning & Refinement example]
# Abstract example
Here is an abstract version of the answer, describing the content of each tag. 
Make sure you follow this structure, but replace the content with your own answer:

<reasoning>  
Think step by step. Based on the "User instruction," "History of interaction," and "AXTree observation at the current time step":  
1. Provide a high-level description of the "AXTree observation at the current time step."  
2. Based on the "User instruction" and "History of interaction," track your progress and provide your reasoning on the next action needed to accomplish the "User instruction."  
</reasoning>  
  
<extraction>
Based on your reasoning, identify the elements (e.g., links, buttons, static text, table row, chart) to focus on.
Then, explain the semantics and functionalities of each extracted element.
Ensure that:
You do not alter the structure of the AXTree observation.
You extract the element ID (id in [ ]) accurately without any errors.
When extracting chart or table, you must extract the entire chart or table to avoid any confusion or loss of information.
</extraction>
'''
    return rephrase_prompt, rephrase_system_prompt


def format_action_prompt_for_datacollection(goal, history, plan, rep_observation):
    action_system_prompt = f'''You are an agent trying to solve a web task based on the content of the page anda user instructions. 
You can interact with the page and explore. 
Each time you submit an action it will be sent to the browser and you will receive a new page.
    '''

    action_prompt = f'''
# Instructions                                                                                                                                                                                                                                                                                           
Review the current state of the page and all other information to find the best possible next action to accomplish your goal. 
Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

## Goal:
{goal}

## History of Interaction:
{history}

# Refined observation of current step:
## Reasoning
{plan}

## Focused AXTree observation
{rep_observation}

# Action space:
                                                                          
13 different types of actions are available.                                                                                                  
                                                                          
noop(wait_ms: float = 1000)
    Description: Do nothing, and optionally wait for the given time (in milliseconds).
    Examples:
        noop()

        noop(500)

send_msg_to_user(text: str)
    Description: Send a message to the user. You should send a short answer as a message and do not ask questions through message.
    Examples:
        send_msg_to_user(\'the city was built in 1751.\')

        send_msg_to_user(\'Yes\')

        send_msg_to_user(\'No\')

        send_msg_to_user(\'31112\')
        
        send_msg_to_user(\'Yoshua Bengio\')

scroll(delta_x: float, delta_y: float)
    Description: Scroll horizontally and vertically. Amounts in pixels, positive for right or down scrolling, negative for left or up scrolling. Dispatches a wheel event.
    Examples:
        scroll(0, 200)

        scroll(-50.2, -100.5)

fill(bid: str, value: str)
    Description: Fill out a form field. It focuses the element and triggers an input event with the entered text. It works for <input>, <textarea> and [contenteditable] elements.
    Examples:
        fill('237', 'example value')

        fill('45', 'multi-line\nexample')

        fill('a12', 'example with "quotes"')

select_option(bid: str, options: str | list[str])
    Description: Select one or multiple options in a <select> element. You can specify option value or label to select. Multiple options can be selected.
    Examples:
        select_option('48', 'blue')

        select_option('48', ['red', 'green', 'blue'])
        
click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'Meta', 'Shift']] = [])
    Description: Click an element.
    Examples:
        click('51')

        click('b22', button='right')

        click('48', button='middle', modifiers=['Shift'])

dblclick(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'Meta', 'Shift']] = [])
    Description: Double click an element.
    Examples:
        dblclick('12')

        dblclick('ca42', button='right')

        dblclick('178', button='middle', modifiers=['Shift'])

hover(bid: str)
    Description: Hover over an element.
    Examples:
        hover('b8')

press(bid: str, key_comb: str)
    Description: Focus the matching element and press a combination of keys. It accepts the logical key names that are emitted in the keyboardEvent.key property of the keyboard events: Backquote, Minus, Equal, Backslash, Backspace, Tab, Delete, Escape, ArrowDown, End, Enter, Home, Insert, PageDow
n, PageUp, ArrowRight, ArrowUp, F1 - F12, Digit0 - Digit9, KeyA - KeyZ, etc. You can alternatively specify a single character you'd like to produce such as "a" or "#". Following modification shortcuts are also supported: Shift, Control, Alt, Meta.
    Examples:
        press('88', 'Backspace')

        press('a26', 'Control+a')

        press('a61', 'Meta+Shift+t')

focus(bid: str)
    Description: Focus the matching element.
    Examples:
        focus('b455')

clear(bid: str)
    Description: Clear the input field.
    Examples:
        clear('996')

drag_and_drop(from_bid: str, to_bid: str)
    Description: Perform a drag & drop. Hover the element that will be dragged. Press left mouse button. Move mouse to the element that will receive the drop. Release left mouse button.
    Examples:
        drag_and_drop('56', '498')

upload_file(bid: str, file: str | list[str])
    Description: Click an element and wait for a "filechooser" event, then select one or multiple input files for upload. Relative file paths are resolved relative to the current working directory. An empty list clears the selected files.
    Examples:
        upload_file('572', 'my_receipt.pdf')

        upload_file('63', ['/home/bob/Documents/image.jpg', '/home/bob/Documents/file.zip'])

Only a single action can be provided at once. Example:
fill('a12', 'example with "quotes"')
Multiple actions are meant to be executed sequentially without any feedback from the page.
Don't execute multiple actions at once if you need feedback from the page. 


# Abstract Example

Here is an abstract version of the answer with description of the content of
each tag. Make sure you follow this structure, but replace the content with your
answer:

<think>
Think step by step. If you need to make calculations such as coordinates, write them here. Describe the effect
that your previous action had on the current content of the page.
</think>

<action>
One single action to be executed. You can only use one action at a time.
</action>

# Concrete Example

Here is a concrete example of how to format your answer.
Make sure to follow the template with proper tags:

<think>
My memory says that I filled the first name and last name, but I can't see any
content in the form. I need to explore different ways to fill the form. Perhaps
the form is not visible yet or some fields are disabled. I need to replan. 
</think>

<action>
fill('a12', 'example with "quotes"')
</action>
'''
    return action_prompt, action_system_prompt



def hf_format_rephrase_prompt(domain_info, goal, history, observation):
    rephrase_system_prompt = f'''You are an agent tasked with extracting and rephrasing a subset of the webpage's observations based on the content of the page and user instructions. 
'''
    rephrase_prompt = f'''[General instructions]
You are currently on the {domain_info} website.
Your task is to generate a "Reasoning" and a "Refined observation" based on the provided inputs.

First, review the "User instruction" and "History of interactions" and, then, generate the "Reasoning".
Analyze the progress made so far, and provide a rationale for the next steps needed to efficiently accomplish the user instruction on the {domain_info} website.

Second, refine the "AXTree observation at the current time step" into a "Refined observation".
Select a subset of the AXTree observation that is essential for completing the user instruction and provide explanations for the corresponding elements in the selected subset.

[Information source]
# User instruction
{goal}

# History of interactions
{history}

# AXTree observation at the current time step
{observation}
'''
    return rephrase_prompt, rephrase_system_prompt


def hf_format_action_prompt(instruction, history, rep_observation):
    action_prompt = f'''
{instruction}

{history}

# Observation of current step
{rep_observation}

# Action space:
                                                                          
13 different types of actions are available.                                                                                                        
                                                                          
noop(wait_ms: float = 1000)
    Description: Do nothing, and optionally wait for the given time (in milliseconds).
    Examples:
        noop()

        noop(500)

send_msg_to_user(text: str)
    Description: Send a message to the user. You should send a short answer as a message and do not ask questions through message.
    Examples:
        send_msg_to_user(\'the city was built in 1751.\')

        send_msg_to_user(\'Yes\')

        send_msg_to_user(\'No\')

        send_msg_to_user(\'31112\')
        
        send_msg_to_user(\'Yoshua Bengio\')

scroll(delta_x: float, delta_y: float)
    Description: Scroll horizontally and vertically. Amounts in pixels, positive for right or down scrolling, negative for left or up scrolling. Dispatches a wheel event.
    Examples:
        scroll(0, 200)

        scroll(-50.2, -100.5)

fill(bid: str, value: str)
    Description: Fill out a form field. It focuses the element and triggers an input event with the entered text. It works for <input>, <textarea> and [contenteditable] elements.
    Examples:
        fill('237', 'example value')

        fill('45', 'multi-line\nexample')

        fill('a12', 'example with "quotes"')

select_option(bid: str, options: str | list[str])
    Description: Select one or multiple options in a <select> element. You can specify option value or label to select. Multiple options can be selected.
    Examples:
        select_option('48', 'blue')

        select_option('48', ['red', 'green', 'blue'])
        
click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'Meta', 'Shift']] = [])
    Description: Click an element.
    Examples:
        click('51')

        click('b22', button='right')

        click('48', button='middle', modifiers=['Shift'])

dblclick(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'Meta', 'Shift']] = [])
    Description: Double click an element.
    Examples:
        dblclick('12')

        dblclick('ca42', button='right')

        dblclick('178', button='middle', modifiers=['Shift'])

hover(bid: str)
    Description: Hover over an element.
    Examples:
        hover('b8')

press(bid: str, key_comb: str)
    Description: Focus the matching element and press a combination of keys. It accepts the logical key names that are emitted in the keyboardEvent.key property of the keyboard events: Backquote, Minus, Equal, Backslash, Backspace, Tab, Delete, Escape, ArrowDown, End, Enter, Home, Insert, PageDow
n, PageUp, ArrowRight, ArrowUp, F1 - F12, Digit0 - Digit9, KeyA - KeyZ, etc. You can alternatively specify a single character you'd like to produce such as "a" or "#". Following modification shortcuts are also supported: Shift, Control, Alt, Meta.
    Examples:
        press('88', 'Backspace')

        press('a26', 'Control+a')

        press('a61', 'Meta+Shift+t')

focus(bid: str)
    Description: Focus the matching element.
    Examples:
        focus('b455')

clear(bid: str)
    Description: Clear the input field.
    Examples:
        clear('996')

drag_and_drop(from_bid: str, to_bid: str)
    Description: Perform a drag & drop. Hover the element that will be dragged. Press left mouse button. Move mouse to the element that will receive the drop. Release left mouse button.
    Examples:
        drag_and_drop('56', '498')

upload_file(bid: str, file: str | list[str])
    Description: Click an element and wait for a "filechooser" event, then select one or multiple input files for upload. Relative file paths are resolved relative to the current working directory. An empty list clears the selected files.
    Examples:
        upload_file('572', 'my_receipt.pdf')

        upload_file('63', ['/home/bob/Documents/image.jpg', '/home/bob/Documents/file.zip'])

Only a single action can be provided at once. Example:
fill('a12', 'example with "quotes"')
Multiple actions are meant to be executed sequentially without any feedback from the page.
Don't execute multiple actions at once if you need feedback from the page. 


# Abstract Example

Here is an abstract version of the answer with description of the content of
each tag. Make sure you follow this structure, but replace the content with your
answer:

<think>
Think step by step. If you need to make calculations such as coordinates, write them here. Describe the effect
that your previous action had on the current content of the page.
</think>

<action>
One single action to be executed. You can only use one action at a time.
</action>

# Concrete Example

Here is a concrete example of how to format your answer.
Make sure to follow the template with proper tags:

<think>
My memory says that I filled the first name and last name, but I can't see any
content in the form. I need to explore different ways to fill the form. Perhaps
the form is not visible yet or some fields are disabled. I need to replan. 
</think>

<action>
fill('a12', 'example with "quotes"')
</action>
'''
    return action_prompt


####################### Action Prompt for self-rephrasing #################################
def format_action_prompt(instruction, history, plan, rep_observation):
    action_prompt = f'''
{instruction}

{history}

# Refined observation of current step:
## Reasoning
{plan}

## Focused AXTree observation
{rep_observation}

# Action space:
                                                                          
13 different types of actions are available.                                                                                                        
                                                                          
noop(wait_ms: float = 1000)
    Description: Do nothing, and optionally wait for the given time (in milliseconds).
    Examples:
        noop()

        noop(500)

send_msg_to_user(text: str)
    Description: Send a message to the user. You should send a short answer as a message and do not ask questions through message.
    Examples:
        send_msg_to_user(\'the city was built in 1751.\')

        send_msg_to_user(\'Yes\')

        send_msg_to_user(\'No\')

        send_msg_to_user(\'31112\')
        
        send_msg_to_user(\'Yoshua Bengio\')

scroll(delta_x: float, delta_y: float)
    Description: Scroll horizontally and vertically. Amounts in pixels, positive for right or down scrolling, negative for left or up scrolling. Dispatches a wheel event.
    Examples:
        scroll(0, 200)

        scroll(-50.2, -100.5)

fill(bid: str, value: str)
    Description: Fill out a form field. It focuses the element and triggers an input event with the entered text. It works for <input>, <textarea> and [contenteditable] elements.
    Examples:
        fill('237', 'example value')

        fill('45', 'multi-line\nexample')

        fill('a12', 'example with "quotes"')

select_option(bid: str, options: str | list[str])
    Description: Select one or multiple options in a <select> element. You can specify option value or label to select. Multiple options can be selected.
    Examples:
        select_option('48', 'blue')

        select_option('48', ['red', 'green', 'blue'])
        
click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'Meta', 'Shift']] = [])
    Description: Click an element.
    Examples:
        click('51')

        click('b22', button='right')

        click('48', button='middle', modifiers=['Shift'])

dblclick(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'Meta', 'Shift']] = [])
    Description: Double click an element.
    Examples:
        dblclick('12')

        dblclick('ca42', button='right')

        dblclick('178', button='middle', modifiers=['Shift'])

hover(bid: str)
    Description: Hover over an element.
    Examples:
        hover('b8')

press(bid: str, key_comb: str)
    Description: Focus the matching element and press a combination of keys. It accepts the logical key names that are emitted in the keyboardEvent.key property of the keyboard events: Backquote, Minus, Equal, Backslash, Backspace, Tab, Delete, Escape, ArrowDown, End, Enter, Home, Insert, PageDow
n, PageUp, ArrowRight, ArrowUp, F1 - F12, Digit0 - Digit9, KeyA - KeyZ, etc. You can alternatively specify a single character you'd like to produce such as "a" or "#". Following modification shortcuts are also supported: Shift, Control, Alt, Meta.
    Examples:
        press('88', 'Backspace')

        press('a26', 'Control+a')

        press('a61', 'Meta+Shift+t')

focus(bid: str)
    Description: Focus the matching element.
    Examples:
        focus('b455')

clear(bid: str)
    Description: Clear the input field.
    Examples:
        clear('996')

drag_and_drop(from_bid: str, to_bid: str)
    Description: Perform a drag & drop. Hover the element that will be dragged. Press left mouse button. Move mouse to the element that will receive the drop. Release left mouse button.
    Examples:
        drag_and_drop('56', '498')

upload_file(bid: str, file: str | list[str])
    Description: Click an element and wait for a "filechooser" event, then select one or multiple input files for upload. Relative file paths are resolved relative to the current working directory. An empty list clears the selected files.
    Examples:
        upload_file('572', 'my_receipt.pdf')

        upload_file('63', ['/home/bob/Documents/image.jpg', '/home/bob/Documents/file.zip'])

Only a single action can be provided at once. Example:
fill('a12', 'example with "quotes"')
Multiple actions are meant to be executed sequentially without any feedback from the page.
Don't execute multiple actions at once if you need feedback from the page. 


# Abstract Example

Here is an abstract version of the answer with description of the content of
each tag. Make sure you follow this structure, but replace the content with your
answer:

<think>
Think step by step. If you need to make calculations such as coordinates, write them here. Describe the effect
that your previous action had on the current content of the page.
</think>

<action>
One single action to be executed. You can only use one action at a time.
</action>

# Concrete Example

Here is a concrete example of how to format your answer.
Make sure to follow the template with proper tags:

<think>
My memory says that I filled the first name and last name, but I can't see any
content in the form. I need to explore different ways to fill the form. Perhaps
the form is not visible yet or some fields are disabled. I need to replan. 
</think>

<action>
fill('a12', 'example with "quotes"')
</action>
'''
    return action_prompt


############################################### prompt for rephrase data collection ##########################


############ Rephrase prompt for WebArena rephrase data collection ###################################

def wa_format_action_prompt_for_datacollection_iter(goal, history, rep_observation):
    action_system_prompt = f'''You are an agent trying to solve a web task based on the content of the page anda user instructions. 
You can interact with the page and explore. 
Each time you submit an action it will be sent to the browser and you will receive a new page.
    '''

    action_prompt = f'''
# Instructions                                                                                                                                                                                                                                                                                           
Review the current state of the page and all other information to find the best possible next action to accomplish your goal. 
Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

## Goal:
{goal}

{history}

# Refined observation of current step:
{rep_observation}

# Action space:
                                                                          
13 different types of actions are available.                                                                                                  
                                                                          
noop(wait_ms: float = 1000)
    Description: Do nothing, and optionally wait for the given time (in milliseconds).
    Examples:
        noop()

        noop(500)

send_msg_to_user(text: str)
    Description: Send a message to the user. You should send a short answer as a message and do not ask questions through message.
    Examples:
        send_msg_to_user(\'the city was built in 1751.\')

        send_msg_to_user(\'Yes\')

        send_msg_to_user(\'No\')

        send_msg_to_user(\'31112\')
        
        send_msg_to_user(\'Yoshua Bengio\')

scroll(delta_x: float, delta_y: float)
    Description: Scroll horizontally and vertically. Amounts in pixels, positive for right or down scrolling, negative for left or up scrolling. Dispatches a wheel event.
    Examples:
        scroll(0, 200)

        scroll(-50.2, -100.5)

fill(bid: str, value: str)
    Description: Fill out a form field. It focuses the element and triggers an input event with the entered text. It works for <input>, <textarea> and [contenteditable] elements.
    Examples:
        fill('237', 'example value')

        fill('45', 'multi-line\nexample')

        fill('a12', 'example with "quotes"')

select_option(bid: str, options: str | list[str])
    Description: Select one or multiple options in a <select> element. You can specify option value or label to select. Multiple options can be selected.
    Examples:
        select_option('48', 'blue')

        select_option('48', ['red', 'green', 'blue'])
        
click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'Meta', 'Shift']] = [])
    Description: Click an element.
    Examples:
        click('51')

        click('b22', button='right')

        click('48', button='middle', modifiers=['Shift'])

dblclick(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'Meta', 'Shift']] = [])
    Description: Double click an element.
    Examples:
        dblclick('12')

        dblclick('ca42', button='right')

        dblclick('178', button='middle', modifiers=['Shift'])

hover(bid: str)
    Description: Hover over an element.
    Examples:
        hover('b8')

press(bid: str, key_comb: str)
    Description: Focus the matching element and press a combination of keys. It accepts the logical key names that are emitted in the keyboardEvent.key property of the keyboard events: Backquote, Minus, Equal, Backslash, Backspace, Tab, Delete, Escape, ArrowDown, End, Enter, Home, Insert, PageDow
n, PageUp, ArrowRight, ArrowUp, F1 - F12, Digit0 - Digit9, KeyA - KeyZ, etc. You can alternatively specify a single character you'd like to produce such as "a" or "#". Following modification shortcuts are also supported: Shift, Control, Alt, Meta.
    Examples:
        press('88', 'Backspace')

        press('a26', 'Control+a')

        press('a61', 'Meta+Shift+t')

focus(bid: str)
    Description: Focus the matching element.
    Examples:
        focus('b455')

clear(bid: str)
    Description: Clear the input field.
    Examples:
        clear('996')

drag_and_drop(from_bid: str, to_bid: str)
    Description: Perform a drag & drop. Hover the element that will be dragged. Press left mouse button. Move mouse to the element that will receive the drop. Release left mouse button.
    Examples:
        drag_and_drop('56', '498')

upload_file(bid: str, file: str | list[str])
    Description: Click an element and wait for a "filechooser" event, then select one or multiple input files for upload. Relative file paths are resolved relative to the current working directory. An empty list clears the selected files.
    Examples:
        upload_file('572', 'my_receipt.pdf')

        upload_file('63', ['/home/bob/Documents/image.jpg', '/home/bob/Documents/file.zip'])

Only a single action can be provided at once. Example:
fill('a12', 'example with "quotes"')
Multiple actions are meant to be executed sequentially without any feedback from the page.
Don't execute multiple actions at once if you need feedback from the page. 


# Abstract Example

Here is an abstract version of the answer with description of the content of
each tag. Make sure you follow this structure, but replace the content with your
answer:

<think>
Think step by step. If you need to make calculations such as coordinates, write them here. Describe the effect
that your previous action had on the current content of the page.
</think>

<action>
One single action to be executed. You can only use one action at a time.
</action>

# Concrete Example

Here is a concrete example of how to format your answer.
Make sure to follow the template with proper tags:

<think>
My memory says that I filled the first name and last name, but I can't see any
content in the form. I need to explore different ways to fill the form. Perhaps
the form is not visible yet or some fields are disabled. I need to replan. 
</think>

<action>
fill('a12', 'example with "quotes"')
</action>
'''
    return action_prompt, action_system_prompt


def wa_format_action_prompt_for_BC(goal, history, observation):
    action_system_prompt = f'''You are an agent trying to solve a web task based on the content of the page and user instructions. 
You can interact with the page and explore. 
Each time you submit an action it will be sent to the browser and you will receive a new page.
    '''

    action_prompt = f'''
# Instructions                                                                                                                                                                                                                                                                                           
Review the current state of the page and all other information to find the best possible next action to accomplish your goal. 
Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

## Goal:
{goal}

## History of Interaction:
{history}

# Observation of current step:
{observation}
'''
    return action_prompt, action_system_prompt


