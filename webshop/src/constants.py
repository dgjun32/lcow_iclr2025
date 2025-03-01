
FEW_SHOT_EXAMPLES_REPR = '''
Webshop 
Instruction:
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[ Search ]  

Action: search[3 ounce bright citrus deodorant sensitive skin]

Observation:
Progress: I searched the keyword '3 ounce bright citrus deodorant sensitive skin' to see the relvant items, And now I am looking at the item list.
Reasoning: Based on the Progress and current observation, the next step is to choose an item satisfying the specification.
I can focus on:
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 

Action: click[B078GWRC1J]

Observation:
Progress: I searched and and clicked the item seems to be most relevant to the goal specification. I am looking at the option list. 
Reasoning: As the goal requires 3-ounce bottle, I can focus on the size option.
I can focus on:
size 
[ travel set (4-pack) ]
[ 3 ounce (pack of 1) ]
[ 3-ounce (2-pack) ]

Action: click[3-ounce (pack of 1)]

Observation:
Progress: I searched and and clicked the item id. Among the option list, and I clicked size option.
Reasoning: According to the progress, I have to focus on the scent option as a next step.
I can focus on:
scent 
[ assorted scents ]
[ bright citrus ]
[ calming lavender ]
[ ginger fresh ]
[ simply non-scents ]

Action: click[bright citrus]

Observation:
Progress: Based on **observation** and **previous actions**, I clicked size option and scent option.
Reasoning: As there is no more options to select and I met all requirements specified in the goal, next step is to buy the item.
I can focus on:
[ Buy Now ]

Action: click[Buy Now]
'''


FEW_SHOT_EXAMPLES_REACT = """Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus. 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

Action: click[Buy Now]
"""

FEW_SHOT_EXAMPLES = '''
Webshop 
Instruction:
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[ Search ]  

Action: search[3 ounce bright citrus deodorant sensitive skin]

Observation: 
[ Back to Search ] 
Page 1 (Total results: 50) 
[ Next > ] 
[ B078GWRC1J ] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[ B078GTKVXY ] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[ B08KBVJ4XN ] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: click[B078GWRC1J]

Observation: 
[ Back to Search ] 
[ < Prev ] 
scent 
[ assorted scents ]
[ bright citrus ]
[ calming lavender ]
[ ginger fresh ]
[ simply non-scents ]
size 
[ travel set (4-pack) ]
[ 3 ounce (pack of 1) ]
[ 3-ounce (2-pack) ]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[ Description ] 
[ Features ] 
[ Reviews ] 
[ Buy Now ]  

Action: click[bright citrus]

Observation:
[ Back to Search ] 
[ < Prev ] 
scent 
[ assorted scents ]
[ bright citrus ]
[ calming lavender ]
[ ginger fresh ]
[ simply non-scents ]
size 
[ travel set (4-pack) ]
[ 3 ounce (pack of 1) ]
[ 3-ounce (2-pack) ]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[ Description ] 
[ Features ] 
[ Reviews ] 
[ Buy Now ] 

Action: click[3 ounce (pack of 1)]

Observation:
[ Back to Search ] 
[ < Prev ] 
scent 
[ assorted scents ]
[ bright citrus ]
[ calming lavender ]
[ ginger fresh ]
[ simply non-scents ]
size 
[ travel set (4-pack) ]
[ 3 ounce (pack of 1) ]
[ 3-ounce (2-pack) ]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[ Description ] 
[ Features ] 
[ Reviews ] 
[ Buy Now ]  

Action: click[Buy Now]
'''



FEW_SHOT_EXAMPLES_3 = '''
You are an autonomous intelligent agent tasked with navigating a web browser. 
You will be given web-shopping task. These tasks will be accomplished through the use of specific actions you can issue.

Here's the information you'll have:
The user's goal: This is the task you are trying to complete.
The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.

Here's the action you can perform:
click[element]
search[element]

You can click elements surrounded by []. And you can search item by 'search[value to type]'

Here are a few examples:

OBSERVATION:
[ Search ] 

GOAL: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
PREVIOUS_ACTIONS: [None]
NEXT ACTION: search[3 ounce bright citrus deodorant sensitive skin]

OBSERVATION: 
[ Back to Search ] 
Page 1 (Total results: 50) 
[ Next > ] 
[ B078GWRC1J ] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[ B078GTKVXY ] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[ B08KBVJ4XN ] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

GOAL: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
PREVIOUS_ACTIONS: [search[3 ounce bright citrus deodorant sensitive skin]]
NEXT_ACTION: click[B078GWRC1J]

OBSERVATION:
[ Back to Search ] 
[ < Prev ] 
scent 
[ assorted scents ]
[ bright citrus ]
[ calming lavender ]
[ ginger fresh ]
[ simply non-scents ]
size 
[ travel set (4-pack) ]
[ 3 ounce (pack of 1) ]
[ 3-ounce (2-pack) ]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[ Description ] 
[ Features ] 
[ Reviews ] 
[ Buy Now ]  

GOAL: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
PREVIOUS_ACTIONS: [search[3 ounce bright citrus deodorant sensitive skin], click[B078GWRC1J]]
NEXT_ACTION: click[bright citrus]

OBSERVATION:
[ Back to Search ] 
[ < Prev ] 
scent 
[ assorted scents ]
[ bright citrus ]
[ calming lavender ]
[ ginger fresh ]
[ simply non-scents ]
size 
[ travel set (4-pack) ]
[ 3 ounce (pack of 1) ]
[ 3-ounce (2-pack) ]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[ Description ] 
[ Features ] 
[ Reviews ] 
[ Buy Now ] 

GOAL: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
PREVIOUS_ACTIONS: [search[3 ounce bright citrus deodorant sensitive skin], click[B078GWRC1J], click[bright citrus]]
NEXT_ACTION: click[3 ounce (pack of 1)]

OBSERVATION:
[ Back to Search ] 
[ < Prev ] 
scent 
[ assorted scents ]
[ bright citrus ]
[ calming lavender ]
[ ginger fresh ]
[ simply non-scents ]
size 
[ travel set (4-pack) ]
[ 3 ounce (pack of 1) ]
[ 3-ounce (2-pack) ]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[ Description ] 
[ Features ] 
[ Reviews ] 
[ Buy Now ] 

GOAL: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
PREVIOUS_ACTIONS: [search[3 ounce bright citrus deodorant sensitive skin], click[B078GWRC1J], click[bright citrus], click[3 ounce (pack of 1)]]
NEXT_ACTION: click[Buy Now] 

Here is the task:

'''

FEW_SHOT_EXAMPLES_3_REPR = '''
You are an autonomous intelligent agent tasked with navigating a web browser. 
You will be given web-shopping task. These tasks will be accomplished through the use of specific actions you can issue.
Here's the information you'll have:
The user's goal: This is the task you are trying to complete.
The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
Here's the action you can perform:
click[element]
search[element]
You can click elements surrounded by []. And you can search item by 'search[value to type]'
Here are a few examples:


GOAL: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars

SUBGOAL: search the 3 ounce bottle of bright citrus deodrant, cheaper than 50 dollars.

OBSERVATION:
[ Search ] 

NEXT ACTION: search[3 ounce bright citrus deodorant sensitive skin]


GOAL: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars

SUBGOAL: select the item which is a deodrant for sensitive skin with citrus scent.

OBSERVATION: 
[ Back to Search ] 
[ Next > ] 
[ B078GWRC1J ] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[ B078GTKVXY ] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 

NEXT_ACTION: click[B078GWRC1J]


GOAL: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars

OBSERVATION:
scent 
[ assorted scents ]
[ bright citrus ]
[ calming lavender ]
[ ginger fresh ]
[ simply non-scents ] 

SUBGOAL: select the scent option as a bright citrus

NEXT_ACTION: click[bright citrus]


GOAL: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars

OBSERVATION:
size 
[ travel set (4-pack) ]
[ 3 ounce (pack of 1) ]
[ 3-ounce (2-pack) ]

SUBGOAL: select a size option as a 3 ounce (pack of 1)

NEXT_ACTION: click[3 ounce (pack of 1)]


GOAL: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars

OBSERVATION:
[ Back to Search ] 
[ < Prev ]  
Price: $10.99 
Rating: N.A. 
[ Description ] 
[ Features ] 
[ Reviews ] 
[ Buy Now ] 

SUBGOAL: As all requirements are satisfied, It's time to buy the item.

NEXT_ACTION: click[Buy Now] 


Here is the task:
'''