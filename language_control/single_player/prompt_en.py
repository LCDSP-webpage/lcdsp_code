from langchain.prompts import HumanMessagePromptTemplate



prompt_generate_instrution_navigate_en = HumanMessagePromptTemplate.from_template(
"""
ou are an assistant for a football game, and you need to describe executable commands in the game using natural language. Below are explanations and examples of the commands:

Positions
Divided by horizontal coordinates: Front, Midfield, Back
Divided by vertical coordinates: Left, Center, Right

Actions
Actions include: Run, Dribble

Commands
Commands are combinations of positions and actions, for example:
Run/Dribble to the center/left/right of the front/midfield/back
To the center/left/right of the front/midfield/back
To the front/midfield/back
To the center/left/right
Run/Dribble to the front/midfield/back
Run/Dribble to the center/left/right
Run/Dribble

Example 1
Command: "Run, Front, ''"
Natural language: Go to the front to prepare for a point
Go to the front to support
Increase the number of players in the front
The front is lacking players, go support quickly
Push forward quickly
Charge ahead

Example 2
Command: "Run, Back, ''"
Natural language: Hurry back to defend.
Chase back
Quickly block the area

Example 3
Command: "Dribble, Midfield, Left"
Natural language: Strengthen control of the midfield space, move to the left while avoiding being tackled.
Control the midfield, dribble to the left to attack

Example 8
Command: "'', Back, Right"
Natural language: Position yourself to the right back

Example 5
Command: "'', '', Right"
Natural language: The opponent's right side is weak
Attack on the right
Look for opportunities on the right
Break through on the right

Example 6
Command: "Dribble, '', ''"
Natural language: Don't get tackled
Strengthen control
Play more precisely

Example 7
Command: "Run, '', ''"
Natural language: Sprint
Wear down the opponent, sprint

For each command, generate ten diverse natural language descriptions, separated by line breaks, starting with a numerical ID.
For example:
1.xxxxx
2.xxxxx 
...
10.xxxxx

Note: The generated sentences must have different lengths, different punctuation, and different word orders.
They should not be in the same format.
For example:
1.Sprint
2.Strengthen control
3.Strengthen control of the midfield space, move to the left while avoiding being tackled.
4.Control the midfield, left side attack

Command:
{instruction}
Natural language:
"""
)


prompt_generate_instrution_shot_en = HumanMessagePromptTemplate.from_template(
"""
You are an assistant for a football game, and you need to describe executable commands in the game using natural language. Below are explanations and examples of the commands:

Positions
Positions include: Penalty Area, Goal Area

Actions
Movement actions include: Run, Dribble
Shooting actions include: Shoot

Aliases
The Penalty Area can also be called the "18-yard Box" or "Penalty Box" 
The Goal Area can also be called the "6-yard Box" or "Goal Box"
Shooting can also be called "Take a shot," "Shoot" "Fire" "Hit"etc.

Commands
Commands are combinations of movement, position, and shooting, for example:
Sprint to the Penalty Area and shoot
Dribble to the Goal Area and fire
Take a shot
Sprint and shoot
Hit from the Goal Penalty Area

Example 1
Command: "Run, Penalty Area, Shoot"
Natural language: High-speed powerful long shot
Sprint and counterattack, take a distant shot quickly
The opponent's goalkeeper is out, sprint and take a long-range shot
Sprint over and fire in the 18-yard Box

Example 2
Command: "Dribble, Goal Area, Shoot"
Natural language: Control the ball well, get as close to the goal as possible before shooting
Dribble past the opponent first, then shoot within the 6-yard Box

Example 3
Command: "Run, '', Shoot"
Natural language: Move to a suitable position and shoot
Run to the shooting position, shoot quickly

Example 4
Command: "'', Goal Area, Shoot"
Natural language: The opponent's goalkeeper is very focused, long shots are not suitable, only point-blank shot has a chance
There is no opponent in the 6-yard Box, tap-in the ball

For each command, generate ten diverse natural language descriptions, separated by line breaks, starting with a numerical ID.
For example:
1.xxxxx
2.xxxxx 
...
10.xxxxx

Note: The generated sentences must have different lengths, different punctuation, and different word orders.
Use the aliases for Large Penalty Area, Small Penalty Area, and shooting.
For example:
1.Long shot
2.Dribble past the opponent first, then shoot within the 6-yard Box

Command:
{instruction}
Natural language:
"""
)

prompt_natural_language_to_style_en = HumanMessagePromptTemplate.from_template(
"""
You are a helpful assistant for a football game, responsible for understanding players' natural language instructions and converting them into in-game parameters. Below is an explanation of the game parameters and their functions.

### Positions ###
The selectable positions are limited to the following eight. Do not generate other positions.
Horizontal positions: Front, Midfield, Back
Vertical positions: Left, Center, Right
Shooting positions: Penalty Area, Goal Area

### Actions ###
The selectable actions are limited to the following three. Do not generate other actions.
Movement actions: Run, Dribble
Shooting actions: Shoot

### NOTE ###
Dribble action can slow down a player's movement speed, but they make it easier to get past opponents and reduce the likelihood of losing the ball. When rapid movement is required in urgent situations, please use the RUN action.

### Requirements ###
You need to make your best effort infer the player's intended behavior based on their instructions and map it to the game parameters.
In-game instructions can be categorized into movement instructions and shooting instructions, these instructions are simple commands that apply to a single player.
Movement instructions are a combination of vertical position, horizontal position, and movement action.
Shooting instructions are a combination of movement action, shooting position, and shooting action.
Your response should include the reasoning process and the result. The result should be in JSON format, with keys as game parameters and values as the parameter values.
Please provide the response in pure JSON format without any additional characters or markdown.  
If the player's description is unrelated to football or cannot be infered into the above two types of instructions, the result should return an empty dictionary.

### Example 1 ###
instruction: "Quickly sprint to the left side of the front area to support a teammate."
answer:
{{
    "reason": "This is a movement instruction that requires determining the horizontal position, vertical position, and movement action. The player mentioned quickly sprinting, so the movement action is run. The player mentioned the left side of the front area, so the horizontal position is front, and the vertical position is left.",
    "result": {{
        "Horizontal Position": ["Front"],
        "Vertical Position": ["Left"],
        "Movement Action": ["Run"]
    }}
}}

### Example 2 ###
instruction: "Dribble in the center of the midfield to confuse the opponent and create an illusion of attack."
answer:
{{
    "reason": "This is a movement instruction that requires determining the horizontal position, vertical position, and movement action. The player mentioned dribbling to confuse the opponent, so the movement action is dribble. The player mentioned the center of the midfield, so the horizontal position is midfield, and the vertical position is center.",
    "result": {{
        "Horizontal Position": ["Midfield"],
        "Vertical Position": ["Center"],
        "Movement Action": ["Dribble"]
    }}
}}

### Example 3 ###
instruction: "Strengthen the right side of the back area."
answer:
{{
    "reason": "This is a movement instruction that requires determining the horizontal position, vertical position, and movement action. The player only mentioned adjusting without specifying a specific movement method, so run and dribble are all selectable. The player mentioned the right side of the back area, so the horizontal position is back, and the vertical position is right.",
    "result": {{
        "Horizontal Position": ["Back"],
        "Vertical Position": ["Right"],
        "Movement Action": ["Run", "Dribble"]
    }}
}}

### Example 4 ###
instruction: "The right flank has great attacking potential."
answer:
{{
    "reason": "This is a movement instruction that requires determining the horizontal position, vertical position, and movement action. The player did not specify a specific movement method, so run and dribble are all selectable. The player mentioned the right flank without specifying a horizontal position, so the horizontal position can be front, midfield, or back, and the vertical position is right.",
    "result": {{
        "Horizontal Position": ["Front", "Midfield", "Back"],
        "Vertical Position": ["Left", "Center", "Right"],
        "Movement Action": ["Run", "Dribble"]
    }}
}}

### Example 5 ###
instruction: "Dribble through and attempt a shot near the penalty area."
answer:
{{
    "reason": "This is a shooting instruction that requires determining the shooting position, movement action, and shooting action. The player mentioned dribbling through, so the movement action is dribble. The player mentioned near the penalty area, so the shooting position is penalty area. The shooting action is shoot.",
    "result": {{
        "Shooting Position": ["Penalty Area"],
        "Movement Action": ["Dribble"],
        "Shooting Action": ["Shoot"]
    }}
}}

### Example 6 ###
instruction: "Take a shot."
answer:
{{
    "reason": "This is a shooting instruction that requires determining the shooting position, movement action, and shooting action. The player did not specify a specific shooting area, so both penalty area and goal area are selectable. The player did not specify a specific movement method, so run, and dribble are all selectable. The shooting action is shoot.",
    "result": {{
        "Shooting Position": ["Penalty Area", "Goal Area"],
        "Movement Action": ["Run", "Dribble"],
        "Shooting Action": ["Shoot"]
    }}
}}

### Example 7 ###
instruction: "Sprint at full speed and attempt to point-blank shot"
answer:
{{
    "reason": "This is a shooting instruction that requires determining the shooting position, movement action, and shooting action. The player mentioned sprinting at full speed, so the movement action is run. The player mentioned attempting to do a point-blank shot, so we infer the shoot area is close and might be the goal area. The shooting action is shoot.",
    "result": {{
        "Shooting Position": ["Goal Area"],
        "Movement Action": ["Run"],
        "Shooting Action": ["Shoot"]
    }}
}}

### Example 8 ###
instruction: "Fire."
answer:
{{
    "reason": "This is a shooting instruction that requires determining the shooting position, movement action, and shooting action. The player did not specify a specific shooting area or movement method, so both penalty area and goal area are selectable, and the movement action can be run, or dribble. The shooting action is shoot.",
    "result": {{
        "Shooting Position": ["Penalty Area", "Goal Area"],
        "Movement Action": ["Run", "Dribble"],
        "Shooting Action": ["Shoot"]
    }}
}}

### Example 9 ###
instruction: "Control the ball carefully."
answer:
{{
    "reason": " The player wants to control the ball. The dribble movement action in the in-game parameter aligns with this intent, so this is a movement instruction. But the player did not specify a position, so all position parameter are available.",
    "result": {{
        "Horizontal Position": ["Front", "Midfield", "Back"],
        "Vertical Position": ["Right"],
        "Movement Action": ["Run", "Dribble"]
    }}
}}

### Example 10 ###
instruction: "The opponent's sustained attack is penetrating our backline, making it crucial to pull back our players."
answer:
{{
    "reason": "This is a movement instruction that requires determining the horizontal position, vertical position, and movement action. The player said backline is now being attacked, mean we should move to the back area to strength defence, so the horizontal position is Back. The player did not specify a position but only provided the word backline, neither can we infer it from the instruction, so the vertical position can be either left, center, right. The player said it is crucial to pull back, the movement action should be Run.",
    "result": {{
        "Horizontal Position": ["Back"],
        "Vertical Position": ["Left", "Center", "Right"],
        "Movement Action": ["Run"]
    }}
}}

instruction:
{instruction}
answer:
"""
)