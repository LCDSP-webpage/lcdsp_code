from langchain.prompts import HumanMessagePromptTemplate



prompt_generate_instrution_ball_control_en = HumanMessagePromptTemplate.from_template(
"""
You are an assistant for a football game, and you need to describe executable commands in the game using natural language. Below are explanations and examples of the commands:

Relative Distance
Relative Distance include: Compact, Normal, Loose

Position
Position incluede: Retreat, Balanced, Press

Ball Control
Ball Handling include: Carry Ball, Pass Ball

Commands
Commands are combinations of relative distance, position, and ball control, for example:

Example 1
Command:"
Relative Distance: Compact
Position: Retreat
Ball Control: Carry Ball
"
Natural Language: 
Chose a condensed formation with a defensive stance, seeking to control the ball.
Retain a close-knit formation in the backline and keeping the ball at your feet.
Move with the ball under control and ensure a dense setup in the defensive area.

Example 2
Command:"
Relative Distance: Loose
Position: Balanced
Ball Control: Pass
"
Natural Language: 
With a spread formation and even positioning, then control the ball through regular passing.
Ensure equilibrium in the front and back while maintaining stretched separation, seeking opportunities through frequent passing. 
Deliver the ball to ally while retain front-to-back symmetry with wider gaps.

Example 3
Command:"
Relative Distance: Normal
Position: Press
Ball Control: Carry Ball
"
Natural Language: 
While holding the ball, maintain standard spacing and press forward.
Maintain regular intervals and advance pressure upfield while in possession.
Preserve regular spacing and apply forward pressure while in possession.
With the ball, preserve usual distances and push forward aggressively.

Example 4
Command:"
Relative Distance: 
Position: Press
Ball Control: Carry Ball
"
Natural Language: 
One player dribbles while the entire team presses forward.
Advances with the ball as the team pushes upfield.
Ball at feet, the team moves upfield.
Pressing forward, the team keeps the ball at their feet.

Example 5
Command:"
Relative Distance: Normal
Position: 
Ball Control: Pass
"
Natural Language: 
Keeping standard spacing, passes the ball frequently.
The ball is passed around, maintaining regular spacing.
The team keeps usual formation and passes the ball.

Example 6
Command:"
Relative Distance: Compact
Position: Retreat
Ball Control: 
"
Natural Language: 
The team maintains close spacing and plays defense in the backcourt.
Close spacing is kept as the team defends in the backcourt.
Players move to the backcourt with reduced spacing.
The spacing between players shrinks as they move to the backcourt.
To the backcourt, the players move with smaller distances between them.

Example 7
Command:"
Relative Distance: 
Position: Balanced
Ball Control: Pass
"
Natural Language: 
The team should be balanced front and back for effective passing.
With balanced front and back positioning, pass the ball.
To pass the ball, ensure the team is evenly positioned front and back.

Example 8
Command:"
Relative Distance: Loose
Position: Press
Ball Control: 
Natural Language: 
To achieve a more dispersed formation, the team should spread out.
Spread your positions by widening the gap.
To disperse positions, increase the distance.
Spread out the team to occupy a larger area.

Example 9
Command:"
Relative Distance: 
Position: 
Ball Control: Carry Ball
"
Natural Language: 
The ball is dribbled by one player.
A player dribbles solo.
The ball should remain at your feet.
Under your feet, control the ball.

For each command, generate ten diverse natural language descriptions, using line breaks to separate them, and start each with a numerical ID.
For example:
1.xxxxx
2.xxxxx 
...
10.xxxxx
Note: The generated sentences must vary in length, punctuation, and structure. They should not follow the same format.

For example:
1.While holding the ball, maintain standard spacing and press forward.
2.Maintain regular intervals and advance pressure upfield while in possession.

Command:
{instruction}
Natural Language:
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

### Team Formations ###
The selectable formations are limited to the following six. Do not generate other formations.
Relative Distance: Compact, Normal, Loose
Position: Retreat, Balanced, Press

### Shoot Positions ###
The selectable shooting positions are limited to the following two. Do not generate other positions.
Shoot Positions: Penalty Area, Goal Area

### Actions ###
The selectable actions are limited to the following three. Do not generate other actions.
Ball Control Actions: Carry Ball, Pass Ball
Shooting Actions: Shoot
Movement Actions: Run, Dribble

### Requirements ###
You need to make your best effort infer the player's intended behavior based on their instructions and map it to the game parameters.
In-game instructions can be categorized into Ball Control instructions and Shooting instructions, these instructions are simple commands that apply to a team.
Ball Control instructions are a combination of relative distance, position, and ball control action.
Shooting instructions are a combination of movement action, shooting position, and shooting action.
Your response should include the reasoning process and the result. The result should be in JSON format, with keys as game parameters and values as the parameter values.
Please provide the response in pure JSON format without any additional characters or markdown.  
If the player's description is unrelated to football or cannot be infered into the above two types of instructions, the result should return an empty dictionary.

### Example 1 ###
instruction: "With close spacing, carry the ball and keep the team evenly poised."
answer:
{{
    "reason": "This is a ball control instruction that requires determining the relative distance, position, and ball control action. The player mentioned with close spacing, so the relative distance is compact. The player mentioned keep the team evenly, so the position action is Balanced. The player mentioned carry the ball so the ball control action is carry ball.",
    "result": {{
        "Relative Distance": ["Compact"],
        "Position": ["Balanced"],
        "Ball Control Action": ["Carry Ball"]
    }}
}}

### Example 2 ###
instruction: "Push upfield, ensuring swift passing to maintain pressure."
answer:
{{
    "reason": "This is a ball control instruction that requires determining the relative distance, position, and ball control action. The player did not mention about relative distance, so the relative distance can be either compact, normal, loose. The player mentioned the center of the midfield, so the horizontal position is midfield, and the vertical position is center.",
    "result": {{
        "Relative Distance": ["Compact",  "Normal", "Loose"],
        "Position": ["Press"],
        "Ball Control Action": ["Pass Ball"]
    }}
}}

### Example 3 ###
instruction: "Dribble with the ball, maintaining a loose formation."
answer:
{{
    "reason": "This is a ball control instruction that requires determining the relative distance, position, and ball control action. The player mentioned maintaining a loose formation, so the relative distance is loose. The player did not mention about position, so either retreat, balanced, press are selectable. The player mentioned dribble with the ball, so the ball control action is carry ball.",
    "result": {{
        "Relative Distance": ["Loose"],
        "Position": ["Retreat", "Balanced", "Press"],
        "Ball Control Action": ["Carry Ball"]
    }}
}}

### Example 4 ###
instruction: "Maintain typical spacing as you fall back."
answer:
{{
    "reason": "This is a ball control instruction that requires determining the relative distance, position, and ball control action. The player mentioned maintain typical spacing, so the relative distance is standrad. The player mentioned fall back, so the position is retreat. The player did not mention about ball control action, so either carry ball, pass ball are selectable.",
    "result": {{
        "Relative Distance": ["Normal"],
        "Position": ["Retreat"],
        "Ball Control Action": ["Carry Ball", "Pass Ball"]
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
instruction: "Sprint at full speed and attempt to shot"
answer:
{{
    "reason": "This is a shooting instruction that requires determining the shooting position, movement action, and shooting action. The player mentioned sprinting at full speed, so the movement action is run. The player mentioned attempting to do a point-blank shot, so we infer the shoot area is close and might be the goal area. The shooting action is shoot.",
    "result": {{
        "Shooting Position": ["Penalty Area", "Goal Area"],
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
instruction: "Make sure the ball is constantly being passed around."
answer:
{{
    "reason": "This is a ball control instruction that requires determining the relative distance, position, and ball control action. The player did not specify a specific relative distance, so the relative distance can be either compact, normal, loose. Also the player did not specify a specific position so the position can be retreat", balanced, press. The player mentioned that ball is constantly being passed, so the ball control action is pass ball.",
    "result": {{
        "Relative Distance": ["Compact",  "Normal", "Loose"],
        "Position": ["Retreat", "Balanced", "Press"],
        "Ball Control Action": ["Pass Ball"]
    }}
}}

instruction:
{instruction}
answer:
"""
)