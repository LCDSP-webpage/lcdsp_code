from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# long
prompt_generate_instrution_long_en = HumanMessagePromptTemplate.from_template(
"""
You are a football coach and need to communicate your intended tactical actions to the players. 
The tactics should contain as much as these elements: Desire to Attack and Defend, tendency of individual ball control, tendency of regaining possession, tendency of passing, relative distance between players, overall positioning of the team on the field, tendency of shoot far or shot close, dribbling style

Example1:
Intended Tactical: Active Offense
Natural Language:
Push our defensive line higher up the field to compress the space and support our attacking play, encourage short passes to maintain possession and create opportunities.
Overlap with the wingers to provide width and stretch their defense, constantly move to create passing lanes and support both the attack and defense.
Encourage players to take on defenders with dribbling to create advantages, look for opportunities to shoot keeping their goalkeeper busy.

Example2:
Intended Tactical: Relentless Attack, Ignoring Defense
Natural Language:
Push all players forward, including fullbacks, regardless of the threat of counterattacks, focus on high possession with short passes to keep the pressure on their backline.
Constantly advance and support the attack, leaving minimal players behind, despite the potential for counterattacks, to keep the team compact and pressing to regain possession immediately.
Maintain a very high defensive line, even at the risk of counterattacks, to keep the team compact and focused on attacking, prioritize shooting at every opportunity.

You should list 10 different natural language expressions for tactics. For each expression, using line breaks to separate them, and start each with a numerical ID.
The generated expression must vary in length, punctuation, and structure, but should keep the length same as the provide examples.Do not copy the examples, generate new ones.
Example:
1. xxxx
2. xxxx
3. xxxx
......
30. xxxx

Intended Tactical:
{instruction}
Natural Language:
"""
)

# normal
prompt_generate_instrution_en = HumanMessagePromptTemplate.from_template(
"""
You are a football coach and need to communicate your intended tactical actions to the players. 
The tactics can contain some of these elements: Desire to Attack and Defend, tendency of individual ball control, tendency of regaining possession, tendency of passing, relative distance between players, back or force positioning of the team on the field, tendency of shot far close, dribbling style.
The tactics should be clear and concise and consist of only one sentence.

Example1:
Intended Tactical: Slightly Positive attack
Natural Language:
Keep a steady pace in attack, make calculated forward movements to create chances without leaving gaps at the back.
Control the tempo, advance methodically, and create scoring opportunities through precise passing and movement.
Keep a disciplined formation, move forward with intent, and focus on creating high-quality chances without overextending.
Advance with a slight edge, focus on maintaining possession, and look for high-percentage scoring chances without leaving ourselves exposed.
Adopt a forward-thinking mindset, maintain a compact formation, and create opportunities through patient build-up play.

Example2:
Intended Tactical: All-out attack, sacrificing defense
Natural Language:
Push everyone forward, sacrifice defensive stability, and focus entirely on scoring goals.
Push the team upfield, accept defensive gaps, and focus on relentless attacking play.
Focus on non-stop attacking, sacrifice defensive structure, and keep the pressure on the opponent's goal.

Example3:
Intended Tactical: Balanced play
Natural Language: 
Focus on a balanced game plan, blending solid defense with strategic offensive plays to ensure both stability and creativity.
Keep the team evenly spread, balancing forward momentum with defensive responsibilities.
Focus on both creating chances and regaining possession, maintaining a balanced approach.

Example4:
Intended Tactical: Counter attack
Natural Language:
Stay compact and deep, absorbing pressure and using quick counterattacks to exploit the opposition's weaknesses.
Prioritize quick, direct passes after winning the ball, launching fast counter-attacks to catch the defense unprepared.
Defend deep in a low block, staying organized and ready to rapid forward movements after regaining possession.

Example5:
Intended Tactical: Park the bus, aim for draw
Natural Language:
Maintain a very deep defensive line, focusing solely on blocking shots and preventing goals to secure a draw.
Focus on a highly defensive strategy, with all players committed to protecting our goal at all costs.
Keep a tight, deep defensive block, with minimal forward movement to protect our goal.

Example6:
Intended Tactical: High pressing, Tiki-Taka
Natural Language:
Apply intense high pressing to regain possession quickly, then use short, quick passes to control the game.
Press aggressively up the field, focusing on quick ball recovery and maintaining possession with short, precise passes. 
Focus on relentless high pressing to win the ball back early, then use Tiki-Taka to control the tempo.

You should list 30 different natural language expressions for tactics. For each expression, using line breaks to separate them, and start each with a numerical ID.
The generated expression must vary in length, punctuation, and structure, but should keep the length similar or longer as the provide examples. Do not copy the examples, generate new ones.
Example:
1. xxxx
2. xxxx
3. xxxx
......
30. xxxx

Intended Tactical:
{instruction}
Natural Language:
"""
)

prompt_positive_generate_instrution_en = HumanMessagePromptTemplate.from_template(
"""
You are a football coach and need to communicate your intended tactical actions to the players. 
The tactics can contain some of these elements: Desire to Attack and Defend, tendency of individual ball control, tendency of regaining possession, tendency of passing, relative distance between players, back or force positioning of the team on the field, tendency of shot far close, dribbling style.
The tactics should be clear and concise and consist of only one sentence.

Example1:
Intended Tactical: Slightly positive attack
Natural Language:
Keep a steady pace in attack, make calculated forward movements to create chances without leaving gaps at the back.
Control the tempo, advance methodically, and create scoring opportunities through precise passing and movement.
Keep a disciplined formation, move forward with intent, and focus on creating high-quality chances without overextending.
Advance with a slight edge, focus on maintaining possession, and look for high-percentage scoring chances without leaving ourselves exposed.
Adopt a forward-thinking mindset, maintain a compact formation, and create opportunities through patient build-up play.

Example2:
Intended Tactical: All-out attack, sacrificing defense
Natural Language:
Push everyone forward, sacrifice defensive stability, and focus entirely on scoring goals.
Push the team upfield, accept defensive gaps, and focus on relentless attacking play.
Focus on non-stop attacking, sacrifice defensive structure, and keep the pressure on the opponent's goal.

Example3:
Intended Tactical: Counter attack
Natural Language:
Stay compact and deep, absorbing pressure and using quick counterattacks to exploit the opposition's weaknesses.
Prioritize quick, direct passes after winning the ball, launching fast counter-attacks to catch the defense unprepared.
Defend deep in a low block, staying organized and ready to rapid forward movements after regaining possession.

Example4:
Intended Tactical: Park the bus, aim for draw
Natural Language:
Maintain a very deep defensive line, focusing solely on blocking shots and preventing goals to secure a draw.
Focus on a highly defensive strategy, with all players committed to protecting our goal at all costs.
Keep a tight, deep defensive block, with minimal forward movement to protect our goal.

Example5:
Intended Tactical: High pressing, Tiki-Taka
Natural Language:
Apply intense high pressing to regain possession quickly, then use short, quick passes to control the game.
Press aggressively up the field, focusing on quick ball recovery and maintaining possession with short, precise passes. 
Focus on relentless high pressing to win the ball back early, then use Tiki-Taka to control the tempo.

You should list 30 different natural language expressions for tactics. For each expression, using line breaks to separate them, and start each with a numerical ID.
The generated expression must vary in length, punctuation, and structure, but should keep the length similar or longer as the provide examples. Do not copy the examples, generate new ones.
Example:
1. xxxx
2. xxxx
3. xxxx
......
30. xxxx

Intended Tactical:
{instruction}
Natural Language:
"""
)

# Example natural language expressions for different tactics:
# 1. This is a crucial moment; we must attack fiercely at all costs. Give it your all and seize every goal opportunity.
# 2. We must defend with everything we've got to ensure a draw. Stay focused and protect every ball!
# 3. We need to slightly increase our offensive efforts. Maintain pressure, create more chances, and aim for more shots.
# 4. We need to strengthen our attack. Keep a positive attitude, create more opportunities, and increase the number of shots.
# 5. We are now defending deeply, waiting for counter-attack opportunities. Maintain a tight defense and counter-attack quickly whenever there's a chance.


# Example3:
# Intended Tactical:
# Natural Language:

# Example4:
# Intended Tactical:
# Natural Language:

# Example5:
# Intended Tactical:
# Natural Language:

# Example6:
# Intended Tactical:
# Natural Language:


prompt_natural_language_to_style_en = HumanMessagePromptTemplate.from_template(
"""
In a football match, each team consists of 5 players: one goalkeeper and four outfield players. The match is played on a standard-sized football field according to standard football rules for a duration of 90 minutes.
The goalkeeper and outfield players are controlled by different strategies.
There are many styles for controlling outfield players, and these styles are effective simultaneously and interact with each other. Each style can be adjusted on a scale from 0 to 10.
Here are the explanations for each style:
Style_1: Emphasis on the importance of winning; the higher the value, the more the team desires to win. A value of 0 means they do not care about winning at all.
Style_2: Emphasis on the importance of scoring goals; the higher the value, the more the team desires to score. A value of 0 means they do not care about scoring at all.
Style_3: Emphasis on the importance of not conceding goals; the higher the value, the less the team wants to concede. A value of 0 means they do not care about conceding goals at all.
Style_4: Emphasis on the importance of individual ball control; a value of 0 means they do not intentionally hold the ball but decide based on the situation and other tendencies, while a value of 10 means they prefer to hold the ball individually.
Style_5: Emphasis on regaining possession; a value of 0 means they do not care about regaining or losing possession, while a value of 10 means they place great importance on regaining possession from the opponent and minimizing loss of possession.
Style_6: Emphasis on passing; a value of 0 means they do not pass intentionally but decide based on the situation on the field and other tendencies, while a value of 10 means they prefer to pass.
Style_7: Relative distance between players; a value of 0 means the team is compact with small relative distances, while a value of 10 means the team has larger relative distances and is more spread out.
Style_8: The overall positioning of the team on the field; a value of 0 means the team is closer to their own goal line, while a value of 10 means the team is closer to the opponent's goal line.
The following two styles cannot be adjusted on a scale from 0 to 10 but must be selected from specific options:
Style_9: Shooting distance; a value of -1 means there is no tendency regarding shooting distance, 0 means they prefer to shoot from close range, and 1 means they prefer to shoot from long range.
Style_10: Dribbling style; a value of -1 means there is no tendency regarding dribbling style, 0 means they prefer a moderate running speed while dribbling, making it moderately easy to be dispossessed, 1 means they prefer to dribble closely while moving slower and being harder to dispossess, and 2 means they prefer to sprint past opponents quickly while dribbling but are more easily dispossessed.
At this point, the coach will give certain instructions. Based on the understanding of football and the styles mentioned above, you should carefully reason step by step to translate the coach's instructions into corresponding style levels and provide the reasoning process.
Finally, provide a standard json format result where the keys are each style and value is the corresponding style parameter, do not add comment in the json file, the result formatted as follow:

```json
{{
  "Style_1": value,  
  "Style_2": value,  
  "Style_3": value,  
  "Style_4": value,  
  "Style_5": value,  
  "Style_6": value,  
  "Style_7": value,  
  "Style_8": value,  
  "Style_9": value,  
  "Style_10": value  
}}
```

## Example1:
instruction:Keep a disciplined formation, move forward with intent, and focus on creating high-quality chances without overextending.
answer:
To translate the coach's instructions into corresponding style levels, let's break down the instructions and reason step by step:

1. **Keep a disciplined formation**:
   - This suggests maintaining a structured and organized team shape, which implies a focus on not conceding goals and keeping the team compact.
   - **Style_3 (Not conceding goals)**: Slightly high importance, so a higher value around 6.
   - **Style_7 (Relative distance between players)**: To maintain a disciplined formation, the team should be more compact, so a lower value around 4.

2. **Move forward with intent**:
   - This indicates a purposeful approach to advancing on the field, suggesting a balance between attacking and defending.
   - **Style_8 (Overall positioning)**: A balanced but slightly forward approach, so a mid-range value around 6.

3. **Focus on creating high-quality chances without overextending**:
   - This emphasizes the importance of creating good scoring opportunities while being cautious not to leave the defense vulnerable.
   - **Style_2 (Scoring goals)**: Moderate emphasis, so a mid-range value around 6.
   - **Style_3 (Not conceding goals)**: Already set to 6 due to the disciplined formation.
   - **Style_6 (Passing)**: To create high-quality chances, passing is important, so a higher value around 7.

4. **Other styles**:
   - **Style_1 (Importance of winning)**:This instruction leans towards a slightly aggressive offense, so a moderate value around 6.
   - **Style_4 (Individual ball control)**: The instructions do not emphasize individual ball control, so a moderate value around 5.
   - **Style_5 (Regaining possession)**: The instructions do not emphasize regaining possession, so a moderate value around 5.
   - **Style_9 (Shooting distance)**: The instructions do not specify a preference for shooting distance, so a value of -1.
   - **Style_10 (Dribbling style)**: The instructions do not specify a dribbling style, so a value of -1.

Based on the reasoning above, the corresponding style levels are:

```json
{{
  "Style_1": 6,
  "Style_2": 6,
  "Style_3": 6,
  "Style_4": 5,
  "Style_5": 5,
  "Style_6": 7,
  "Style_7": 4,
  "Style_8": 6,
  "Style_9": -1,
  "Style_10": -1
}}
```

## Example2:
instruction: Push everyone forward, sacrifice defensive stability, and focus entirely on scoring goals.
answer:
To translate the coach's instructions into corresponding style levels, let's break down the instructions and reason step by step:

1. **Push everyone forward**:
   - This suggests an aggressive approach with the team positioned closer to the opponent's goal.
   - **Style_8 (Overall positioning)**: High importance, so a higher value around 9.
   - **Style_7 (Relative distance between players)**: To push everyone forward, the team all in oppontent's half, so a small value around 4.

2. **Sacrifice defensive stability**:
   - This indicates a lower emphasis on not conceding goals.
   - **Style_3 (Not conceding goals)**: Low importance, so a lower value around 3.

3. **Focus entirely on scoring goals**:
   - This suggests a very high emphasis on scoring goals.
   - **Style_2 (Scoring goals)**: Very high importance, so a maximum value of 10.

4. **Other styles**:
   - **Style_1 (Importance of winning)**: The instructions emphasize scoring goals, which is often tied to winning, so a high value around 8.
   - **Style_4 (Individual ball control)**: To score goals, individual skill and breakthrough are needed, so a slightly high value around 7.
   - **Style_5 (Regaining possession)**: To score goals, it is necessary to gain possession of the ball, so a higher value around 8.
   - **Style_6 (Passing)**: To maintain an aggressive forward approach, passing is important, so a higher value around 7.
   - **Style_9 (Shooting distance)**: The instructions do not specify a preference for shooting distance, so a value of -1.
   - **Style_10 (Dribbling style)**: The instructions do not specify a dribbling style, so a value of -1.

Based on the reasoning above, the corresponding style levels are:

```json
{{
  "Style_1": 8,
  "Style_2": 10,
  "Style_3": 3,
  "Style_4": 7,
  "Style_5": 8,
  "Style_6": 7,
  "Style_7": 4,
  "Style_8": 9,
  "Style_9": -1,
  "Style_10": -1
}}
```

## Example3:
instruction: Focus on a balanced game plan, blending solid defense with strategic offensive plays to ensure both stability and creativity.
answer:
To translate the coach's instructions into corresponding style levels, let's break down the instructions and reason step by step:

1. **Focus on a balanced game plan**:
   - This suggests an approach that balances both defense and offense.
   - **Style_1 (Importance of winning)**: Moderate importance, so a mid-range value around 5.

2. **Blending solid defense**:
   - This indicates a focus on not conceding goals.
   - **Style_3 (Not conceding goals)**: Balanced game plan, so a mid-range value around 5.

3. **Strategic offensive plays**:
   - This suggests a focus on scoring goals through well-thought-out strategies.
   - **Style_2 (Scoring goals)**: Balanced game plan, so a mid-range value around 5.
   - **Style_6 (Passing)**: To execute strategic offensive plays, passing is important, so a higher value around 7.

4. **Ensure both stability and creativity**:
   - This indicates a need for both defensive stability and creative offensive plays.
   - **Style_4 (Individual ball control)**: Creativity often involves individual ball control, so a moderate value around 5.
   - **Style_5 (Regaining possession)**: To maintain stability, regaining possession is important, so a moderate value around 6.
   - **Style_7 (Relative distance between players)**: A balanced game plan would require moderate relative distances, so a mid-range value around 5.
   - **Style_8 (Overall positioning)**: A balanced approach would mean a mid-range positioning on the field, so a value around 5.

5. **Other styles**:
   - **Style_9 (Shooting distance)**: The instructions do not specify a preference for shooting distance, so a value of -1.
   - **Style_10 (Dribbling style)**: The instructions do not specify a dribbling style, so a value of -1.

Based on the reasoning above, the corresponding style levels are:

```json
{{
  "Style_1": 5,
  "Style_2": 5,
  "Style_3": 5,
  "Style_4": 5,
  "Style_5": 6,
  "Style_6": 7,
  "Style_7": 5,
  "Style_8": 5,
  "Style_9": -1,
  "Style_10": -1
}}
```

## Example4:
instruction: Stay compact and deep, absorbing pressure and using quick counterattacks to exploit the opposition's weaknesses.
answer:
To translate the coach's instructions into corresponding style levels, let's break down the instructions and reason step by step:

1. **Stay compact and deep**:
   - This suggests maintaining a tight and organized defensive shape, with the team positioned closer to their own goal.
   - **Style_3 (Not conceding goals)**: High importance, so a higher value around 8.
   - **Style_7 (Relative distance between players)**: To stay compact, the team should have small relative distances, but also need to consider counterattacks, so a slightly lower value around 4.
   - **Style_8 (Overall positioning)**: To stay deep, the team should be closer to their own goal line, but also need to consider counterattacks, so a slightly lower a lower value around 4.

2. **Absorbing pressure**:
   - This indicates a defensive approach, focusing on withstanding the opponent's attacks.
   - **Style_3 (Not conceding goals)**: Already set to a high value (8) due to the emphasis on staying compact and deep.

3. **Using quick counterattacks to exploit the opposition's weaknesses**:
   - This suggests a strategy of quickly transitioning from defense to attack, focusing on speed and efficiency.
   - **Style_2 (Scoring goals)**: Moderate emphasis, so a mid-range value around 6.
   - **Style_5 (Regaining possession)**: To initiate quick counterattacks, regaining possession is important, so a higher value around 7.
   - **Style_6 (Passing)**: Quick counterattacks require swift passing, so a higher value around 7.
   - **Style_10 (Dribbling style)**: Quick counterattacks may involve sprinting past opponents, so a value of 2 (prefer to sprint past opponents quickly while dribbling but are more easily dispossessed).

4. **Other styles**:
   - **Style_1 (Importance of winning)**:  This instruction means to counterattack defensively, thus favoring a win, so a higher value around 7.
   - **Style_4 (Individual ball control)**: Counterattacks sometimes require a solo long-distance run., so a higher value around 7.
   - **Style_9 (Shooting distance)**: The instructions do not specify a preference for shooting distance, so a value of -1.

Based on the reasoning above, the corresponding style levels are:

```json
{{
  "Style_1": 7,
  "Style_2": 6,
  "Style_3": 8,
  "Style_4": 7,
  "Style_5": 7,
  "Style_6": 7,
  "Style_7": 4,
  "Style_8": 4,
  "Style_9": -1,
  "Style_10": 2
}}
```

## Example5:
instruction: Maintain a very deep defensive line, focusing solely on blocking shots and preventing goals to secure a draw.
answer:
To translate the coach's instructions into corresponding style levels, let's break down the instructions and reason step by step:

1. **Maintain a very deep defensive line**:
   - This suggests positioning the team very close to their own goal line.
   - **Style_8 (Overall positioning)**: Very high importance, so a very low value around 1.

2. **Focusing solely on blocking shots and preventing goals**:
   - This indicates a very high emphasis on not conceding goals.
   - **Style_3 (Not conceding goals)**: Very high importance, so a maximum value of 10.

3. **To secure a draw**:
   - This suggests that the team is not focused on winning or scoring goals, but rather on maintaining the current scoreline.
   - **Style_1 (Importance of winning)**: Low importance, so a low value around 2.
   - **Style_2 (Scoring goals)**: Low importance, so a low value around 2.

4. **Other styles**:
   - **Style_4 (Individual ball control)**: The instructions do not emphasize individual ball control, so a lower value around 3.
   - **Style_5 (Regaining possession)**: The instructions do not emphasize regaining possession aggressively, so a moderate value around 4.
   - **Style_6 (Passing)**: The instructions do not emphasize passing, so a moderate value around 4.
   - **Style_7 (Relative distance between players)**: To maintain a deep defensive line and block shots effectively, the team should be compact, so a lower value around 2.
   - **Style_9 (Shooting distance)**: The instructions do not specify a preference for shooting distance, so a value of -1.
   - **Style_10 (Dribbling style)**: The instructions do not specify a dribbling style, so a value of -1.

Based on the reasoning above, the corresponding style levels are:

```json
{{
  "Style_1": 2,
  "Style_2": 2,
  "Style_3": 10,
  "Style_4": 3,
  "Style_5": 4,
  "Style_6": 4,
  "Style_7": 2,
  "Style_8": 1,
  "Style_9": -1,
  "Style_10": -1
}}
```

## Example6:
instruction: Focus on relentless high pressing to win the ball back early, then use Tiki-Taka to control the tempo.
answer:
To translate the coach's instructions into corresponding style levels, let's break down the instructions and reason step by step:

1. **Relentless high pressing to win the ball back early**:
   - This suggests a very aggressive approach to regaining possession, with the team pressing high up the pitch.
   - **Style_5 (Regaining possession)**: High importance, so a higher value around 9.
   - **Style_8 (Overall positioning)**: To pressing, the team should be positioned closer to the opponent's goal line, so a higher value around 8.

2. **Use Tiki-Taka to control the tempo**:
   - Tiki-Taka is a style of play characterized by short passing and maintaining possession to control the game.
   - **Style_6 (Passing)**: Very high importance, so a maximum value of 10.
   - **Style_4 (Individual ball control)**: Tiki-Taka involves quick, short passes rather than individual dribbling, so a lower value around 2.
   - **Style_7 (Relative distance between players)**: Tiki-Taka requires players to be close to each other to facilitate short passes, so a lower value around 2.

3. **Other styles**:
   - **Style_1 (Importance of winning)**: The instructions do not explicitly emphasize winning at all costs, so a moderate value around 5.
   - **Style_2 (Scoring goals)**: The instructions focus more on controlling the game and regaining possession rather than purely on scoring, so a moderate value around 5.
   - **Style_3 (Not conceding goals)**: The instructions do not emphasize defense as much as pressing and controlling the game, so a moderate value around 5.
   - **Style_9 (Shooting distance)**: The instructions do not specify a preference for shooting distance, so a value of -1.
   - **Style_10 (Dribbling style)**: Although dribbling is slower, it reduces the likelihood of being intercepted by opponents, possession-based play requires precise ball control, so a value of 1.

Based on the reasoning above, the corresponding style levels are:

```json
{{
  "Style_1": 5,
  "Style_2": 5,
  "Style_3": 5,
  "Style_4": 2,
  "Style_5": 9,
  "Style_6": 10,
  "Style_7": 2,
  "Style_8": 8,
  "Style_9": -1,
  "Style_10": 1
}}
```

## Example7:
instruction: Defend deep in a low block, staying organized and ready to rapid forward movements after regaining possession.
answer: 
To translate the coach's instructions into corresponding style levels, let's break down the instructions and reason step by step:

1. **Defend deep in a low block**:
   - This suggests positioning the team very close to their own goal line and maintaining a tight, organized defensive shape.
   - **Style_8 (Overall positioning)**: Very high importance, but also need to consider ready to rapid forward, so a slightly low value around 4.
   - **Style_7 (Relative distance between players)**: To stay organized in a low block, the team should have small relative distances, but also need to consider ready to rapid forward, so a slightly low value around 4.
   - **Style_3 (Not conceding goals)**: High importance, so a higher value around 8.

2. **Staying organized**:
   - This indicates a focus on maintaining a structured and disciplined team shape.
   - **Style_3 (Not conceding goals)**: Already set to a high value (8) due to the emphasis on defending deep and staying organized.

3. **Ready to rapid forward movements after regaining possession**:
   - This suggests a strategy of quickly transitioning from defense to attack, focusing on speed and efficiency.
   - **Style_5 (Regaining possession)**: High importance, so a higher value around 8.
   - **Style_6 (Passing)**: Quick transitions often require swift passing, so a higher value around 7.
   - **Style_2 (Scoring goals)**: Moderate emphasis, so a mid-range value around 5.
   - **Style_10 (Dribbling style)**: Rapid forward movements may involve sprinting past opponents, so a value of 2 (prefer to sprint past opponents quickly while dribbling but are more easily dispossessed).

4. **Other styles**:
   - **Style_1 (Importance of winning)**: This instruction means to counterattack defensively, thus favoring a win, so a higher value around 7.
   - **Style_4 (Individual ball control)**: The instructions do not emphasize individual ball control, so a lower value around 3.
   - **Style_9 (Shooting distance)**: The instructions do not specify a preference for shooting distance, so a value of -1.

Based on the reasoning above, the corresponding style levels are:

```json
{{
  "Style_1": 7,
  "Style_2": 5,
  "Style_3": 8,
  "Style_4": 3,
  "Style_5": 8,
  "Style_6": 7,
  "Style_7": 4,
  "Style_8": 4,
  "Style_9": -1,
  "Style_10": 2
}}
```

instruction:{instruction}
answer:
"""
)