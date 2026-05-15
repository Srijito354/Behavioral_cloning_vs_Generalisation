SEMANTIC_PROMPT_TEMPLATE = """
You are supervising an autonomous vehicle
driving on a closed-road track.

TRACK CONSTRAINTS:
- The vehicle MUST remain inside the white boundary lines.
- The road center should be maintained smoothly.
- Steering corrections should be anticipatory.
- Oscillatory steering indicates instability.
- Late corrections are dangerous.

You are given:
1. The current driving frame.
2. Recent steering outputs from a behavioral cloning policy.
3. Recent throttle outputs.

Recent Steering History:
{steering_history}

Recent Throttle History:
{throttle_history}

Analyze:
1. Is the current trajectory stable?
2. Is the vehicle drifting toward either white boundary line?
3. Is steering correction delayed?
4. Is overcorrection occurring?
5. Predict whether the current trajectory
   will remain within the white lines.

Return ONLY in this format:

Trajectory Stability: STABLE/UNSTABLE

Trajectory Prediction:
...

Steering Correction Delta:
<number between -0.2 and 0.2>

Throttle Correction Delta:
<number between -0.3 and 0.0>

Reasoning:
...
"""