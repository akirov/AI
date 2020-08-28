# Breakout
# "Maximize your score in the Atari 2600 game Breakout. In this environment, the
# observation is an RGB image of the screen, which is an array of shape
# (210, 160, 3). Each action is repeatedly performed for a duration of kkk frames,
# where kkk is uniformly sampled from {2,3,4}."
# "v0 vs v4: v0 has repeat_action_probability of 0.25 (meaning 25% of the time the
# previous action will be used instead of the new action), while v4 has 0 (always
# follow your issued action)"
# Need to remove the score bar (the top 25 pixels)...
#
# Uses parts of https://karpathy.github.io/2016/05/31/rl/
