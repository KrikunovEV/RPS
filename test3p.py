from RockPaperScissors import Environment3p, Choice


env = Environment3p(steps=2, debug=True)

obs = env.reset()
choices = [Choice.ROCK, Choice.ROCK, Choice.ROCK]
print(env.action(choices))
print('Check: [1., 0., 0., 1., 0., 0., 1., 0., 0.],        [0., 0., 0.]')

obs = env.reset()
choices = [Choice.PAPER, Choice.PAPER, Choice.PAPER]
print(env.action(choices))
print('Check: [0., 1., 0., 0., 1., 0., 0., 1., 0.],        [0., 0., 0.]')

obs = env.reset()
choices = [Choice.SCISSORS, Choice.SCISSORS, Choice.SCISSORS]
print(env.action(choices))
print('Check: [0., 0., 1., 0., 0., 1., 0., 0., 1.],        [0., 0., 0.]')

obs = env.reset()
choices = [Choice.ROCK, Choice.PAPER, Choice.SCISSORS]
print(env.action(choices))
print('Check: [1., 0., 0., 0., 1., 0., 0., 0., 1.],        [0., 0., 0.]')

obs = env.reset()
choices = [Choice.SCISSORS, Choice.PAPER, Choice.ROCK]
print(env.action(choices))
print('Check: [0., 0., 1., 0., 1., 0., 1., 0., 0.],        [0., 0., 0.]')

obs = env.reset()
choices = [Choice.ROCK, Choice.SCISSORS, Choice.PAPER]
print(env.action(choices))
print('Check: [1., 0., 0., 0., 0., 1., 0., 1., 0.],        [0., 0., 0.]')

obs = env.reset()
choices = [Choice.ROCK, Choice.ROCK, Choice.PAPER]
print(env.action(choices))
print('Check: [1., 0., 0., 1., 0., 0., 0., 1., 0.],        [0., 0., 1.]')

obs = env.reset()
choices = [Choice.ROCK, Choice.ROCK, Choice.SCISSORS]
print(env.action(choices))
print('Check: [1., 0., 0., 1., 0., 0., 0., 0., 1.],        [0.5, 0.5, 0.]')

obs = env.reset()
choices = [Choice.SCISSORS, Choice.PAPER, Choice.SCISSORS]
print(env.action(choices))
print('Check: [0., 0., 1., 0., 1., 0., 0., 0., 1.],        [0.5, 0., 0.5]')