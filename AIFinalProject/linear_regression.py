# %%
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from DobutsuBoard import DobutsuGameState, Score, gamestate_to_bits, input_to_move, root_state
from joblib import dump, load

'''Model trained here.'''

# %%
# Read training data
classical = pd.read_csv("moves_classical.csv")
nnue = pd.read_csv("moves_nnue.csv")

# %%
# Increase game_id to differentiate from classical games.
incremented_nnue = nnue.copy(deep=True)

incremented_nnue.iloc[:,0] = nnue.iloc[:,0].apply(lambda x: x + 11)

incremented_nnue.head()
# %%
# Union of two datasets.
all_evals = pd.concat([classical, incremented_nnue])
print(all_evals)

print(all_evals.columns[0])
# %%
# Group moves into games.
games = all_evals.groupby('game_id')
print(len(games))
# %%
# Split group into list.
game_datasets : list[pd.DataFrame] = [games.get_group(index) for index in range(len(games))]

# Filter out games, if necessary.
game_datasets = game_datasets[11:]

# Map of game state to evaluation
position_eval_map = dict[DobutsuGameState, Score]()

# %%

# Transform moves into sequence of gamestates.
for game_df in game_datasets:
    
    gamestate = root_state.instance

    for row in game_df.iterrows():
        position_eval_map[gamestate] = row[1][3]
        gamestate = gamestate.move(input_to_move(row[1][2]))
# %%
print(len(position_eval_map))

# %%
'''
Flatten inputs into 1D array input for Linear Regression.
    is_white bool is a categorical value, so it is one-hot encoded.
    Bitboards are flatten into 1D lists of 0's and 1's.
    [0, 1, 2, 3],
    [4, 5, 6, 7], -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    [8, 9, 10, 11]
    Hand dicts are converted into 1D lists: [count, count, count]
Arrangement:
[is_white][is_black][white][black][Chick->Lion][hand_white][hand_black]
'''

flattened_inputs = [gamestate_to_bits(gamestate)
                    for gamestate in position_eval_map]

# %%
X = flattened_inputs
y = [eval for _, eval in position_eval_map.items()]

print(y)

def test(test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, random_state = 1)

    lr = LinearRegression()

    lr.fit(X_train, y_train)

    y_pred_test = lr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred_test)

    print(f'MSE: {mse}')

    r2 = r2_score(y_test, y_pred_test)

    print(f'R-squared: {r2}')

for i in [0.1, 0.15, 0.2, 0.25, 0.3]:
    test(i)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 1)

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred_test = lr.predict(X_test)

dump(lr, 'linear_regression.joblin')

# %%
print(lr.predict([gamestate_to_bits(root_state.instance)])[0]) # type: ignore
print(position_eval_map[root_state.instance])
# %%
