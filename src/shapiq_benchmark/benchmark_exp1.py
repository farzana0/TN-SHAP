from shapiq.benchmark import load_games_from_configuration

game_identifier = "SentimentAnalysisLocalXAI"  # local XAI benchmark
config_id = 1
n_player_id = 0
n_games = 1   # just take one game to start

games = list(load_games_from_configuration(
    game_class=game_identifier,
    n_player_id=n_player_id,
    config_id=config_id,
    n_games=n_games,
))
game = games[0]
print(game, game.n_players)  # Game(14 players, ...)
