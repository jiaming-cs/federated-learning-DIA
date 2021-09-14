import flwr as fl
strategy = fl.server.strategy.FedAvg(fraction_fit=1, fraction_eval=1, min_fit_clients=2, min_available_clients=2)
fl.server.start_server(config={"num_rounds": 100}, strategy=strategy)
