import flwr as fl
strategy = fl.server.strategy.FedAvg(fraction_fit=1, fraction_eval=1)
fl.server.start_server(config={"num_rounds": 10}, strategy=strategy)
