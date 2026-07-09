criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Net.parameters(), lr=1e-3)
# can also try sgd or adamw later
