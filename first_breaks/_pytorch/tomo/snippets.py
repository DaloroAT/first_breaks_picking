import pandas as pd

from first_breaks.const import PROJECT_ROOT

df = pd.read_csv(PROJECT_ROOT / "surf_forward_times_fixed.ST", skipinitialspace=True, delimiter=" ", index_col=False)
print(df["ft"].max(), df["ft"].min())

assert False


model = Model(2, 10, 10)

optimizer = Adam(model.parameters(), lr=1e-2)

bsize = 100
target = 50
num_steps = 1000

pbar = tqdm(range(num_steps))

for _ in pbar:
    optimizer.zero_grad()

    point = torch.rand(size=(bsize, 2)).requires_grad_(True)
    target_values = target * torch.ones((bsize, 1))

    preds = model(point)
    loss = mse_loss(target_values, preds)
    loss.backward()
    optimizer.step()

    pbar.set_postfix_str(loss.item())






