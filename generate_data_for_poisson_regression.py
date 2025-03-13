import numpy as np
import pandas as pd

np.random.seed(123)
y = np.random.poisson(3.5, 50)
data = pd.DataFrame({"y": y})
data.to_csv("poisson_data.csv")
