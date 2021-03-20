import pandas as pd
import math
import sys
from diffprivlib.mechanisms import Laplace

def add_noise_to_rating(rating, mechanism):
    nr = mechanism.randomise(rating)
    if nr > 5.0:
        nr = 5.0
    if nr < 1.0:
        nr = 1.0
    return nr

dataset = sys.argv[1]
df = pd.read_csv(dataset)

l1_sens = df['Rate'].max() - df['Rate'].min()
# Generates private datasets with epsilon = 3, 10, 100, 10000
for i in [3, 10, 100, 1000]:
    lap = Laplace(epsilon=math.log(i), sensitivity=l1_sens)
    df['private_ratings'] = df.apply(lambda row: add_noise_to_rating(row['Rate'], lap), axis = 1)
    df.to_csv('private_slice_ratings_' + str(i) + '.csv')


