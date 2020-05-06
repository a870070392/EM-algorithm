# implemented with pure numpy broadcasting without a single for loop!
import numpy as np
import pandas as pd


def convert(i):
    if i == '1':
        return 1
    elif i == '0':
        return 0
    else:
        return -1


with open('ratings.txt') as f:
    ratings = np.array([[convert(i) for i in line.strip().split()] for line in f])

with open('movieTitles.txt') as f:
    titles = np.array([line.strip() for line in f])

order = np.argsort(np.ma.array(ratings, mask=ratings == -1).mean(axis=0))

with open('probZ_init.txt') as f:
    pz = np.array([float(line.strip()) for line in f])

with open('probRgivenZ_init.txt') as f:
    pRofZ = np.array([[float(num) for num in line.strip().split()] for line in f])

k = len(pz)
ratings = np.ma.array(ratings, mask=ratings == -1)
T = len(ratings)
movies = len(titles)


def likelihood() -> float:
    return np.log(np.matmul(np.abs(1 - ratings[:, :, np.newaxis] - pRofZ).prod(axis=1), pz)).mean()


postProb = np.empty((k, T))


def E():
    global postProb
    postProb = (pz * np.abs(1 - ratings[:, :, np.newaxis] - pRofZ).prod(axis=1)).T
    postProb /= postProb.sum(axis=0)
    postProb = postProb.data


def M():
    global pz, pRofZ
    pz = postProb.mean(axis=1)
    arr = np.where(np.tile((~ratings.mask)[:, :, np.newaxis], (1, 1, k)),
                   np.tile((ratings == 1)[:, :, np.newaxis], (1, 1, k)),
                   np.tile(pRofZ, (T, 1, 1)))
    pRofZ = np.matmul(arr.T, postProb[:, :, np.newaxis]).squeeze().T
    pRofZ /= postProb.sum(axis=1)


iterations = np.power(2, range(8))
iterations = np.insert(iterations, 0, 0)

iteration = 0
L = []
while True:
    if iteration in iterations:
        L.append(likelihood())
    E()
    M()
    iteration += 1
    if iteration > 128:
        break

table = pd.DataFrame(L, index=iterations, columns=['log-likelihood'])
with open('studentPIDs.txt') as f:
    row = [line.strip() for line in f].index('none')

ratings = ratings[row]
predict = np.matmul(postProb[:, row], pRofZ.T) * ratings.mask

predict = pd.DataFrame(predict, index=titles)[predict != 0].sort_values(0,ascending=False)
