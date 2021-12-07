import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas_datareader import data as wb


def getStocks(companies, start, end):
    df = pd.DataFrame()

    for a in companies:
        df[a] = wb.DataReader(a, data_source='yahoo', start=start, end=end)['Adj Close']

    return df


def getReturns(df):
    '''Compute the log return of a dataframe'''
    return np.log(df / df.shift(1))


def get_returns_and_cov(companies, start, end):
    '''
    Args:
        companies: list of comapnies
        start: string, starting date
        end: string, ending date

    Returns:
        the returns (array) and the covariance matrix of companies. Those data are annual.
    '''
    df = getStocks(companies, start, end)
    log_returns = getReturns(df)
    return log_returns.mean()*250, log_returns.cov()*250


def decode_bits_sequence(bits, Nq):
    '''

    Args:
        bits: list of 0 and 1
        Nq: Number of bits used to encode a discrete value

    Returns:
        A list with discrete values.
    '''
    N = int(len(bits) / Nq)
    step = 1 / (2 ** Nq - 1)
    weights = []

    for n in range(N):
        sq = 0
        for q in range(Nq):
            sq += (2 ** q) * bits[n * Nq + q]
        weights.append(sq * step)

    return weights


def plot_random_portfolio(mu, Sigma, Npoints=1000, solution=None):
    returns = []
    volatilities = []
    N = Sigma.shape[0]

    plt.figure(figsize=(10, 7))

    if solution != None:
        Nq = int(len(solution) / N)
        weight0 = decode_bits_sequence(solution, Nq)
        return0 = np.dot(weight0, mu)
        risk0 = np.sqrt(np.dot(weight0, np.dot(Sigma, weight0)))
        plt.plot([risk0], [return0], '*', c='orange', label='solution found', ms=12)

    for x in range(Npoints):
        weights = np.random.random(N)
        weights /= np.sum(weights)
        returns.append(np.sum(weights * mu))  # annual basis
        volatilities.append(np.sqrt(np.dot(weights.T, np.dot(Sigma, weights))))

    plt.plot(volatilities, returns, 'o', alpha=0.2, label='random portfolio')
    plt.title('return-volatility of random portfolios')
    plt.xlabel('volatility')
    plt.ylabel('return')
    plt.legend()
    plt.show()


def portfolio_to_qubo(mu, Sigma, gamma, rho, Nq=1):

    '''
    Function that maps a portfolio optimization problem into a QUBO.
    E = -omega x mu + gamma x omega^T x Sigma x omega + rho x (sum(omega) - 1)^2
    where omega is a vector with discrete values that will be encoded with Nq bits.
    Args:
        mu: returns vector
        Sigma:  covariance matrix
        gamma:  risk adversion
        rho:  penalty for omega whose sums are not 1
        Nq: number of bits on which we encode the discrete value of omega's component.

    Returns:
        The matrix Q such that minimizing E is equivalent to minimizing x^T Q x where x are
        binary vectors

    NB: there should be a constant between the two expressions (=rho) that shouldn't matter for the optimization.
    '''

    if isinstance(Sigma, pd.DataFrame):
        Sigma = Sigma.to_numpy()

    step = 1 / (2 ** Nq - 1)
    Na = Sigma.shape[0]
    M = Na*Nq
    Q = np.zeros((M, M))
    for i in range(Na):
        for j in range(Na):
            for q in range(Nq):
                for l in range(Nq):
                    i2 = i * Nq + q
                    j2 = j * Nq + l
                    Q[i2, j2] = + gamma * (2 ** q) * (2 ** l) * Sigma[i, j] * (step ** 2)
                    Q[i2, j2] += rho * (2 ** q) * (2 ** l) * (step ** 2)
                    if i == j:
                        if q == l:
                            Q[i2, j2] -= mu[i] * (2 ** q) * step
                            Q[i2, j2] += rho*(-2 * (2 ** q) * step)

    return Q
