import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from statsmodels.stats.moment_helpers import corr2cov

def eigsorted(M):
    vals, vecs = np.linalg.eigh(M)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def orientation_from_covariance(M, sigma=1):
    vals, vecs = eigsorted(M)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * sigma * np.sqrt(np.abs(vals))
    return w, h, theta

def get_ellipse(mu_x, mu_y, covariance, color='blue', linewidth=2, alpha=0.5):
    h, w, a = orientation_from_covariance(covariance, 2)
    angle = 90 - a
    el = Ellipse((mu_x, mu_y), w, h, angle=angle)
    el.set_alpha(alpha)
    el.set_linewidth(linewidth)
    el.set_edgecolor(color)
    el.set_facecolor(color)
    el.set_fill(True)
    return el

def get_example(angle):
    M = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    mu = np.array([0, 0])
    return mu, M