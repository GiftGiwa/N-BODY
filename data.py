import numpy as np

from .calculations import centerOfMass, angle, distance


def generateObjects(count, center=0, width=10, massmult=100, vcenter=0, vwidth=10):
    """Generate a `count` x 5 matrix of (x, y) coordinates corresponding
    to `count` number of arbitrarly placed elements

    Args:
        count (int): the number of (x, y) pairs to generate
        center (float): the center of our distribution
        width (float): the width of our distribution
        massmult (float): mass multiplier
        vcenter (float): the center of our velocity distribution
        vwidth (float): the width of our velocity distribution

    Returns:
        a (`count`, 5) list/array of positions

        The return value should look like:
            [[x1, y1, mass1, velocityx1, velocityy1],
             [x2, y2, mass2, velocityx2, velocityy2],
             ...
            ]
    """
    # Numpy
    positions = width * np.random.random((count, 2)) - ((width / 2) - center)
    weights = np.random.random((count, 1)) * massmult
    velocities = vwidth * np.random.random((count, 2)) - ((vwidth / 2) - vcenter)
    data = np.concatenate((positions, weights, velocities), axis=1)
    return data


def generateObjectsNormal(count, center=0, width=100000, massmult=10, vmult=100):
    """Generate objects in a 2d normal distribution (to simulate a galaxy) with initial velocities
    roughly orthogonal to the center of mass in a e.g. rotating"""

    # first, generate the position and mass distribution
    positions = np.random.multivariate_normal(
        [center, center],
        [
            [width, 0],
            [0, width],
        ],
        count,
    )
    weights = np.random.random((count, 1)) * massmult
    data = np.concatenate((positions, weights), axis=1)

    # now calculate the center of mass
    com = centerOfMass(data)

    # now get the angles between the com and the data points
    angles = [angle(com, datum[:2]) for datum in data]

    # if each point is θ from com, then 90+θ (π/2+θ rad) should be good
    angles = angles + np.ones(count) * np.pi / 2
    velocity_magnitudes = np.array([vmult * np.pi * (distance(com, vec[:2]) / (massmult * width) ) for vec in data])
    velocities = np.array(
        [
            [np.cos(theta) * mag, np.sin(theta) * mag]
            for theta, mag in zip(angles, velocity_magnitudes)
        ]
    )

    # concat to data
    data = np.concatenate((data, velocities), axis=1)

    # as a last step, put a black hole at the center
    data[0] = np.array([center, center, data[:, 2].max() * 10000, 0, 0])

    return data
