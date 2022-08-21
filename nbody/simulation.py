from .calculations import sumOfForces, nextPositionAndVelocity


def simulateStep(objects, delta_t):
    """Given the objects, which have the form [x, y, mass, xvel, yvel], calculate the sum of forces
    acting on each object and UPDATE THE OBJECT with their next position and velocity"""
    forces = sumOfForces(objects)
    for i in range(len(objects)):
        nextPositionAndVelocity(objects[i], forces[i], delta_t)

