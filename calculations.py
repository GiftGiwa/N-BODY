import numpy as np

G = 6.6742 * 10 ** -5


def distance(s1, s2):
    return np.linalg.norm([s1[:2], s2[:2]])


def force(m1, m2, s):
    return (G * m1 * m2) / s ** 2


def angle(vec1, vec2):
    return np.arctan2(vec2[1] - vec1[1], vec2[0] - vec1[0])


def forceVec(vec1, vec2):
    """Calculate the force vector between vec1 and vec2, where vec1 and vec2 are like [x, y, mass]

    The force vector should be of the form [Fx, Fy]
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    mass1 = vec1[2]
    mass2 = vec2[2]
    F = force(mass1, mass2, distance(vec1, vec2))
    θ = angle(vec1, vec2)
    return F * np.cos(θ), F * np.sin(θ)  # tuple


def nextPosition(vec, forcevec, delta_t):
    return vec[0] + vec[3] * delta_t, vec[1] + vec[4] * delta_t


def nextVelocity(vec, forcevec, delta_t):
    return (
        vec[3] + (forcevec[0] / vec[2]) * delta_t,
        vec[4] + (forcevec[1] / vec[2]) * delta_t,
    )


def nextPositionAndVelocity(vec, forcevec, delta_t):
    """Calculate the next velocity and position of vector based on the forces acting on it"""
    next_x, next_y = nextPosition(vec, forcevec, delta_t)
    next_vx, next_vy = nextVelocity(vec, forcevec, delta_t)
    vec[0] = next_x
    vec[1] = next_y
    vec[3] = next_vx
    vec[4] = next_vy


def sumOfForces(objects):
    """Given an Nx5 array of objects of the form (x, y, mass, xvel, yvel), return an Nx2 array of the cumulative force acting on each object in x and y"""
    # Non-numpy solution
    all_object_forces = []
    for obj in objects:
        object_forces = []

        # First calculate the forces
        for other_obj in objects:
            if isinstance(obj, np.ndarray) and (obj == other_obj).all():
                # if using numpy
                continue
            elif not isinstance(obj, np.ndarray) and obj == other_obj:
                # if not
                continue
            object_forces.append(forceVec(obj, other_obj))

        # then sum them up:
        sum_x = sum(x[0] for x in object_forces)
        sum_y = sum(x[1] for x in object_forces)
        all_object_forces.append([sum_x, sum_y])

    return all_object_forces


def centerOfMass(objects):
    """Calculate the center of mass of the objects"""
    #[x1, y1, mass1, velocityx1, velocityy1]
    moment_x = 0
    moment_y = 0
    total_mass = 0
    for object in objects:
        total_mass = total_mass + object[2]
        moment_x = moment_x + (object[0] * object[2])
        moment_y = moment_y + (object[1] * object[2])
    return [moment_x / total_mass, moment_y / total_mass]

