import numpy as np

# POPULATE ARRAYS =========================================================================

# np.array
one_dimensional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
print(one_dimensional_array)

# np.array
two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8]])
print(two_dimensional_array)

# np.zeros
two_dimensional_array = np.zeros([3,3])
print(two_dimensional_array)

# np.ones
two_dimensional_array = np.ones([3,3])
print(two_dimensional_array)

# np.arrange
sequence_of_integers = np.arange(5, 12)
print(sequence_of_integers)

# np.random.randint for random ints
random_integers_between_50_and_100 = np.random.randint(low=50, high=101, size=(6))
print(random_integers_between_50_and_100)

# np.random.random for random floats between 0.0 and 1.0
random_floats_between_0_and_1 = np.random.random([6])
print(random_floats_between_0_and_1) 

# MATH=====================================================================================

# add x to every item in the vector
random_floats_between_2_and_3 = random_floats_between_0_and_1 + 2.0
print(random_floats_between_2_and_3)

# multiply each cell in a vector by x
random_integers_between_150_and_300 = random_integers_between_50_and_100 * 3
print(random_integers_between_150_and_300)

# Create linear Dataset====================================================================

# simple dataset of a single feature and a label
feature = np.arange(6, 21)
print(feature)
label = (feature * 3) + 4
print(label)

# adding noise to dataset
noise = (np.random.random([15]) * 4) - 2
print(noise)
label = label + noise 
print(label)