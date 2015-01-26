CS 4641 - Machine Learning
==========================

 - What is machine learning?
    - The art, science, engineering, math, computing, of building systems that adapt and improve over time (typically in response to data)
    - Supervised
        - Neural Nets
        - Decision Trees
        - Function Approximation, given inputs and outputs, approximate a function that given an unknown input can guess an output
        - Inductive
            - Going from a series of datapoints to a general rule
            - "If you don't believe the sun is going to come up tomorrow, why are you here?"
        - in general, y=f(x), find f
    - Unsupervised
        - description or partition
        - efficient/compact way to describe the data
        - what is the best way
        - inductive bias in both
        - compression is a form of unsupervised learning
    - Reinforcement
        - you get results from actions
        - decision making
        - goal is to maximise "good" results from a series of actions
 - "some of you have called me Cedric, you racist bastards"

 - in machine learning, the single most important thing is the data, the algorithm is not important, at least not as important as the data
 - "Chess is like pacman, for sufficiently large values of pacman"

 - classification is function approximation, and thus deals with discrete values
    - you have discrete things that a classifier maps to
    - these are chosen based on aspects of the training data

 - decision trees
    - powerful
    - can, without loss of generality, represent all boolean functions
    - 2^2^n decision trees
    - must be clever when building decision trees
    - best attribute is the most important, is the one that removes the most questions
    - entropy has to do with random
        - you need to reduce randomness to have effective decision trees
    - Restriction bias
        - the set of functions you're willing to use
    - shorter trees are better than longer
        - fewer nodes/smaller
    - high information gain is better than low
    - correctness
    - IM3 to pick a good one
 - Overfitting is bad yo
    - don't
    - low error on training and high error on test
    - inferring about the noise instead of the underlying structure of the data

- Neural Nets
    - made up of perceptrons
        - a perceptrion is an output based on a set of weighted inputs and a linear transformation
    - can represent any arbitrary function
        - so picking the correct one is hard
        - use gradient descent algorithm to pick
            - gradient descent also has inductive bias
                - prefers systems with local maxima (perturbations cause no change)
                - correctness
                - prefers simpler functions