
## DICEPOKER/ESCALERO-AN EPSILON GREEDY ALGORITHM

This report has originally been written as a Python/Jupyter notebook. The notebook itself will be part of the submission, in the form of .pdf and .ipynb. It is advised to use these notebooks because of better readability.

Additionally, the notebook can be found under: [RL-Dice-Poker-Escalero](https://www.kaggle.com/code/thomaskovarovics/rl-dice-poker-escalero)

### Basic Idea and Implementation

Dice poker, known as Escalero, is a dice game inspired by Poker. Players roll five dice up to three times. The score on a basic level is calculated by the maximum of: highest value you scored * number of dice showing that score. Additionally, players are aiming for combinations like Grande (Five of a kind) or Poker (Four of a kind). They record points on a table, and the player with the highest total points across columns wins. Strategy, and therefore the need for an optimal policy, comes into play on two points: 1. When selecting which of the five dice in the second and third throw are chosen to be rerolled. 2. Choosing where to place the score within the scoreboard. We will be focused on the first point in this project.

We interpret the classical Poker dice with the faces: Nine, Ten, Jack, Queen, King, and Ace as numerical values from 1 to 6. Our first implementation is focused on the basic scoring system.

The first part of the code is a user interface that allows us to roll the dice for a user. After that, we use an epsilon-greedy algorithm that simply optimizes the final score of one round (including one initial throw and two rerolls).

The goal is to further implement the special combinations that are part of the game, as well as the scoring board, that will eventually also allow to optimize the placement of scores within the scoreboard.

### RLModel 1 - Epsilon-Greedy

We are optimizing for the score of the previous game. This encapsulates the core of Dice Poker.

Our action space is every possible way to choose a subset of 5 dice to reroll them. Since for every dice we can choose to reroll the dice or not, we get 2 to the power of 5 possibilities.

Since our calculated score is not dependent on the order of the dice, we will not consider order within our state space. The number of possible states coincides with the number of choosing a 5-element multiset out of the numbers 1 to six. In general choosing a k-element multiset from a set of n numbers is C(n+k-1, k), therefore in our case, our state space has the size C(10,5) = 252.

This state space does not differentiate between an initial state, a state after the first reroll, and a state after the second reroll. This decision is made to keep the computations necessary minimal.

To accomplish this, we implement a slightly different rerolling function as before so that our rerolled dice are again an element of the state space.

We implement the epsilon-greedy function, which given a certain state and an exploration probability epsilon, will return the greedy choice, which is the action that has the maximum in the Q table assigned, (1-epsilon) times. It will return a random action epsilon times. The second choice will ensure exploration.

The q_learning function runs a given number of episodes. Each episode will initiate a state. Next, we will apply our epsilon-greedy function. Then we use the action that we received to reroll and evaluate the score at the new state. We update our Q Table based on the score we got and a fixed learning rate.

This first implementation shows a significant weakness. It takes a considerable amount of time, to be precise: exploration, to assign meaningful values to each (state, action) pair. Until then, the action() of not rerolling any dices gets assigned significantly more score because it will be chosen by default as the greedy action in the initial state of the Q Table, and furthermore as the greedy action because with the probability of (1-epsilon) it will be the action for a given state that first receives a value unequal 0.

To mitigate this, we have two options: 1. Use GLIE to have more exploration in the beginning and still ensure convergence. 2. Use a non-zero initial Q Table

We will start by exploring option 2. To calculate the average of scores of every state, we need to take the likelihood of each state occurring into account. To accomplish this, we create another state space, which does not take order into account. The probability of each state occurring at the initial roll is equal to the number of times it appears as an ordered tuple in this list, divided by the cardinality of this set which is 5 to the power of 6.

If every action is chosen at random, we will also reach each unordered state in this second state space with a uniform probability. Therefore the average score, that we are computing over this second state space, can also be seen as the average score that an algorithm that is choosing an action completely randomly is going to achieve.

This second state space is not involved in learning since it would be too large. But we will use it for further analysis later. We set our Q Table to this average score. Testing reveals that this has solved the previous problem almost entirely.

apply_policy allows us to play the game of rerolling a certain amount of dice on the set of all unordered states according to our learned Q Table at once. Afterward, we will evaluate the scores again. This will give us a clear indicator if we have improved over choosing actions randomly. To give further meaning to the score, we will take an average over five "games" of dice poker for all unordered states.

Clearly, we still have not optimized our Q Table. From looking at the states, where our Q Table indicates, we should not reroll any dice, we can see that we are not using the potential of these states to the fullest.

### RLModel 2 - GLIE

While we see quicker convergence due to the higher exploration in the beginning, we unfortunately do not see a better score. We will try again with a higher number of episodes.

### Implementing Special Rules

We are now implementing the special scoring rules Grande, Poker, Straight, and Full House into our scoring function.

We can see that overall the score is higher, but the general convergence stays the same.

### Conclusion and Future Directions

There is still a lot of room for improvement. Generally, the learning algorithm has problems finding the optimal action and will often stick to one of the best ones. A higher number of episodes has not shown an improvement in this regard. An optimization that does not rely on hardcoding certain actions is not obvious.

Expanding the state space to differentiate between a state after the initial roll or the first reroll would better encapsulate the game of Dice Poker. Further implementation could be made to include the scoreboard, which would be the second source of strategy in the game of dice poker.

We have generally succeeded with our goal to implement a reinforcement learning algorithm. We have shown that all our algorithms, epsilon-greedy and GLIE, with and without the special rule set, are able to perform better than the random choice of actions.

### Feedback & Bug Reports

This is an offline tool, your data stays locally and is not sent to any server!


We will now additionally implement GLIE.

With our current experience, we know that we have to stay on a high exploration first. If one takes a deep dive into the q_learning process, one can see that updating the Q Table itself has an element of randomness since given a state and an action there is the process of rerolling based on that action.

It is, therefore, necessary to reach every (state, action) pair often enough such that the Q Table entry for this pair can converge to the expected value.

Having an exploration probability that is too low too soon may rule certain (state, action) pairs that did have "bad luck" in the first couple of times that this pair was chosen. We implement glie_learning similar to q_learning. Because of the consideration we made earlier, we choose k such that epsilon converges linearly. This way we have a high exploration probability for a long time.

# DICEPOKER-RL
