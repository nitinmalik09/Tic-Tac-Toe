import numpy as np

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # 3x3 grid, 0 = empty
        self.current_player = 1  # 1 = X, -1 = O

    def reset(self):
        """Resets the game to the initial state."""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.board.flatten()  # Return the initial state

    def is_valid_move(self, row, col):
        """Checks if a move is valid."""
        return self.board[row, col] == 0

    def make_move(self, row, col):
        """Makes a move for the current player if valid."""
        if self.is_valid_move(row, col):
            self.board[row, col] = self.current_player
            return True
        return False

    def check_winner(self):
        """Checks if there is a winner or if the game is a draw."""
        # Check rows, columns, and diagonals
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return self.current_player  # Winner
        if abs(self.board.trace()) == 3 or abs(np.fliplr(self.board).trace()) == 3:
            return self.current_player
        # Check for a draw
        if not (self.board == 0).any():
            return 0  # Draw
        return None  # Game ongoing

    def switch_player(self):
        """Switches to the other player."""
        self.current_player *= -1

    def render(self):
        """Prints the board."""
        symbols = {1: 'X', -1: 'O', 0: ' '}
        for row in self.board:
            print('|'.join([symbols[cell] for cell in row]))
            print('-' * 5)

import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.q_table = defaultdict(float)  # Default value of 0 for unseen states
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state, available_actions):
        """Choose an action using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return random.choice(available_actions)  # Explore
        # Exploit: Choose the action with the highest Q-value
        q_values = [self.q_table[(state, action)] for action in available_actions]
        return available_actions[np.argmax(q_values)]

    def update_q_table(self, state, action, reward, next_state, next_available_actions, done):
        """Update the Q-value using the Bellman equation."""
        best_next_q = max([self.q_table[(next_state, a)] for a in next_available_actions], default=0)
        target = reward + (0 if done else self.discount_factor * best_next_q)
        self.q_table[(state, action)] += self.learning_rate * (target - self.q_table[(state, action)])

    def decay_epsilon(self):
        """Decay the exploration rate."""
        self.epsilon = max(0.1, self.epsilon * self.epsilon_decay)

def train_agent(episodes=10000):
    env = TicTacToeEnv()
    agent = QLearningAgent()
    
    for episode in range(episodes):
        state = tuple(env.reset())  # Get initial state
        done = False
        
        while not done:
            available_actions = [(r, c) for r in range(3) for c in range(3) if env.is_valid_move(r, c)]
            action = agent.choose_action(state, available_actions)
            
            # Perform the chosen action
            env.make_move(*action)
            next_state = tuple(env.board.flatten())
            reward = 0
            winner = env.check_winner()
            if winner is not None:
                done = True
                reward = 10 if winner == env.current_player else -10
                if winner == 0:  # Draw
                    reward = 1

            # Update Q-table
            next_available_actions = [(r, c) for r in range(3) for c in range(3) if env.is_valid_move(r, c)]
            agent.update_q_table(state, action, reward, next_state, next_available_actions, done)
            
            state = next_state
            env.switch_player()

        agent.decay_epsilon()

    return agent

def play_against_agent(agent):
    env = TicTacToeEnv()
    state = tuple(env.reset())
    
    choice = input("Who should go first? Enter 'me' for yourself or 'agent' for the agent: ").strip().lower()
    if choice == 'agent':
        env.current_player = -1  # Agent goes first
    elif choice == 'me':
        env.current_player = 1  # Human goes first
    else:
        print("Invalid choice. Defaulting to 'agent' going first.")
        env.current_player = -1  # Default to agent if input is invalid

    done = False

    while not done:
        env.render()
        if env.current_player == 1:
            row, col = map(int, input("Enter your move (row col): ").split())
        else:
            available_actions = [(r, c) for r in range(3) for c in range(3) if env.is_valid_move(r, c)]
            action = agent.choose_action(state, available_actions)
            row, col = action
            print(f"Agent plays: {row} {col}")

        if not env.make_move(row, col):
            print("Invalid move. Try again.")
            continue

        state = tuple(env.board.flatten())
        winner = env.check_winner()
        if winner is not None:
            env.render()
            if winner == 1:
                print("You win!")
            elif winner == -1:
                print("Agent wins!")
            else:
                print("It's a draw!")
            done = True

        env.switch_player()
if __name__ == "__main__":
    print("Starting training...")
    agent = train_agent()  # Train the agent for a given number of episodes
    print("Training completed. Now you can test the agent.")
    play_against_agent(agent)  # Test the agent by playing against it
