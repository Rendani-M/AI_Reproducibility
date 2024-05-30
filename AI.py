import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

class testTasks:
    def __init__(self, num_tasks, min_moves=10, max_moves=100):
        self.num_tasks = num_tasks
        self.min_moves = min_moves
        self.max_moves = max_moves
        self.tasks = []

    def generate_task(self):
        def manhattan_distance(state):
            goal_position = {i: (i // 4, i % 4) for i in range(1, 16)}
            goal_position[0] = (3, 3)
            total_distance = 0
            for index, value in enumerate(state):
                if value != 0:
                    current_pos = (index // 4, index % 4)
                    goal_pos = goal_position[value]
                    total_distance += abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])
            return total_distance

        def valid_moves(state):
            index = state.index(0)
            moves = []
            if index % 4 > 0:  # Can move left
                moves.append('left')
            if index % 4 < 3:  # Can move right
                moves.append('right')
            if index // 4 > 0:  # Can move up
                moves.append('up')
            if index // 4 < 3:  # Can move down
                moves.append('down')
            return moves

        def apply_move(state, move):
            new_state = state[:]
            index = new_state.index(0)
            if move == 'left':
                new_index = index - 1
            elif move == 'right':
                new_index = index + 1
            elif move == 'up':
                new_index = index - 4
            elif move == 'down':
                new_index = index + 4
            new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
            return new_state

        goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
        current_state = goal_state[:]

        num_moves = random.randint(self.min_moves, self.max_moves)
        for _ in range(num_moves):
            move = random.choice(valid_moves(current_state))
            current_state = apply_move(current_state, move)

        cost = manhattan_distance(current_state)
        return (current_state, cost)

    def generate_tasks(self):
        self.tasks = [self.generate_task() for _ in range(self.num_tasks)]
        return self.tasks

class BayesianNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.25):
        super(BayesianNNModel, self).__init__()
        self.ffnn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, 2)  # Output both mean and log variance
        )
        self.wunn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, 2)  # Output both mean and log variance
        )
        self.ffnn_optimizer = optim.Adam(self.ffnn.parameters(), lr=0.001, weight_decay=1e-5)
        self.wunn_optimizer = optim.Adam(self.wunn.parameters(), lr=0.01, weight_decay=1e-5)
        self.clip_value = 1.0

    def forward(self, x):
        ffnn_output = self.ffnn(x)
        wunn_output = self.wunn(x)
        return ffnn_output, wunn_output

    def uncertainty_loss(self, output, target):
        mean, log_var = output[:, 0], output[:, 1]
        var = torch.exp(log_var)
        loss = (mean - target) ** 2 / (2 * var) + 0.5 * log_var
        return loss.mean()

    def train_model(self, D, epsilon, y_alpha):
        criterion = self.uncertainty_loss
        for epoch in range(1000):
            inputs = torch.tensor([x for x, y in D], dtype=torch.float32)
            labels = torch.tensor([y for x, y in D], dtype=torch.float32).unsqueeze(1)

            # Train FFNN
            self.ffnn_optimizer.zero_grad()
            ffnn_output, _ = self(inputs)
            loss_ffnn = criterion(ffnn_output, labels)
            loss_ffnn.backward()
            nn.utils.clip_grad_norm_(self.ffnn.parameters(), self.clip_value)
            self.ffnn_optimizer.step()

            # Train WUNN
            self.wunn_optimizer.zero_grad()
            _, wunn_output = self(inputs)
            loss_wunn = criterion(wunn_output, labels)
            loss_wunn.backward()
            nn.utils.clip_grad_norm_(self.wunn.parameters(), self.clip_value)
            self.wunn_optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, FFNN Loss: {loss_ffnn.item()}, WUNN Loss: {loss_wunn.item()}")

        print("Model training completed.")

    def predict(self, X):
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32)
            ffnn_output, wunn_output = self(inputs)
            ffnn_mean, ffnn_log_var = ffnn_output[:, 0], ffnn_output[:, 1]
            wunn_mean, wunn_log_var = wunn_output[:, 0], wunn_output[:, 1]
            ffnn_preds = ffnn_mean.numpy().flatten()
            wunn_preds = wunn_mean.numpy().flatten()
        return ffnn_preds, wunn_preds

class LearnHeuristicPrac:
    def __init__(self, num_tasks_per_iter, epsilon, alpha, max_iter, input_size=16, hidden_size=20):
        self.num_tasks_per_iter = num_tasks_per_iter
        self.epsilon = epsilon
        self.alpha = alpha
        self.y_alpha = None
        self.max_iter = max_iter
        self.model = BayesianNNModel(input_size, hidden_size)
        self.memory_buffer = []

    def generate_task(self, min_moves=10, max_moves=600):

        def manhattan_distance(state):
            goal_position = {i: (i // 4, i % 4) for i in range(1, 16)}
            goal_position[0] = (3, 3)
            total_distance = 0
            for index, value in enumerate(state):
                if value != 0:
                    current_pos = (index // 4, index % 4)
                    goal_pos = goal_position[value]
                    total_distance += abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])
            return total_distance

        def valid_moves(state):
            index = state.index(0)
            moves = []
            if index % 4 > 0:
                moves.append('left')
            if index % 4 < 3:
                moves.append('right')
            if index // 4 > 0:
                moves.append('up')
            if index // 4 < 3:
                moves.append('down')
            return moves

        def apply_move(state, move):
            new_state = state[:]
            index = new_state.index(0)
            if move == 'left':
                new_index = index - 1
            elif move == 'right':
                new_index = index + 1
            elif move == 'up':
                new_index = index - 4
            elif move == 'down':
                new_index = index + 4
            new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
            return new_state

        goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
        current_state = goal_state[:]

        if self.epsilon is None:
            final_max_moves = int(max_moves * self.alpha)
        else:
            final_max_moves = int(max_moves * self.epsilon)

        num_moves = random.randint(min_moves, max(final_max_moves, min_moves + 1))  # Ensure valid range

        for _ in range(num_moves):
            move = random.choice(valid_moves(current_state))
            current_state = apply_move(current_state, move)

        cost = manhattan_distance(current_state)
        return (current_state, cost)

    def generate_task_prac(self):
        print("Starting GenerateTaskPrac...")
        tasks = [self.generate_task() for _ in range(self.num_tasks_per_iter)]
        self.memory_buffer.extend(tasks)
        print("Generated tasks:", tasks)
        print("Task generation completed.")
        return tasks

    def compute_y_alpha(self):
        if not self.memory_buffer:
            self.generate_task_prac()
        costs = [cost for _, cost in self.memory_buffer]
        self.y_alpha = np.percentile(costs, (1 - self.alpha) * 100) / 100  # Scale to be in range (0, 1)
        # print(f"Computed y_alpha for alpha={self.alpha}: {self.y_alpha}")
        return self.y_alpha

    def adjust_epsilon(self, iteration):
        self.epsilon = self.epsilon * self.alpha
        # print(f"Adjusted epsilon for iteration {iteration}, alpha={self.alpha}: {self.epsilon}")

    def run(self):
        start_time = time.time()
        self.compute_y_alpha()  # Ensure y_alpha is computed before the first iteration
        for iteration in range(1, self.max_iter + 1):
            print(f"Iteration {iteration} / {self.max_iter}")
            tasks = self.generate_task_prac()
            self.model.train_model(tasks, self.epsilon, self.y_alpha)  # Pass epsilon and y_alpha
            self.adjust_epsilon(iteration)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time for LearnHeuristicPrac: {total_time:.2f} seconds")

        X_test = [x for x, y in self.memory_buffer]
        predicted_costs, _ = self.model.predict(X_test)

        # Round predicted costs to the nearest integer
        predicted_costs = np.round(predicted_costs)

        print("Predicted costs:", predicted_costs)

        optimal_costs = [y for x, y in self.memory_buffer]
        print("Optimal costs:", optimal_costs)

        number_of_optimal_predictions = sum(1 for pred, opt in zip(predicted_costs, optimal_costs) if pred == opt)
        suboptimality_values = [(pred - opt) / opt for pred, opt in zip(predicted_costs, optimal_costs)]
        average_suboptimality = np.mean(suboptimality_values) * 100  # percentage
        optimality_percentage = (number_of_optimal_predictions / len(self.memory_buffer)) * 100

        print(f"Train Average Suboptimality: {average_suboptimality:.2f}%")
        print(f"Train Optimality Percentage: {optimality_percentage:.2f}%")

def main():
    num_tasks_per_iter = 20
    epsilon = 1
    alpha = 0.9
    max_iter = 50
    input_size = 16  
    hidden_size = 20  

    model = LearnHeuristicPrac(num_tasks_per_iter, epsilon, alpha, max_iter, input_size, hidden_size)
    model.run()

    # Generate tasks with a wider range of difficulties
    min_moves = 10
    max_moves = 400  # Increased maximum moves for higher difficulty
    test_task_generator = testTasks(num_tasks=10, min_moves=min_moves, max_moves=max_moves)
    generated_tasks = test_task_generator.generate_tasks()

    print("Testing is starting")

    test_tasks = [(np.array(puzzle).flatten().tolist(), cost) for puzzle, cost in generated_tasks]

    X_test = [x for x, y in test_tasks]
    predicted_costs, _ = model.model.predict(X_test)

    # Round predicted costs to the nearest integer
    predicted_costs = np.round(predicted_costs)

    optimal_costs = [y for x, y in test_tasks]
    suboptimality_values = [(pred - opt) / opt for pred, opt in zip(predicted_costs, optimal_costs)]
    average_suboptimality = np.mean(suboptimality_values) * 100  # percentage
    number_of_optimal_predictions = sum(1 for pred, opt in zip(predicted_costs, optimal_costs) if pred == opt)
    optimality_percentage = (number_of_optimal_predictions / len(test_tasks)) * 100

    print(f"Average Suboptimality: {average_suboptimality:.2f}%")
    print(f"Optimality Percentage: {optimality_percentage:.2f}%")

    for i, (board, cost) in enumerate(test_tasks):
        print(f"\nInitial Board {i+1}:\n{np.array(board).reshape(4, 4)}\nInitial Cost: {cost}")
        print(f"Predicted Cost: {predicted_costs[i]}")

if __name__ == "__main__":
    main()
