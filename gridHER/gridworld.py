import random


class GridWorld:
    def __init__(self, x_size=64, y_size=64, seed=0):
        if x_size * y_size < 2:
            raise ValueError("격자는 최소 두 칸 이상이어야 합니다.")

        self.x_size = x_size
        self.y_size = y_size

        self.start = (0, 0)
        self.goal = None
        self.grid = None

        self.random = random.Random(seed)

    def reset(self):
        self.grid = [
            [0 for _ in range(self.x_size)]
            for _ in range(self.y_size)
        ]

        while True:
            self.goal = (
                self.random.randrange(self.x_size),
                self.random.randrange(self.y_size),
            )

            if self.goal != self.start:
                break

        start_x, start_y = self.start
        goal_x, goal_y = self.goal

        self.grid[start_y][start_x] = "S"
        self.grid[goal_y][goal_x] = "G"

        return self.grid


if __name__ == "__main__":
    env = GridWorld()
    grid = env.reset()

    print("Start:", env.start)
    print("Goal:", env.goal)

    for row in reversed(grid):
        print(row)