from numba import jit
import numpy as np


@jit(nopython=True)
def dfs(matrix: np.ndarray, visited: np.ndarray, i: int, j: int, M: int, N: int):
    # Base conditions: check if we are out of bounds or in water or already visited
    if i < 0 or i >= M or j < 0 or j >= N or matrix[i, j] == 0 or visited[i, j]:
        return

    # Mark the current cell as visited
    visited[i, j] = True

    # Check all 4 directions (up, down, left, right)
    dfs(matrix, visited, i - 1, j, M, N)  # up
    dfs(matrix, visited, i + 1, j, M, N)  # down
    dfs(matrix, visited, i, j - 1, M, N)  # left
    dfs(matrix, visited, i, j + 1, M, N)  # right


# Main function to count islands
@jit(nopython=True)
def count_islands(matrix: np.ndarray, M: int, N: int) -> int:
    """Count the number of islands (connected components of 1s) in the matrix."""
    visited = np.full((M, N), False, dtype=np.bool_)  # Efficient boolean array creation
    island_count = 0

    # Traverse the matrix
    for i in range(M):
        for j in range(N):
            # If a land cell is found and it has not been visited, it's a new island
            if matrix[i, j] == 1 and not visited[i, j]:
                dfs(matrix, visited, i, j, M, N)
                island_count += 1

    return island_count


def validate_input_size(M: int, N: int):
    if not (0 < M <= 100_000 and 0 < N <= 100_000):  # Limiting to reasonable size
        raise ValueError(
            "Matrix dimensions are out of bounds. Max 100 000 x 100 000 allowed."
        )


def validate_matrix(matrix: np.ndarray, M: int, N: int):
    if len(matrix) != M or any(len(row) != N for row in matrix):
        raise ValueError("Matrix dimensions do not match the provided data.")

    for row in matrix:
        if not all(el in [0, 1] for el in row):
            raise ValueError("Matrix can only contain 0 or 1.")


def main():
    """Main function to handle input, validate data, and count islands."""
    try:
        # Input matrix dimensions
        M, N = map(int, input("Input:\n").split())
        validate_input_size(M, N)

        # Input matrix data
        matrix = np.array([list(map(int, input().split())) for _ in range(M)])
        validate_matrix(matrix, M, N)

        # Count and display the number of islands
        output = count_islands(matrix, M, N)
        print(f"Output: {output}")

    except ValueError as e:
        print(f"Input error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
