from numba import jit
import numpy as np


@jit(nopython=True)
def dfs(matrix: np.ndarray, visited: np.ndarray, i: int, j: int, M: int, N: int):
    """
    Perform depth-first search (DFS) to mark all cells of the current island as visited.
    """
    if i < 0 or i >= M or j < 0 or j >= N or matrix[i, j] == 0 or visited[i, j]:
        return

    visited[i, j] = True  # Mark current cell as visited

    # Explore the neighboring cells (up, down, left, right)
    dfs(matrix, visited, i - 1, j, M, N)  # up
    dfs(matrix, visited, i + 1, j, M, N)  # down
    dfs(matrix, visited, i, j - 1, M, N)  # left
    dfs(matrix, visited, i, j + 1, M, N)  # right


@jit(nopython=True)
def count_islands(matrix: np.ndarray, M: int, N: int) -> int:
    """
    Count the number of islands (connected components of 1s) in the matrix.
    """
    visited = np.zeros((M, N), dtype=np.bool_)  # Initialize visited boolean array
    island_count = 0

    # Traverse the matrix to find unvisited land cells (1) and start DFS from there
    for i in range(M):
        for j in range(N):
            if matrix[i, j] == 1 and not visited[i, j]:
                dfs(matrix, visited, i, j, M, N)
                island_count += 1

    return island_count


def validate_input_size(M: int, N: int):
    """
    Ensure the input matrix dimensions are within valid bounds.
    """
    max_size = 100_000
    if not (0 < M <= max_size and 0 < N <= max_size):
        raise ValueError(
            f"Matrix dimensions are out of bounds. Max {max_size} x {max_size} allowed."
        )


def validate_matrix(matrix: np.ndarray, M: int, N: int):
    """
    Validate matrix dimensions and contents (only 0 and 1 values allowed).
    """
    if matrix.shape != (M, N):
        raise ValueError("Matrix dimensions do not match the provided data.")

    if not np.all(np.isin(matrix, [0, 1])):
        raise ValueError("Matrix can only contain 0 or 1.")


def get_matrix_input(M: int, N: int) -> np.ndarray:
    """
    Prompt user for matrix input and convert it to a numpy array.
    """
    return np.array([list(map(int, input().split())) for _ in range(M)], dtype=np.int8)


def main():
    """
    Main function to handle input, validation, and island counting.
    """
    try:
        # Get matrix dimensions
        M, N = map(int, input("Input:\n").split())
        validate_input_size(M, N)

        # Get matrix data from input
        matrix = get_matrix_input(M, N)
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
