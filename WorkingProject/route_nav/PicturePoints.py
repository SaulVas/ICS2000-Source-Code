from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpBinary

def ILP_picture_points_problem(partition):
    problem = LpProblem("Picture_Point_Selection", LpMinimize)
    points = {(i, j): LpVariable(f"p_{i}_{j}", cat=LpBinary)
              for i in range(partition[0], partition[2])
              for j in range(partition[1], partition[3])}

    # Objective: Minimize the number of picture points
    problem += lpSum(points.values())

    # Constraints: Cover all cells in the partition
    for i in range(partition[0], partition[2]):
        for j in range(partition[1], partition[3]):
            covered_by = []
            # Include cells within 4 cells distance in each direction
            for di in range(-3, 4):  # Range from -4 to 4
                for dj in range(-4, 5):  # Range from -4 to 4
                    if (i + di, j + dj) in points:
                        covered_by.append(points[(i + di, j + dj)])

            # Ensure at least one point covers (i, j)
            problem += lpSum(covered_by) >= 1, f"Coverage_for_cell_{i}_{j}"

    problem.solve()
    return [(i, j) for (i, j), var in points.items() if var.value() == 1]

def generate_picture_points(grid_width, grid_height, partition_size=1):
    # Partitioning the grid (Variant 2)
    partitions = [(i, j, i + partition_size, j + partition_size)
                  for i in range(0, grid_height, partition_size)
                  for j in range(0, grid_width, partition_size)]
    picture_points = []
    for partition in partitions:
        picture_points.extend(ILP_picture_points_problem(partition))

    return picture_points
