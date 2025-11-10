import pygame
import heapq
import random

# --- Settings ---
GRID_SIZE = 10
CELL_SIZE = 50
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
OBSTACLE_COUNT = 10
MOVING_OBSTACLE_COUNT = 3
DELIVERY_COUNT = 4

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 50))  # extra space for stats
pygame.display.set_caption("AI Delivery Agent Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# --- Helper functions ---
def draw_grid(agent_pos, delivery_points, static_obs, moving_obs, distance_traveled, deliveries_done):
    screen.fill(WHITE)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = WHITE
            if (x, y) in static_obs:
                color = BLACK
            elif (x, y) in moving_obs:
                color = ORANGE
            elif (x, y) in delivery_points:
                color = GREEN
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)
    # Draw agent
    agent_rect = pygame.Rect(agent_pos[0]*CELL_SIZE, agent_pos[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, RED, agent_rect)
    # Stats
    stats_text = font.render(f"Distance: {distance_traveled}  Deliveries: {deliveries_done}/{DELIVERY_COUNT}", True, BLUE)
    screen.blit(stats_text, (10, WINDOW_SIZE + 10))
    pygame.display.flip()

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def a_star(start, goal, obstacles):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start:0}
    f_score = {start:heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        neighbors = [(0,1),(1,0),(0,-1),(-1,0)]
        for dx, dy in neighbors:
            neighbor = (current[0]+dx, current[1]+dy)
            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
                if neighbor in obstacles:
                    continue
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

def move_obstacles(moving_obs, static_obs, agent_pos, delivery_points):
    new_positions = []
    for ox, oy in moving_obs:
        dx, dy = random.choice([(0,1),(1,0),(0,-1),(-1,0),(0,0)])
        nx, ny = ox+dx, oy+dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            if (nx, ny) not in static_obs and (nx, ny) not in delivery_points and (nx, ny) != agent_pos:
                new_positions.append((nx, ny))
            else:
                new_positions.append((ox, oy))
        else:
            new_positions.append((ox, oy))
    return new_positions

def nearest_delivery(agent_pos, delivery_points):
    # Greedy nearest neighbor
    min_dist = float('inf')
    nearest = None
    for point in delivery_points:
        dist = heuristic(agent_pos, point)
        if dist < min_dist:
            min_dist = dist
            nearest = point
    return nearest

# --- Initialize grid ---
all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
static_obstacles = random.sample(all_positions, OBSTACLE_COUNT)
available_positions = [p for p in all_positions if p not in static_obstacles]
moving_obstacles = random.sample(available_positions, MOVING_OBSTACLE_COUNT)
available_positions = [p for p in available_positions if p not in moving_obstacles]
delivery_points = random.sample(available_positions, DELIVERY_COUNT)

agent_pos = (0,0)
distance_traveled = 0
deliveries_done = 0
current_target = nearest_delivery(agent_pos, delivery_points)
path = a_star(agent_pos, current_target, set(static_obstacles + moving_obstacles))

# --- Main loop ---
running = True
while running:
    clock.tick(5)  # speed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move obstacles
    moving_obstacles = move_obstacles(moving_obstacles, static_obstacles, agent_pos, delivery_points)
    
    # Check path validity
    if not path or any(p in moving_obstacles for p in path):
        path = a_star(agent_pos, current_target, set(static_obstacles + moving_obstacles))

    # Move agent
    if path:
        next_pos = path.pop(0)
        distance_traveled += 1
        agent_pos = next_pos

    # Check delivery
    if agent_pos == current_target:
        delivery_points.remove(current_target)
        deliveries_done += 1
        if delivery_points:
            current_target = nearest_delivery(agent_pos, delivery_points)
            path = a_star(agent_pos, current_target, set(static_obstacles + moving_obstacles))
        else:
            path = []

    draw_grid(agent_pos, delivery_points, static_obstacles, moving_obstacles, distance_traveled, deliveries_done)

pygame.quit()