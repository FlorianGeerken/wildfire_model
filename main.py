import rioxarray
import xarray
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib

matplotlib.use('Qt5Agg')  # Switch to Qt backend

# Define the custom colormap for animation
brown_white_red = LinearSegmentedColormap.from_list(
    "BrownWhiteRed", ["firebrick", "white", "red"]
)

# Input variables
windspeed = 1  # 1: low, 2: high
wind_direction = 0  # 0: east, 90: south, 180: west, 270: north

# Define the land-use to probability mapping
landuse_to_probability = {
    0: 0,
    1: 0.5 * windspeed,
    2: 0.7 * windspeed,
    5: 0.5 * windspeed,
    6: 0.4,
    8: 0.8,
    10: 0.8,
    11: 0.4,
    12: 0.4,
    13: 0.4,
    14: 0.4,
    15: 0.3,
    16: 0.2,
    17: 0.2,
    18: 0.03 * windspeed,
    19: 0,
}

# Load land-use raster
def load_landuse():
    raster_path = "data/landuse_4326.tif"
    raster = rioxarray.open_rasterio(raster_path)
    raster_array = raster.values[0]
    return raster_array, raster

landuse_raster_array, landuse_raster = load_landuse()

# Assign probabilities based on land-use raster
def assign_landuse_probabilities(landuse_raster):
    landuse_prob_array = landuse_raster.copy(deep=True)
    for landuse_type, probability in landuse_to_probability.items():
        landuse_prob_array = landuse_prob_array.where(landuse_raster != landuse_type, probability)
    return landuse_prob_array

landuse_prob_array = assign_landuse_probabilities(landuse_raster)

# Load slope raster and normalize
def load_slope(landuse_raster):
    raster_path = "data/slope_4326.tif"
    raster_slope = rioxarray.open_rasterio(raster_path).rio.reproject_match(landuse_raster)
    raster_slope = (raster_slope - raster_slope.min()) / (raster_slope.max() - raster_slope.min())
    return raster_slope

raster_slope = load_slope(landuse_raster)

# Combine land-use probabilities with slope
grid = landuse_prob_array * raster_slope

# Parse .clr file to create a colormap
def parse_clr_file(clr_file_path):
    colors = []
    with open(clr_file_path, 'r') as clr_file:
        for line in clr_file:
            parts = line.strip().split()
            if len(parts) == 6:  # Index R G B A Value
                r, g, b, a = map(int, parts[1:5])
                colors.append((r / 255.0, g / 255.0, b / 255.0, a / 255.0))
    return ListedColormap(colors, name="CustomCLR")

clr_file_path = "cmap.clr"
basemap_cmap = parse_clr_file(clr_file_path)

# Stochastic spread with wind
def stochastic_spread_with_wind(grid, max_steps=1000, wind_direction=wind_direction, wind_weight=windspeed):
    if "band" in grid.dims:
        grid = grid.sel(band=1)

    grid_array = grid.values
    rows, cols = grid_array.shape
    # Start point
    start_row = random.randint(0, rows - 1)
    start_col = random.randint(0, cols - 1)
    # start_row, start_col = 188, 529
    # print(start_row)
    # print(start_col)
    spread_mask = np.zeros_like(grid_array, dtype=bool)
    spread_mask[start_row, start_col] = True

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    wind_angle_rad = math.radians(wind_direction)
    wind_vector = (math.sin(wind_angle_rad), math.cos(wind_angle_rad))

    def compute_wind_bias(dr, dc):
        neighbor_vector = (dr, dc)
        dot_product = (wind_vector[0] * neighbor_vector[0] +
                       wind_vector[1] * neighbor_vector[1])
        return max(0, dot_product) * wind_weight

    wind_biases = {neighbor: compute_wind_bias(*neighbor) for neighbor in neighbors}
    frontier = [(start_row, start_col)]
    steps = 0
    total_spread = 0
    spread_mask_history = [spread_mask.copy()]

    while frontier and steps < max_steps:
        new_frontier = []
        for row, col in frontier:
            for dr, dc in neighbors:
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols and not spread_mask[nr, nc]:
                    base_probability = grid_array[nr, nc]
                    wind_bias = wind_biases.get((dr, dc), 0)
                    augmented_probability = min(base_probability + wind_bias, 1.0)
                    if random.random() < augmented_probability * (0.3 + (0.1 * wind_weight)):
                        spread_mask[nr, nc] = True
                        new_frontier.append((nr, nc))
        frontier = new_frontier
        total_spread += len(frontier)
        spread_mask_history.append(spread_mask.copy())
        steps += 1

    return spread_mask_history, total_spread

# Animate spread
def animate_spread(grid, spread_mask_history, basemap_data, basemap_cmap):
    fig, ax = plt.subplots()

    # Display the basemap
    img = ax.imshow(
        basemap_data, cmap=basemap_cmap, interpolation="nearest", vmin=0, vmax=len(basemap_cmap.colors) - 1
    )

    # Initialize the overlay for the spread animation
    overlay = ax.imshow(np.zeros_like(grid), cmap=brown_white_red, alpha=0.65, vmin=-1, vmax=1)

    # Update function for animation
    def update(frame):
        current_spread = spread_mask_history[frame]
        prev_spread = spread_mask_history[frame - 1] if frame > 0 else np.zeros_like(grid, dtype=bool)
        new_frontier = current_spread & ~prev_spread

        # Calculate the number of pixels burned in the current frame
        pix_spread = np.sum(current_spread)  # Count the pixels that have burned
        perc_burned = pix_spread / grid.size * 100  # Percentage of area burned

        overlay_data = np.zeros_like(grid, dtype=float)
        overlay_data[current_spread] = -1  # Spread area
        overlay_data[new_frontier] = 1  # New frontier

        overlay.set_data(overlay_data)
        ax.set_title(f"Step {frame} - Percentage of area burned: {perc_burned:.2f}%")
        #ax.set_title(f"Step {frame} - Spread Progress")

    anim = FuncAnimation(fig, update, frames=len(spread_mask_history), interval=200, repeat=False)
    plt.show()
    return anim

# Run simulation and animation
if __name__ == "__main__":
    spread_mask_history, total_spread = stochastic_spread_with_wind(
        grid, max_steps=1000, wind_direction=wind_direction, wind_weight=windspeed
    )
    print("Percentage burned:", total_spread / grid.size * 100, "%")

    print(np.unique(landuse_raster_array))
    # Get unique values and their counts
    unique_values, counts = np.unique(landuse_raster_array, return_counts=True)

    # Print results
    print("Unique values:", unique_values)
    print("Counts:", counts)
    print(len(landuse_raster_array.shape))
    # Remove the band dimension from the base map
    anim = animate_spread(grid.sel(band=1).values, spread_mask_history, landuse_raster_array, basemap_cmap)
    print("Total spread:", total_spread)

