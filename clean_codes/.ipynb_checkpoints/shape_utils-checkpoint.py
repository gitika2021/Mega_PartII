import numpy as np
from scipy.special import comb
import sys
from tqdm import tqdm
import os

# Define the constants for the shape generation
SHAPE_SIZE = 38
N_VERTICES = 16  # Each shape will be a closed loop made of 8 cubic Bezier segments.
NUM_STEPS = 100 # Higher steps for smoother line drawing

def bernstein_polynomial(n, k, t):
    """
    Calculates the Bernstein polynomial B_n,k(t), the mathematical basis for Bezier curves.
    This is used to compute the intermediate points on the curve.
    """
    if t.ndim == 0:
        # Handle single float input
        return comb(n, k) * (t ** k) * ((1 - t) ** (n - k))
    else:
        # Handle numpy array input
        return comb(n, k) * (t ** k) * (np.power(1 - t, n - k))

def bezier_curve_points(points, num_steps=NUM_STEPS):
    """
    Generates a dense set of points along a single Bezier curve segment 
    defined by a list of control points.

    Args:
        points (np.ndarray): Array of shape (N+1, 2) where N is the degree (e.g., 4 points for cubic).
        num_steps (int): The number of intermediate points to generate along the curve.

    Returns:
        np.ndarray: An array of shape (num_steps, 2) representing the curve path.
    """
    N = len(points) - 1
    t = np.linspace(0.0, 1.0, num_steps)
    curve_points = np.zeros((num_steps, 2))

    # Sum of B_n,k(t) * P_k for all control points P_k
    for k in range(N + 1):
        B = bernstein_polynomial(N, k, t)
        curve_points += np.outer(B, points[k])

    return curve_points

def generate_random_closed_bezier_shape(shape_size, n_vertices=N_VERTICES, randomness_scale=0.3):
    """
    Generates the control points for a single, closed, random cubic Bezier shape.

    The shape is constructed from 'n_vertices' connected segments.
    The segments are smoothly connected by using a heuristic to define P1 and P2
    based on the surrounding main vertices.

    Returns:
        tuple: A list of segment control point arrays, and the initial vertices.
    """
    # 1. Generate N main vertices (V0, V1, ...) within the size bounds [0, shape_size-1]
    # These vertices serve as the P0 (start) and P3 (end) points of the segments.
    vertices = np.random.uniform(0, shape_size - 1, size=(n_vertices, 2))
    
    segments_control_points = []
    
    for i in range(n_vertices):
        # Current segment runs from V_i to V_{i+1} (V_i is P0, V_{i+1} is P3)
        V_start = vertices[i]
        V_end = vertices[(i + 1) % n_vertices] # Modulo ensures the shape closes back to V0

        # Heuristic for Control Points (P1 and P2) to ensure a relatively smooth curve
        # V_prev is used to calculate the direction/tangent at V_start
        V_prev = vertices[(i - 1) % n_vertices]
        
        # T_start approximates the tangent vector at V_start, scaled by segment length
        T_start = V_end - V_prev 
        length = np.linalg.norm(T_start)
        T_start = T_start / length if length > 0 else np.array([1, 0])

        # Control Point distance: a random factor scaled by the shape size and a base scale
        # This makes the curves more chaotic/random
        random_dist = shape_size * randomness_scale * np.random.uniform(0.5, 1.5)
        
        # P1 (Control point 'out' from V_start): Placed along the approximated tangent T_start
        P1 = V_start + T_start * random_dist
        
        # P2 (Control point 'in' to V_end): 
        # For P2, we calculate the control point relative to V_end. For a smooth closed shape, 
        # P2 should be aligned with P3' of the previous segment. A simpler way is to 
        # use the tangent between the current V_end and the next V_end.
        V_next = vertices[(i + 2) % n_vertices]
        T_end = V_next - V_start # Direction from start of segment i to V_next
        
        length_end = np.linalg.norm(T_end)
        T_end = T_end / length_end if length_end > 0 else np.array([1, 0])

        P2 = V_end - T_end * random_dist * np.random.uniform(0.5, 1.5)

        # Clip all control points to the bounds [0, shape_size - 1]
        V_start = np.clip(V_start, 0, shape_size - 1)
        P1 = np.clip(P1, 0, shape_size - 1)
        P2 = np.clip(P2, 0, shape_size - 1)
        V_end = np.clip(V_end, 0, shape_size - 1)
        
        # Store the segment control points (P0, P1, P2, P3) as a single array
        segment_array = np.array([V_start, P1, P2, V_end])
        segments_control_points.append(segment_array)

    return segments_control_points, vertices


def render_filled_shape(segments_control_points, shape_size):
    """
    Converts the Bezier curve segments into a filled binary mask (NumPy array).
    
    This function first plots the outline points, then uses a simplified 
    scanline fill algorithm to fill the area between the leftmost and 
    rightmost boundary points on each row.
    
    Args:
        segments_control_points (list): List of (4, 2) arrays defining the shape segments.
        shape_size (int): The dimension of the square mask (e.g., 38 for 38x38).
        
    Returns:
        np.ndarray: A (shape_size, shape_size) NumPy array with 1s in the filled shape.
    """
    # 1. Generate a dense path of points for the outline
    all_path_points = []
    for segment in segments_control_points:
        curve = bezier_curve_points(segment)
        all_path_points.append(curve)
        
    full_path = np.vstack(all_path_points)
    
    # 2. Convert floating-point coordinates to integer grid indices (pixels)
    x_indices = np.clip(np.round(full_path[:, 0]).astype(int), 0, shape_size - 1)
    y_indices = np.clip(np.round(full_path[:, 1]).astype(int), 0, shape_size - 1)
    
    # 3. Create the initial mask with only the boundary marked
    mask = np.zeros((shape_size, shape_size), dtype=np.uint8)
    # Mark the grid points for the shape outline (y, x order)
    mask[y_indices, x_indices] = 1 
    
    # 4. Perform the Scanline Fill
    for y in range(shape_size):
        # Find all x-coordinates where the outline intersects the current row (scanline y)
        x_intersections = np.where(mask[y] == 1)[0]
        
        if len(x_intersections) > 0:
            # Find the leftmost (min x) and rightmost (max x) points of the shape on this row
            start_x = np.min(x_intersections)
            end_x = np.max(x_intersections)
            
            # Fill all pixels between the start and end point (inclusive)
            # This is the "fill" operation to make the shape solid
            mask[y, start_x:end_x + 1] = 1
            
    return mask


def generate_n_random_bezier_shapes(n_shapes, shape_size=SHAPE_SIZE):
    """
    Generates a list of N random Bezier shapes, each rasterized as a 38x38 mask.
    
    Args:
        n_shapes (int): The number of shapes to generate.
        shape_size (int): The dimension of the square space (e.g., 38 for 38x38).

    Returns:
        np.ndarray: An array of shape (N, shape_size, shape_size) containing the binary masks.
    """
    all_masks = []
    print(f"Generating {n_shapes} random closed Bezier shapes of size {shape_size}x{shape_size}...")
    
    for k in tqdm(range(n_shapes)):
        segments_control_points, _ = generate_random_closed_bezier_shape(shape_size)
        # Use the updated function to get a filled shape mask
        shape_mask = render_filled_shape(segments_control_points, shape_size)
        all_masks.append(shape_mask)
        
    # Stack the masks into a single (N, 38, 38) array
    return np.array(all_masks)

def generate_circles(num_maps=1000, size=SHAPE_SIZE):
    radius = size // 2
    center = (size - 1) / 2

    y, x = np.ogrid[:size, :size]
    dist_from_center = (x - center)**2 + (y - center)**2
    mask = dist_from_center <= radius**2

    single_circle = np.zeros((size, size), dtype=np.uint8)
    single_circle[mask] = 1

    return np.repeat(single_circle[np.newaxis, :, :], num_maps, axis=0)
    
# --- Main Execution ---

if __name__ == '__main__':
    # this is for PBS script
    # try:
    #     if len(sys.argv) >= 4:
    #         N = int(sys.argv[1])
    #         na = str(sys.argv[2])
    #         out_dir = str(sys.argv[3])
    #     else:
    #         N = 3
    #         na = "default"
    #         out_dir = "."
    # except ValueError:
    #     print("Invalid input. Defaulting to N=3.")
    #     N = 3
    #     na = "default"
    #     out_dir = "."

    # # Create directory if it doesn't exist
    # os.makedirs(out_dir, exist_ok=True)

    # shapes_data_masks = generate_n_random_bezier_shapes(N, SHAPE_SIZE)
    # shapes_circle = generate_circles(num_maps=int(1*N/100), size=SHAPE_SIZE)

    # shapes_all = np.copy(shapes_data_masks)
    # shapes_data_masks = np.copy(shapes_all)

    # if shapes_data_masks.size > 0:
    #     save_path = os.path.join(out_dir, f"{na}.npy")
    #     np.save(save_path, shapes_data_masks)

    #     print(f"Saved file to: {save_path}")
    # else:
    #     print("No shapes were generated.")
    # Set a default value for N (number of shapes) if not provided by the user
    try:
        if len(sys.argv) > 2:
            N = int(sys.argv[1])
            na=str(sys.argv[2])
        else:
            N = 3 # Default to 3 shapes for demonstration
    except ValueError:
        print("Invalid number of shapes provided. Defaulting to 3.")
        N = 3

    # Generate the rasterized masks for all shapes
    shapes_data_masks = generate_n_random_bezier_shapes(N, SHAPE_SIZE)
    shapes_circle = generate_circles(num_maps=int(1*N/100), size=SHAPE_SIZE)
    
    #shapes_all = np.concatenate((shapes_data_masks,shapes_circle))
    shapes_all = np.copy(shapes_data_masks)
    shapes_data_masks = np.copy(shapes_all)
    if shapes_data_masks.size > 0:
        # Save the rasterized shapes to a file
        #np.save(f'OM10/{na}.npy', shapes_data_masks)
        #np.save(f'{na}.npy', shapes_data_masks)
        np.save(f'/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/OM10/{na}.npy', shapes_data_masks)
        print("\n--- Output Summary (Filled Shapes) ---")
        print(f"Successfully generated and rasterized {N} FILLED shapes.")
        print(f"Saved NumPy array 'randomshapes.npy' with shape: {shapes_data_masks.shape}")
        print("This array now contains the binary masks of the SOLID Bezier shapes.")
        print("\nExample of the first 10 rows of the first shape's mask (1s now represent the solid area):")
        # Display a small section of the mask for demonstration
        print(shapes_data_masks[0][:10])
    else:
        print("No shapes were generated.")
        
    print("\n----------------------------------")