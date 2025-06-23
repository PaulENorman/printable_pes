import numpy as np
from stl import mesh # Using numpy-stl library

# --- Condense the potential energy surface down into one function ---
def calculate_leps_pes(rab, rbc):
    """
    Calculates the LEPS potential energy for a given F-H (rab) and H-H (rbc)
    bond distance, assuming a linear F-H-H configuration (rac = rab + rbc).

    Args:
        rab (float or np.ndarray): Distance between atom A (F) and atom B (H).
        rbc (float or np.ndarray): Distance between atom B (H) and atom C (H).

    Returns:
        float or np.ndarray: The potential energy in kcal/mol.
    """
    # Define parameters
    params = {
        "DFH": 591.1,  # kcal/mol
        "DHH": 458.2,  # kcal/mol
        "betaFH": 2.2189,  # 1/Angstrom
        "betaHH": 1.9420,  # 1/Angstrom
        "r0FH": 0.917,  # Angstrom
        "r0HH": 0.7419,  # Angstrom
        "SFH": 0.167,
        "SHH": 0.106
    }

    # Helper function Q
    def Q(r, d, beta, ro):
        return 0.5 * d * (1.5 * np.exp(-2 * beta * (r - ro)) - np.exp(-beta * (r - ro)))

    # Helper function J
    def J(r, d, beta, ro):
        return 0.5 * d * (0.5 * np.exp(-2 * beta * (r - ro)) - 3 * np.exp(-beta * (r - ro)))

    # LEPS Potential Core Calculation
    rac = rab + rbc

    den_ab = 1 + params["SFH"]
    den_ac = 1 + params["SFH"] # F-H bond
    den_bc = 1 + params["SHH"]

    Qab_term = Q(rab, params["DFH"], params["betaFH"], params["r0FH"]) / den_ab
    Qbc_term = Q(rbc, params["DHH"], params["betaHH"], params["r0HH"]) / den_bc
    Qac_term = Q(rac, params["DFH"], params["betaFH"], params["r0FH"]) / den_ac

    Jab_term = J(rab, params["DFH"], params["betaFH"], params["r0FH"]) / den_ab
    Jbc_term = J(rbc, params["DHH"], params["betaHH"], params["r0HH"]) / den_bc
    Jac_term = J(rac, params["DFH"], params["betaFH"], params["r0FH"]) / den_ac

    sum_Q = Qab_term + Qbc_term + Qac_term
    sq_diff_1 = (Jab_term - Jbc_term)**2
    sq_diff_2 = (Jbc_term - Jac_term)**2
    sq_diff_3 = (Jac_term - Jab_term)**2
    sqrt_term = np.sqrt(0.5 * (sq_diff_1 + sq_diff_2 + sq_diff_3))

    return sum_Q - sqrt_term

# --- Function to generate a 3D printable STL from an arbitrary function ---
def generate_3d_printable_stl(
    func,
    x_min_orig, x_max_orig,
    y_min_orig, y_max_orig,
    x_scale, y_scale, z_scale,
    z_limit_orig_units, # Z values above this limit will be clipped
    mesh_resolution,
    floor_thickness,
    output_filename="surface.stl"
):
    """
    Generates a 3D printable STL file for an arbitrary function of two variables,
    including a base floor and vertical side walls, ensuring watertightness and
    consistent normal orientation.

    Args:
        func (callable): The function f(x, y) that returns the height (Z value).
        x_min_orig (float): Minimum value for the x-variable in original units.
        x_max_orig (float): Maximum value for the x-variable in original units.
        y_min_orig (float): Minimum value for the y-variable in original units.
        y_max_orig (float): Maximum value for the y-variable in original units.
        x_scale (float): Scale factor for the x-dimension (e.g., mm per original unit).
        y_scale (float): Scale factor for the y-dimension (e.g., mm per original unit).
        z_scale (float): Scale factor for the function's return value (e.g., mm per function unit).
        z_limit_orig_units (float): Maximum Z value in original units. Z values above this will be clipped.
        mesh_resolution (int): Number of points along each x and y axis for the mesh grid.
        floor_thickness (float): The desired thickness of the base floor in scaled units (e.g., mm).
        output_filename (str): The name of the output STL file.
    """
    # 1. Generate grid points and surface Z values
    x_vals_orig = np.linspace(x_min_orig, x_max_orig, mesh_resolution)
    y_vals_orig = np.linspace(y_min_orig, y_max_orig, mesh_resolution)
    X_orig, Y_orig = np.meshgrid(x_vals_orig, y_vals_orig)

    Z_surface_orig = func(X_orig, Y_orig)
    
    # Clip Z values above the specified limit to prevent extreme spikes
    Z_surface_orig[Z_surface_orig > z_limit_orig_units] = z_limit_orig_units

    # Offset Z values so the lowest point of the surface is at Z=0 (before scaling)
    z_offset_orig = np.min(Z_surface_orig)
    Z_surface_orig = Z_surface_orig - z_offset_orig

    # Apply scaling to all dimensions
    X_scaled = X_orig * x_scale
    Y_scaled = Y_orig * y_scale
    Z_surface_scaled = Z_surface_orig * z_scale
    
    # Determine Z-coordinate for the base of the model (floor bottom)
    floor_z_scaled = -floor_thickness 

    # Initialize a list to hold all triangle faces
    faces = []

    # 2. Generate Top Surface Triangles (normals pointing upwards)
    # Each quad (square) is split into two triangles, ordered CCW from above.
    for i in range(mesh_resolution - 1):
        for j in range(mesh_resolution - 1):
            # Vertices of the current quad (scaled coordinates)
            v0 = [X_scaled[i, j], Y_scaled[i, j], Z_surface_scaled[i, j]]
            v1 = [X_scaled[i + 1, j], Y_scaled[i + 1, j], Z_surface_scaled[i + 1, j]]
            v2 = [X_scaled[i, j + 1], Y_scaled[i, j + 1], Z_surface_scaled[i, j + 1]]
            v3 = [X_scaled[i + 1, j + 1], Y_scaled[i + 1, j + 1], Z_surface_scaled[i + 1, j + 1]]

            # First triangle: (v0, v1, v2) -> Bottom-left, Bottom-right, Top-left
            faces.append([v0, v1, v2])
            # Second triangle: (v1, v3, v2) -> Bottom-right, Top-right, Top-left
            faces.append([v1, v3, v2])

    # 3. Generate Floor Triangles (normals pointing downwards)
    # The floor is now a grid of quads, matching the surface resolution for watertightness.
    for i in range(mesh_resolution - 1):
        for j in range(mesh_resolution - 1):
            # Vertices of the current quad on the floor plane
            f0 = [X_scaled[i, j], Y_scaled[i, j], floor_z_scaled]
            f1 = [X_scaled[i + 1, j], Y_scaled[i + 1, j], floor_z_scaled]
            f2 = [X_scaled[i, j + 1], Y_scaled[i, j + 1], floor_z_scaled]
            f3 = [X_scaled[i + 1, j + 1], Y_scaled[i + 1, j + 1], floor_z_scaled]

            # Triangles for the floor (CW from above for normal to point down/outward)
            # This order ensures normals point into the negative Z direction (down).
            faces.append([f0, f1, f3]) # Bottom-left, Bottom-right, Top-right (CW from above)
            faces.append([f0, f3, f2]) # Bottom-left, Top-right, Top-left (CW from above)

    # 4. Generate Side Walls (normals pointing outwards)
    # Each wall segment connects top surface edge to floor edge.

    # Wall along x_min_orig (left side, normal towards X-neg)
    # Iterates along the Y-axis at the X_min boundary (column 0 of X_scaled)
    for i in range(mesh_resolution - 1): # i corresponds to row index (X-direction)
        # Top edge vertices
        v_top_curr = [X_scaled[i, 0], Y_scaled[i, 0], Z_surface_scaled[i, 0]]
        v_top_next = [X_scaled[i + 1, 0], Y_scaled[i + 1, 0], Z_surface_scaled[i + 1, 0]]
        # Bottom edge vertices
        v_bot_curr = [X_scaled[i, 0], Y_scaled[i, 0], floor_z_scaled]
        v_bot_next = [X_scaled[i + 1, 0], Y_scaled[i + 1, 0], floor_z_scaled]
        
        # Triangles for this wall strip (CCW when viewed from outside, i.e., from X < x_min_scaled)
        faces.append([v_bot_curr, v_bot_next, v_top_curr]) # Bottom-left, Bottom-right, Top-left
        faces.append([v_bot_next, v_top_next, v_top_curr]) # Bottom-right, Top-right, Top-left

    # Wall along x_max_orig (right side, normal towards X-pos)
    # Iterates along the Y-axis at the X_max boundary (last column of X_scaled)
    for i in range(mesh_resolution - 1):
        v_top_curr = [X_scaled[i, mesh_resolution - 1], Y_scaled[i, mesh_resolution - 1], Z_surface_scaled[i, mesh_resolution - 1]]
        v_top_next = [X_scaled[i + 1, mesh_resolution - 1], Y_scaled[i + 1, mesh_resolution - 1], Z_surface_scaled[i + 1, mesh_resolution - 1]]
        v_bot_curr = [X_scaled[i, mesh_resolution - 1], Y_scaled[i, mesh_resolution - 1], floor_z_scaled]
        v_bot_next = [X_scaled[i + 1, mesh_resolution - 1], Y_scaled[i + 1, mesh_resolution - 1], floor_z_scaled]

        # Triangles for this wall strip (CCW when viewed from outside, i.e., from X > x_max_scaled)
        faces.append([v_top_curr, v_top_next, v_bot_curr]) # Top-left, Top-right, Bottom-left
        faces.append([v_top_next, v_bot_next, v_bot_curr]) # Top-right, Bottom-right, Bottom-left

    # Wall along y_min_orig (front side, normal towards Y-neg)
    # Iterates along the X-axis at the Y_min boundary (row 0 of Y_scaled)
    for j in range(mesh_resolution - 1): # j corresponds to column index (Y-direction)
        v_top_curr = [X_scaled[0, j], Y_scaled[0, j], Z_surface_scaled[0, j]]
        v_top_next = [X_scaled[0, j + 1], Y_scaled[0, j + 1], Z_surface_scaled[0, j + 1]]
        v_bot_curr = [X_scaled[0, j], Y_scaled[0, j], floor_z_scaled]
        v_bot_next = [X_scaled[0, j + 1], Y_scaled[0, j + 1], floor_z_scaled]

        # Triangles for this wall strip (CCW when viewed from outside, i.e., from Y < y_min_scaled)
        # Note: Order adjusted for clarity and consistency
        faces.append([v_top_curr, v_bot_curr, v_bot_next]) # Top-left, Bottom-left, Bottom-right
        faces.append([v_top_curr, v_bot_next, v_top_next]) # Top-left, Bottom-right, Top-right

    # Wall along y_max_orig (back side, normal towards Y-pos)
    # Iterates along the X-axis at the Y_max boundary (last row of Y_scaled)
    for j in range(mesh_resolution - 1):
        v_top_curr = [X_scaled[mesh_resolution - 1, j], Y_scaled[mesh_resolution - 1, j], Z_surface_scaled[mesh_resolution - 1, j]]
        v_top_next = [X_scaled[mesh_resolution - 1, j + 1], Y_scaled[mesh_resolution - 1, j + 1], Z_surface_scaled[mesh_resolution - 1, j + 1]]
        v_bot_curr = [X_scaled[mesh_resolution - 1, j], Y_scaled[mesh_resolution - 1, j], floor_z_scaled]
        v_bot_next = [X_scaled[mesh_resolution - 1, j + 1], Y_scaled[mesh_resolution - 1, j + 1], floor_z_scaled]

        # Triangles for this wall strip (CCW when viewed from outside, i.e., from Y > y_max_scaled)
        # Note: Order adjusted for clarity and consistency
        faces.append([v_bot_curr, v_top_next, v_top_curr]) # Bottom-left, Top-right, Top-left
        faces.append([v_bot_curr, v_bot_next, v_top_next]) # Bottom-left, Bottom-right, Top-right

    # Convert list of faces to a NumPy array suitable for numpy-stl
    faces_array = np.array(faces)

    # Create the mesh object from the faces
    surface_mesh = mesh.Mesh(np.zeros(faces_array.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces_array):
        # Assign the vertices of each triangle directly
        # numpy-stl will automatically calculate normals if not explicitly provided
        surface_mesh.vectors[i] = f

    # Write the mesh to an STL file
    surface_mesh.save(output_filename)
    print(f"STL file '{output_filename}' generated successfully with {faces_array.shape[0]} triangles.")


# --- Example Usage for PES ---
if __name__ == "__main__":
    # Original ranges from the PES contour plot
    x_min_pes = 0.4 # r_FH in Angstroms
    x_max_pes = 4.0
    y_min_pes = 0.3 # r_HH in Angstroms
    y_max_pes = 4.0

    # Define scaling factors for 3D printing
    x_scale_factor = 40.0 # mm / Angstrom
    y_scale_factor = 40.0 # mm / Angstrom
    
    # Z-scale: The PES ranges from approx -500 to 100 kcal/mol.
    # After clipping and offsetting, the lowest Z will be 0.
    # If original min was -500, and max is 100, clipped max is 100.
    # After offset by -(-500)=+500, values become 0 to 600.
    # If we want the max height (corresponding to 100 kcal/mol) to be e.g. 30mm from the floor:
    # (100 - (-500)) * z_scale_factor = 30mm  =>  600 * z_scale_factor = 30  => z_scale_factor = 30/600 = 0.05
    z_scale_factor = 0.05 # mm / kcal/mol

    # Maximum Z value in original units. Values above this will be clipped.
    # This prevents extreme spikes from dominating the model's height and can smooth out noisy areas.
    z_limit_orig_units = 100 # kcal/mol (clipping anything above 100 kcal/mol)

    # Resolution of the mesh for the STL file
    resolution = 500 # Number of points along x and y axes. Higher = smoother but larger file.

    # Floor thickness in millimeters
    floor_thickness_mm = 2.0

    output_file_name = "PES_3D_Printable_Watertight.stl" # New filename to distinguish

    print(f"Starting STL generation for PES...")
    print(f"Original ranges: X=[{x_min_pes:.1f},{x_max_pes:.1f}]Å, Y=[{y_min_pes:.1f},{y_max_pes:.1f}]Å")
    print(f"Scale factors: X={x_scale_factor:.1f}x, Y={y_scale_factor:.1f}x, Z={z_scale_factor:.3f}x")
    print(f"Z values clipped at: {z_limit_orig_units} kcal/mol")
    print(f"Mesh resolution: {resolution}x{resolution} grid")
    print(f"Floor thickness: {floor_thickness_mm:.1f} mm")
    print(f"Output file: {output_file_name}")

    generate_3d_printable_stl(
        calculate_leps_pes,
        x_min_pes, x_max_pes,
        y_min_pes, y_max_pes,
        x_scale_factor, y_scale_factor, z_scale_factor,
        z_limit_orig_units, # Correctly pass the z_limit parameter
        resolution,
        floor_thickness_mm,
        output_file_name
    )
    print("STL generation complete for PES.")