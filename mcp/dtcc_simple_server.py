# server.py (with bounds as a parameter)
from fastmcp import FastMCP
from datetime import date
from pathlib import Path
import dtcc

# Create an MCP server instance
mcp = FastMCP("MultiToolServer")

@mcp.tool()
def say_hello(name: str) -> str:
    """Returns a greeting."""
    return f"Hello, {name}!"

@mcp.tool()
def get_current_date() -> str:
    """Returns the current date."""
    return date.today().isoformat()

# 1. --- UPDATE THE TOOL SIGNATURE ---
# Add the four bounding box coordinates as parameters with default values.
@mcp.tool()
def build_city_mesh(
    mesh_resolution: float = 5.0,
    min_building_height: float = 2.5,
    xmin: float = 319891.0,
    ymin: float = 6399790.0,
    xmax: float = 321891.0, # 319891 + 2000
    ymax: float = 6401790.0  # 6399790 + 2000
) -> str:
    """
    Builds a 3D city mesh using dtcc-builder and saves it to files.

    :param mesh_resolution: The resolution of the output mesh.
    :param min_building_height: The minimum height of buildings to include.
    :param xmin: The minimum x-coordinate of the bounding box.
    :param ymin: The minimum y-coordinate of the bounding box.
    :param xmax: The maximum x-coordinate of the bounding box.
    :param ymax: The maximum y-coordinate of the bounding box.
    :return: A message indicating success or failure.
    """
    print(f"Server received request to build a city mesh with params: resolution={mesh_resolution}, min_height={min_building_height}, bounds=({xmin},{ymin},{xmax},{ymax})")
    try:
        # 2. --- USE THE PARAMETERS TO CREATE THE BOUNDS ---
        # The hardcoded bounds are replaced by the function arguments.
        bounds = dtcc.Bounds(xmin, ymin, xmax, ymax)

        # The rest of your proven workflow remains the same
        pointcloud = dtcc.download_pointcloud(bounds=bounds)
        buildings = dtcc.download_footprints(bounds=bounds)
        pointcloud = pointcloud.remove_global_outliers(3.0)
        raster = dtcc.build_terrain_raster(pointcloud, cell_size=2, radius=3, ground_only=True)
        buildings = dtcc.extract_roof_points(buildings, pointcloud)
        buildings = dtcc.compute_building_heights(buildings, raster, overwrite=True)
        city = dtcc.City()
        city.add_terrain(raster)
        city.add_buildings(buildings, remove_outside_terrain=True)
        mesh = dtcc.build_city_mesh(city, lod=dtcc.GeometryType.LOD0)
        
        # You might want to save the mesh with a unique name in the future,
        # but for now, this is fine.
        dtcc.io.save_mesh(mesh, "./city_mesh.obj")
        
        return f"Success! City mesh built for bounds ({xmin},{ymin},{xmax},{ymax}) and saved to city_mesh.obj"

    except Exception as e:
        print(f"[DTCC Error] {e}")
        return f"Error building city mesh: {e}"

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)