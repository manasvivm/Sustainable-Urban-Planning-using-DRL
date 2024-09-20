import pickle
import numpy as np

pickle_file = 'urban_planning/cfg/test_data/real/hlg/init_plan_hlg copy.pickle'

with open(pickle_file, 'rb') as file:
    data = pickle.load(file)

gdf = data['gdf']

#print(gdf)

gdf_points = gdf[gdf['geometry'].geom_type == 'Point']
print(gdf_points)
# random_densities = np.random.randint(0, 101, size=len(gdf_points))
# gdf_points['density'] = random_densities
# gdf.loc[gdf['geometry'].geom_type == 'Point', 'density'] = random_densities
# print(gdf['density'].unique())

# data['gdf'] = gdf

# with open(pickle_file, 'wb') as file:
#     pickle.dump(data, file)

# print(f"Updated pickle file saved back to: {pickle_file}")

#unique_geometry_types = gdf.geom_type.unique()
# Output the unique geometry types
#print("Unique geometry types in the GeoDataFrame:", unique_geometry_types)
#print("Before modification:", gdf['type'].unique())
#print(gdf)


# gdf['type'] = gdf['type'].replace(13, 14)

# print("After modification:", gdf['type'].unique())

# data['gdf'] = gdf
# with open(pickle_file, 'wb') as file:
#     pickle.dump(data, file)

# print("Data saved back to pickle file.")
