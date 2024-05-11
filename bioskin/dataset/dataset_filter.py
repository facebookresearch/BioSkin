# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import matplotlib
from sklearn.utils.fixes import parse_version
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import torch
import argparse
import json
from scipy.spatial import KDTree
from scipy.stats import qmc
from scipy.spatial import ConvexHull, Delaunay
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
import bioskin.spectrum.color_spectrum as color_spectrum
import bioskin.dataset.dataset_bin2pt as b2pt


if parse_version(matplotlib.__version__) >= parse_version('2.1'):
    density_param = {'density': True}
else:
    density_param = {'normed': True}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Filtering the ranges')
    # modes
    parser.add_argument('--folder',  nargs='?',
                        type=str, default='../datasets/',
                        help='Dataset Folder')
    parser.add_argument('--dataset', nargs='?',
                        type=str, default='dataset_wide_thickness_wide_spectrum_380_1000nm_2nm_1M_photons',
                        help='Dataset Name')
    parser.add_argument("--ranges",
                        nargs=10,
                        metavar=('Mmin', 'Mmax', 'Bmin', 'Bmax', 'Tmin', 'Tmax', 'Emin', 'Emax', 'Omin', 'Omax'),
                        help="input parameters as floats (min, max) for MBTEO",
                        type=float,
                        default=None)

    args = parser.parse_args()
    return args


def stringtify(lower_limits, upper_limits):

    pstring = '_m[' + "{:.3f}".format(lower_limits[0]) + '-' + "{:.3f}".format(upper_limits[0]) + ']'
    pstring += '_b[' + "{:.3f}".format(lower_limits[1]) + '-' + "{:.3f}".format(upper_limits[1]) + ']'
    pstring += '_t[' + "{:.3f}".format(lower_limits[2]) + '-' + "{:.3f}".format(upper_limits[2]) + ']'
    pstring += '_r[' + "{:.3f}".format(lower_limits[3]) + '-' + "{:.3f}".format(upper_limits[3]) + ']'
    pstring += '_o[' + "{:.3f}".format(lower_limits[4]) + '-' + "{:.3f}".format(upper_limits[4]) + ']'
    return pstring


def process_arguments(args):

    if not args.ranges:
        ## parameter ranges (melanin, haemoglobin, epi_thickness, mel_ratio, oxygenation):
        # regular skin ranges:
        args.lower_limits = torch.tensor([0.001, 0.001,   0.001, 0.7, 0.9])
        args.upper_limits = torch.tensor([0.43,    0.2,    0.035, 0.78, 1.0])
    else:
        args.lower_limits = torch.tensor(args.ranges[::2])
        args.upper_limits = torch.tensor(args.ranges[1::2])

    # input
    args.params_train_name = args.folder + args.dataset + '/train/tensors/' + args.dataset + '_params_train.pt'
    args.params_test_name = args.folder + args.dataset + '/test/tensors/' + args.dataset + '_params_test.pt'
    args.spectrums_train_name = args.folder + args.dataset + '/train/tensors/' + args.dataset + '_spec_train.pt'
    args.spectrums_test_name = args.folder + args.dataset + '/test/tensors/' + args.dataset + '_spec_test.pt'
    args.rgb_train_name = args.folder + args.dataset + '/train/tensors/' + args.dataset + '_rgb_train.pt'
    args.rgb_test_name = args.folder + args.dataset + '/test/tensors/' + args.dataset + '_rgb_test.pt'

    #output
    params_string = stringtify(args.lower_limits, args.upper_limits)
    out_dataset_name = args.dataset + params_string
    path_train = args.folder + out_dataset_name + '/train/tensors/'
    path_test = args.folder + out_dataset_name + '/test/tensors/'
    if not os.path.exists(path_train):
        os.makedirs(path_train)
    if not os.path.exists(path_test):
        os.makedirs(path_test)

    args.train_name_out = path_train + out_dataset_name
    args.test_name_out = path_test + out_dataset_name

    dataset_specs_json = args.folder + args.dataset + '/' + args.dataset + '.json'
    print('Loading Dataset Specs from ' + str(dataset_specs_json))
    with open(dataset_specs_json) as f:
        dataset_specs = json.load(f)
        print(dataset_specs)
        with open(args.folder + out_dataset_name + '/' + out_dataset_name + '.json', "w") as f_out:
            json.dump(dataset_specs, f_out)
            f_out.close()
        f.close()

    return args


def filter_skin_only(spectrum_tensor, rgb_tensor, parameters_tensor, lower_limits, upper_limits, wavelength_begin=380,
                     wavelength_end=1000):

    print(torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device) + "(" + str(torch.cuda.device_count()) + ")")

    num_spectrums = spectrum_tensor.size()[0]
    size_spectrum = spectrum_tensor.size()[1]
    num_tuples = parameters_tensor.size()[0]
    if num_spectrums != num_tuples:
        print('ERROR: number of parameter tuples and spectrums is different!')
        exit(0)
    parameter_tuple_size = parameters_tensor.size()[1]

    lower_limits_size = int(lower_limits.shape[0])

    if lower_limits_size != parameter_tuple_size:
        print('ERROR: Check out lower and upper limits for the masking')
        exit(0)

    lower_limits = lower_limits.expand(num_spectrums, lower_limits_size)
    upper_limits = upper_limits.expand(num_spectrums, lower_limits_size)

    mask_lower = parameters_tensor.ge(lower_limits)
    mask_upper = parameters_tensor.le(upper_limits)

    mask = torch.logical_and(mask_lower, mask_upper)
    mask_count_trues_per_tuple = mask.sum(1)
    mask_flat_filtered = mask_count_trues_per_tuple.ge(parameter_tuple_size)  # all parameters within range
    mask_flat_filtered = mask_flat_filtered.resize(num_tuples, 1)  # torch.t(mask_flat_filtered)

    mask_filtered_spectrum = mask_flat_filtered.expand(num_spectrums, size_spectrum)
    mask_filtered_rgb = mask_flat_filtered.expand(num_spectrums, 3)
    mask_filtered_parameters = mask_flat_filtered.expand(num_tuples, parameter_tuple_size)

    spectrum_tensor_filtered = torch.masked_select(spectrum_tensor, mask_filtered_spectrum)
    rgb_tensor_filtered = torch.masked_select(rgb_tensor, mask_filtered_rgb)
    parameters_tensor_filtered = torch.masked_select(parameters_tensor, mask_filtered_parameters)

    new_size = int(np.floor(spectrum_tensor_filtered.shape[0] / size_spectrum))
    spectrum_tensor_filtered = spectrum_tensor_filtered.resize(new_size, size_spectrum)

    rgb_channels = 3
    new_size = int(np.floor(rgb_tensor_filtered.shape[0] / rgb_channels))
    rgb_tensor_filtered = rgb_tensor_filtered.resize(new_size, rgb_channels)

    new_size = int(np.floor(parameters_tensor_filtered.shape[0] / parameter_tuple_size))
    parameters_tensor_filtered = parameters_tensor_filtered.resize(new_size, parameter_tuple_size)

    ratio = (780 - wavelength_begin) / (wavelength_end - wavelength_begin)  # assuming VR + IR, not UV
    visible_range_limit = int(ratio * spectrum_tensor_filtered.shape[1])

    # color spectrum functions
    xbarp, ybarp, zbarp = color_spectrum.create_xyz(visible_range_limit)
    matrix_xyz_to_lrgb = color_spectrum.create_matrix_xyz_to_lrgb("D65prime")

    xbarp = xbarp.to(device)
    ybarp = ybarp.to(device)
    zbarp = zbarp.to(device)
    matrix_xyz_to_lrgb = matrix_xyz_to_lrgb.to(device)

    rgb_tensor_filtered = color_spectrum.spectrum_to_rgb(
        torch.tensor(spectrum_tensor_filtered[:, :visible_range_limit], device='cuda'), xbarp, ybarp, zbarp,
        matrix_xyz_to_lrgb)

    return spectrum_tensor_filtered, rgb_tensor_filtered, parameters_tensor_filtered, \
        mask_filtered_spectrum, mask_filtered_rgb, mask_filtered_parameters



def filter_and_export(lower_limits, upper_limits, spectrum, rgb, params, name_out, suffix):

    spec_f, rgb_f, params_f, spec_mask, rgb_mask, params_mask = \
        filter_skin_only(spectrum, rgb, params, lower_limits, upper_limits)

    torch.save(spec_f, name_out + '_' + suffix + '.pt')
    torch.save(params_f, name_out + '_' + suffix + '.pt')
    b2pt.save_tensor_to_exr(rgb_f, name_out + '_' + suffix + '.exr', bgr=True)
    b2pt.save_tensor_to_exr(rgb_mask, name_out + '_' + suffix + '_mask.exr', bgr=True)
    b2pt.save_tensor_to_text(spec_f, name_out + '_' + suffix + '.txt')
    b2pt.save_tensor_to_text(spec_f, name_out + '_' +  suffix + '.csv')
    return 0


def is_inside_convex_hull(hull, point):
    return all((np.dot(eq[:-1], point) + eq[-1] <= 0) for eq in hull.equations)


def sample_points_in_hull(hull, n_samples=1000):
    min_x, min_y, min_z = np.min(hull.points, axis=0)
    max_x, max_y, max_z = np.max(hull.points, axis=0)

    sampled_points = []
    while len(sampled_points) < n_samples:
        random_point = np.array([np.random.uniform(min_x, max_x),
                                 np.random.uniform(min_y, max_y),
                                 np.random.uniform(min_z, max_z)])
        if is_inside_convex_hull(hull, random_point):
            sampled_points.append(random_point)

    return np.array(sampled_points)


def triangle_area(vertices):
    v0, v1, v2 = vertices
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


def sample_points_on_surface(hull, n_surface_samples=100):
    surface_points = []
    total_area = sum(triangle_area(hull.points[simplex]) for simplex in hull.simplices)
    for simplex in hull.simplices:
        vertices = hull.points[simplex]
        area = triangle_area(vertices)
        n_samples = n_surface_samples
        samples_for_this_triangle = int(n_samples * (area / total_area))

        # Sample points on the triangle
        for _ in range(samples_for_this_triangle):
            r1, r2 = np.random.rand(2)
            sqrt_r1 = np.sqrt(r1)
            point = (1 - sqrt_r1) * vertices[0] + (sqrt_r1 * (1 - r2)) * vertices[1] + (r2 * sqrt_r1) * vertices[2]
            surface_points.append(point)

    return np.array(surface_points)


def sample_points_on_hull_vertices(hull, n_samples):
    surface_points = []
    for simplex in hull.simplices:
        vertices = hull.points[simplex]
        for vertex in vertices:
            surface_points.append(vertex)

    surface_points = select_representative_points(surface_points, n_samples)
    return np.array(surface_points)


def sample_uniform_in_color_space(points, n_vertex_samples=1000, n_surface_samples=1000, n_volume_samples=1000):
    hull = ConvexHull(points)

    extra_samples = 0
    n_vertex_samples = n_vertex_samples
    vertex_samples = sample_points_on_hull_vertices(hull, n_vertex_samples)
    while vertex_samples.shape[0] < n_vertex_samples:
        print('Samples:' + str(vertex_samples.shape[0]))
        vertex_samples = sample_points_on_hull_vertices(hull, n_vertex_samples + extra_samples)
        extra_samples += int(0.2*n_vertex_samples)

    extra_samples = 0
    n_samples_surface = n_surface_samples
    surface_samples = sample_points_on_surface(hull, n_samples_surface)
    while surface_samples.shape[0] < n_surface_samples:
        print('Samples:' + str(surface_samples.shape[0]))
        surface_samples = sample_points_on_surface(hull, n_samples_surface + extra_samples)
        extra_samples += int(0.2*n_surface_samples)

    extra_samples = 0
    n_sample_volume = n_volume_samples
    volume_samples = sample_points_in_hull(hull, n_volume_samples)
    while volume_samples.shape[0] < n_volume_samples:
        print('Samples:' + str(volume_samples.shape[0]))
        volume_samples = sample_points_in_hull(hull, n_sample_volume + extra_samples)
        extra_samples += int(0.2*n_volume_samples)

    sampled_points = np.concatenate((vertex_samples, surface_samples), axis=0)
    sampled_points = np.concatenate((volume_samples, sampled_points), axis=0)
    print("Sampled points: " + str(sampled_points.shape[0]) + " total, "
          + str(vertex_samples.shape[0]) + " vertex hull, "
          + str(surface_samples.shape[0]) + " surface, " + str(volume_samples.shape[0]) + " volume")

    return sampled_points, hull


def filter_subset_samples(pcd1, pcd2, threshold):
    result = []
    for point1 in pcd1:
        distances = np.sqrt(np.sum((pcd2 - point1)**2, axis=1))
        if not np.any(distances <= threshold):
            result.append(point1)
    return np.array(result)


def filter_points(points, cloud, threshold):
    tree = KDTree(cloud)
    distances, _ = tree.query(points)
    return points[distances <= threshold]


def sample_hull_with_inward_offset(points, offset, num_samples):
    # Compute the convex hull
    hull = ConvexHull(points)
    # Compute the areas and normal vectors of the faces
    areas = []
    normals = []
    for simplex in hull.simplices:
        # Get the points of the simplex
        p0, p1, p2 = points[simplex]
        # Compute two vectors of the simplex
        v1 = p1 - p0
        v2 = p2 - p0
        # Compute the normal vector and area of the simplex
        normal = np.cross(v1, v2)
        area = np.linalg.norm(normal) / 2
        normal = normal / np.linalg.norm(normal)
        areas.append(area)
        normals.append(normal)
    # Normalize the areas to get a probability distribution
    areas = np.array(areas)
    probabilities = areas / np.sum(areas)
    # Draw samples
    samples = []
    for _ in range(num_samples):
        # Sample a face from this distribution
        face_index = np.random.choice(len(areas), p=probabilities)
        # Compute the centroid of the sampled face
        simplex = hull.simplices[face_index]
        centroid = np.mean(points[simplex], axis=0)
        # Offset the centroid slightly inwards in the reversed normal direction
        normal = normals[face_index]
        point_inwards = centroid - offset * normal
        samples.append(point_inwards)
    samples = np.array(samples)
    samples = np.clip(samples, 0.0, 1.0)
    return np.array(samples), hull


def sample_points_inwards(points, offset_ratio, num_samples):
    # Compute the convex hull
    hull = ConvexHull(points)
    # Compute the centroid of the convex hull
    hull_centroid = np.mean(points[hull.vertices], axis=0)
    # Compute the areas of the faces
    areas = []
    for simplex in hull.simplices:
        # Get the points of the simplex
        p0, p1, p2 = points[simplex]
        # Compute two vectors of the simplex
        v1 = p1 - p0
        v2 = p2 - p0
        # Compute the area of the simplex
        area = np.linalg.norm(np.cross(v1, v2)) / 2
        areas.append(area)
    # Normalize the areas to get a probability distribution
    areas = np.array(areas)
    probabilities = areas / np.sum(areas)
    # Draw samples
    samples = []
    for _ in range(num_samples):
        # Sample a face from this distribution
        face_index = np.random.choice(len(areas), p=probabilities)
        # Compute the centroid of the sampled face
        simplex = hull.simplices[face_index]
        centroid = np.mean(points[simplex], axis=0)
        # Offset the centroid inwards in the direction of the hull centroid
        direction = hull_centroid - centroid
        length = np.linalg.norm(direction)
        direction = direction / length
        point_inwards = centroid + offset_ratio * length * direction
        samples.append(point_inwards)
    # Sample the vertices of the hull
    for vertex in points[hull.vertices]:
        # Offset the vertex inwards in the direction of the hull centroid
        direction = hull_centroid - vertex
        length = np.linalg.norm(direction)
        direction = direction / length
        point_inwards = vertex + offset_ratio * length * direction
        samples.append(point_inwards)
    samples = np.array(samples)
    return samples, hull


def sample_points_evenly(points, numx, numy, numz):
    # Compute the convex hull
    hull = ConvexHull(points)
    # Compute the Delaunay triangulation of the hull points
    delaunay = Delaunay(points[hull.vertices])
    # Compute the range of the hull in each axis
    min_coords = np.min(points[hull.vertices], axis=0)
    max_coords = np.max(points[hull.vertices], axis=0)
    # Create a 3D grid that spans the range of the hull
    x = np.linspace(min_coords[0], max_coords[0], numx)
    y = np.linspace(min_coords[1], max_coords[1], numy)
    z = np.linspace(min_coords[2], max_coords[2], numz)
    grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1,3)
    # Check for each point in the grid if it is inside the convex hull
    inside = delaunay.find_simplex(grid) >= 0
    samples = grid[inside]
    return samples, hull


def sample_points_from_centroid(points, num_samples, offset_ratio):
    # Compute the convex hull
    hull = ConvexHull(points)
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)
    # Generate random directions uniformly distributed over the sphere
    directions = np.random.normal(size=(num_samples, 3))
    directions = normalize(directions)
    # Initialize an array to store the samples
    samples = []
    # For each direction, find the intersection with the hull and offset it towards the centroid
    for direction in directions:
        # Find the intersection of the ray with each face of the hull
        intersections = []
        for simplex in hull.simplices:
            # Get the points of the simplex
            p0, p1, p2 = points[simplex]
            # Compute two vectors of the simplex
            v1 = p1 - p0
            v2 = p2 - p0
            # Compute the normal vector of the simplex
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            # If the direction is not parallel to the simplex, compute the intersection
            if np.dot(normal, direction) > 0:
                t = np.dot(normal, p0 - centroid) / np.dot(normal, direction)
                intersection = centroid + t * direction
                # Check if the intersection is inside the simplex
                if np.all(np.cross(p1 - intersection, p2 - intersection) == np.cross(p1 - p0, p2 - p0)):
                    intersections.append(intersection)
        # If there are intersections, choose the closest one to the centroid
        if intersections:
            intersection = min(intersections, key=lambda point: euclidean(point, centroid))
            # Offset the intersection towards the centroid
            vector = centroid - intersection
            length = np.linalg.norm(vector)
            vector = vector / length
            sample = intersection + offset_ratio * length * vector
            samples.append(sample)
    # Add the vertices of the hull to the samples and apply the same kind of offset
    # for vertex in points[hull.vertices]:
    #     # Offset the vertex towards the centroid
    #     vector = centroid - vertex
    #     length = np.linalg.norm(vector)
    #     vector = vector / length
    #     sample = vertex + offset_ratio * length * vector
    #     samples.append(sample)
    return np.array(samples), hull


def sample_points_from_surface(points, num_samples, offset_ratio):
    # Compute the convex hull
    hull = ConvexHull(points)
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)
    # Compute the areas of the faces
    areas = []
    for simplex in hull.simplices:
        # Get the points of the simplex
        p0, p1, p2 = points[simplex]
        # Compute two vectors of the simplex
        v1 = p1 - p0
        v2 = p2 - p0
        # Compute the area of the simplex
        area = np.linalg.norm(np.cross(v1, v2)) / 2
        areas.append(area)
    # Normalize the areas to get a probability distribution
    areas = np.array(areas)
    probabilities = areas / np.sum(areas)
    # Draw samples
    samples = []
    for _ in range(num_samples):
        # Sample a face from this distribution
        face_index = np.random.choice(len(areas), p=probabilities)
        # Sample a point from the face
        simplex = hull.simplices[face_index]
        weights = np.random.dirichlet((1, 1, 1))
        sample = weights @ points[simplex]
        # Offset the sample towards the centroid
        vector = centroid - sample
        length = np.linalg.norm(vector)
        vector = vector / length
        sample = sample + offset_ratio * length * vector
        samples.append(sample)
    return np.array(samples), hull


def sample_points_from_surface_discard_subset(points, points2, num_samples, offset_ratio, dilation_ratio):
    # Compute the convex hull
    hull = ConvexHull(points)
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)
    # Compute the dilated hull of points2
    centroid2 = np.mean(points2, axis=0)
    points2_dilated = points2 + dilation_ratio * (points2 - centroid2)
    delaunay2 = Delaunay(points2_dilated)
    # Compute the minimum coordinates of points2
    min_coords2 = np.min(points2, axis=0)
    # Compute the areas of the faces
    areas = []
    for simplex in hull.simplices:
        # Get the points of the simplex
        p0, p1, p2 = points[simplex]
        # Compute two vectors of the simplex
        v1 = p1 - p0
        v2 = p2 - p0
        # Compute the area of the simplex
        area = np.linalg.norm(np.cross(v1, v2)) / 2
        areas.append(area)
    # Normalize the areas to get a probability distribution
    areas = np.array(areas)
    probabilities = areas / np.sum(areas)
    # Draw samples
    samples = []
    for _ in range(num_samples):
        # Sample a face from this distribution
        face_index = np.random.choice(len(areas), p=probabilities)
        # Sample a point from the face
        simplex = hull.simplices[face_index]
        weights = np.random.dirichlet((1, 1, 1))
        sample = weights @ points[simplex]
        # Offset the sample towards the centroid
        vector = centroid - sample
        length = np.linalg.norm(vector)
        vector = vector / length
        sample = sample + offset_ratio * length * vector
        # Discard the sample if it falls into the dilated hull of points2
        if delaunay2.find_simplex(sample) == -1 and np.all(sample >= min_coords2):
            samples.append(sample)
    return np.array(samples), hull


def sample_points_close_to_surface_with_offset(points, num_samples, num_grid_points_per_axis, offset_ratio):
    # Compute the convex hull
    hull = ConvexHull(points)
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)
    # Compute the Delaunay triangulation of the hull points
    delaunay = Delaunay(points[hull.vertices])
    # Compute the range of the hull in each axis
    min_coords = np.min(points[hull.vertices], axis=0)
    max_coords = np.max(points[hull.vertices], axis=0)
    # Create a dense 3D grid that spans the range of the hull
    x = np.linspace(min_coords[0], max_coords[0], num_grid_points_per_axis)
    y = np.linspace(min_coords[1], max_coords[1], num_grid_points_per_axis)
    z = np.linspace(min_coords[2], max_coords[2], num_grid_points_per_axis)
    grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1,3)
    # Check for each point in the grid if it is inside the convex hull
    inside = delaunay.find_simplex(grid) >= 0
    mask = grid[inside]
    # For each point in the mask, compute its distance to the surface of the hull
    distances = np.array([np.min(np.linalg.norm(mask - point)) for point in points[hull.simplices].reshape(-1, 3)])
    # Sort the points in the mask by their XYZ coordinates
    mask_sorted = mask[np.lexsort(mask.T[::-1])]
    # Select the num_samples points that are more evenly distributed
    step = len(mask_sorted) // num_samples
    samples = mask_sorted[::step][:num_samples]
    # Offset the samples towards the centroid
    for i in range(len(samples)):
        vector = centroid - samples[i]
        length = np.linalg.norm(vector)
        vector = vector / length
        samples[i] = samples[i] + offset_ratio * vector
    return samples, hull


def select_representative_points(points, num_samples):
    kmeans = KMeans(n_clusters=num_samples)
    kmeans.fit(points)
    selected_points = kmeans.cluster_centers_
    return selected_points


def select_extreme_points(points, num_samples_per_axis):
    centroid = np.mean(points, axis=0)
    vectors = points - centroid
    selected_points = []
    for axis in range(points.shape[1]):
        sorted_indices = np.argsort(vectors[:, axis])
        selected_points.extend(points[sorted_indices[:num_samples_per_axis]])  # Negative direction
        selected_points.extend(points[sorted_indices[-num_samples_per_axis:]])  # Positive direction
    return np.array(selected_points)


def select_extreme_and_evenly_distributed_points(points, num_extreme_points_per_axis, num_clusters):
    centroid = np.mean(points, axis=0)
    vectors = points - centroid
    selected_points = []
    for axis in range(points.shape[1]):
        sorted_indices = np.argsort(vectors[:, axis])
        selected_points.extend(points[sorted_indices[:num_extreme_points_per_axis]])  # Negative direction
        selected_points.extend(points[sorted_indices[-num_extreme_points_per_axis:]])  # Positive direction
    # Use KMeans to partition the remaining points into clusters
    remaining_points = np.delete(points, sorted_indices[:num_extreme_points_per_axis], axis=0)
    remaining_points = np.delete(remaining_points, sorted_indices[-num_extreme_points_per_axis:], axis=0)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(remaining_points)
    selected_points.extend(kmeans.cluster_centers_)
    return np.array(selected_points)


def select_extreme_and_evenly_distributed_points2(points, num_extreme_points_per_axis, num_clusters, threshold):
    centroid = np.mean(points, axis=0)
    vectors = points - centroid
    selected_points = []
    for axis in range(points.shape[1]):
        sorted_indices = np.argsort(vectors[:, axis])
        selected_points.extend(points[sorted_indices[:num_extreme_points_per_axis]])  # Negative direction
        selected_points.extend(points[sorted_indices[-num_extreme_points_per_axis:]])  # Positive direction
    # Use KMeans to partition the remaining points into clusters
    remaining_points = np.delete(points, sorted_indices[:num_extreme_points_per_axis], axis=0)
    remaining_points = np.delete(remaining_points, sorted_indices[-num_extreme_points_per_axis:], axis=0)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(remaining_points)
    selected_points.extend(kmeans.cluster_centers_)
    selected_points = np.array(selected_points)
    distances = cdist(selected_points, selected_points)
    np.fill_diagonal(distances, np.inf)
    mask = np.all(distances >= threshold, axis=1)
    selected_points = selected_points[mask]
    return selected_points


def calculate_luminance(colors, rgb=False):
    if rgb:
        return 0.299*colors[:, 0] + 0.587*colors[:, 1] + 0.114*colors[:, 2]
    else:
        return colors[:, 0]


def sort_by_luminance(vertices):
    luminance = calculate_luminance(vertices)
    sorted_indices = np.argsort(luminance)
    sorted_vertices = vertices[sorted_indices]
    return sorted_vertices


def get_evenly_distributed_vertices(vertices, num_vertices):
    sorted_vertices = sort_by_luminance(vertices)
    step = len(sorted_vertices) // num_vertices
    selected_vertices = sorted_vertices[::step][:num_vertices]
    return selected_vertices


def select_qmc_closest(skin_tones, num_samples, device, max_error=0.01):
    sampler = qmc.Halton(d=3, scramble=True)
    potential_skin_tones = torch.tensor(sampler.random(n=num_samples), device=device)
    distances = torch.cdist(potential_skin_tones, skin_tones)
    mask = distances < max_error
    selected_skin_tones = skin_tones[mask]
    return  selected_skin_tones


def select_qmc_closest2(skin_tones, num_samples, device, max_error=0.01):
    sampler = qmc.Halton(d=3, scramble=True)
    potential_skin_tones = torch.tensor(sampler.random(n=num_samples), device=device)
    distances = torch.cdist(potential_skin_tones, skin_tones)
    closest_indices = torch.argmin(distances, dim=1)
    # Select the closest points in skin_tones
    closest_points = skin_tones[closest_indices]
    # Compute the distances to the closest points
    closest_distances = distances[torch.arange(distances.shape[0]), closest_indices]
    # Create a mask for the points where the distance is less than 0.01
    mask = closest_distances < max_error
    result = closest_points[mask]
    return result


def run(args):
    spectrums_train = torch.load(args.spectrums_train_name)
    rgb_train = torch.load(args.rgb_train_name)
    parameters_train = torch.load(args.params_train_name)

    spectrums_test = torch.load(args.spectrums_test_name)
    rgb_test = torch.load(args.rgb_test_name)
    parameters_test = torch.load(args.params_test_name)

    # train is 480k qMC random, test is 120k random uniform
    filter_and_export(args.lower_limits, args.upper_limits, spectrums_train, rgb_train, parameters_train,
                      args.train_name_out, 'train')
    filter_and_export(args.lower_limits, args.upper_limits, spectrums_test, rgb_test, parameters_test,
                      args.test_name_out, 'test')


if __name__ == '__main__':
    args = parse_arguments()
    args = process_arguments(args)
    run(args)
    print("Finished Filtering Tensors!")