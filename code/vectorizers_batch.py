import numpy as np
import matplotlib.pyplot as plt
import gudhi.representations as gdr
import gtda.diagrams as gtd
# from gudhi.representations import Landscape, PersistenceImage
# from gtda.diagrams import PersistenceLandscape, PersistenceImage


DIMENSION_CNT = 3
MAX_DIST = 12.43843407284584  # computed from find_max_dist()


def diagrams_giotto_to_gudhi(diagrams: np.ndarray):
    alt_diagrams = []
    for diag in diagrams:
        alt_diagrams.append(diag[:, [2, 0, 1]])
    return np.array(alt_diagrams)


def remove_diagram_padding(diagram, eps=1e-12):
    diagram = np.asarray(diagram, dtype=float)
    if len(diagram) == 0:
        print("EMPTY INPUT")

    finite_mask = np.isfinite(diagram).all(axis=1)
    persistence_mask = (diagram[:, 1] - diagram[:, 0]) > eps
    final_diagram = diagram[finite_mask & persistence_mask]
    if len(final_diagram) == 0:
        print("EMPTY DIAGRAM")
    else:
        print("NONEMPTY")
    return final_diagram


def diagrams_isolate_dim(diagrams: np.ndarray):
    # return (hom_dim_n) list of diagram array (1000)
    dimension_array = []

    for dim in range(DIMENSION_CNT):
        dim_diagrams = []
        for diagram in diagrams:
            # rows (holes) in diagram with correct dim
            triplets_in_dim = diagram[diagram[:, 2] == dim]
            doubles_in_dim = triplets_in_dim[:, :2]
            final_diagram = remove_diagram_padding(doubles_in_dim)
            dim_diagrams.append(final_diagram)
        dimension_array.append(dim_diagrams)
    return dimension_array


def plot_landscape(
        p_landscape, 
        landscape_list, 
        num_landscapes, 
        resolution,
        plot_index, 
    ):
    landscape_to_plot = landscape_list[plot_index]
    x_values = np.linspace(*p_landscape.sample_range_fixed_, resolution)
    plt.figure(figsize=(8, 5))  

    for i in range(num_landscapes):
        y_values = landscape_to_plot[i * resolution : (i + 1) * resolution]
        plt.plot(x_values, y_values, label=f"Landscape {i+1}")

    plt.title("Persistence Landscape")
    plt.xlabel("Parameter $t$")
    plt.ylabel("$\lambda_k(t)$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def generate_landscape_gudhi(
        diagrams: np.ndarray,  # giotto-tda format (b, d, dim)
        num_landscapes=5, 
        resolution=100,
        plot=False, 
        plot_dim=0,
        plot_index=0
    ):
    diagrams_dims = diagrams_isolate_dim(diagrams)

    # can only handle one homology dimension at a time
    p_landscape = gdr.Landscape(
        num_landscapes=num_landscapes, 
        resolution=resolution, 
        sample_range=[0, MAX_DIST + 1]
    )
    
    all_dimensions = dict()
    for dim in range(DIMENSION_CNT):
        all_dimensions[dim] = p_landscape.fit_transform(diagrams_dims[dim])
        if plot and dim == plot_dim:
            plot_landscape(
                p_landscape, 
                all_dimensions[dim], 
                num_landscapes, 
                resolution,
                plot_index, 
            )
    
    return all_dimensions


def generate_image_gudhi(
        diagrams: np.ndarray, 
        resolution=[20, 20], 
        bandwidth=1.0, 
        # im_range,
        plot=False, 
        plot_dim=0,
        plot_index=0
    ):
    diagrams_dims = diagrams_isolate_dim(diagrams)

    # can only handle one homology dimension at a time
    p_image = gdr.PersistenceImage(
        bandwidth=bandwidth, 
        # weight=<function>,  # diagram weight function, default constant 
        resolution=resolution,
    )
    
    all_dimensions = dict()
    for dim in range(DIMENSION_CNT):
        if dim == 2:
            print(diagrams_dims[dim])
        all_dimensions[dim] = p_image.fit_transform(diagrams_dims[dim])

    if plot:
        image_to_plot = all_dimensions[plot_dim][plot_index]
        img_matrix = image_to_plot.reshape(resolution)

        plt.figure(figsize=(6, 6))
        plt.imshow(img_matrix, cmap='viridis', origin='lower', 
                extent=p_image.im_range, interpolation='nearest')
        plt.title("Persistence Image (Dimension 1)")
        plt.xlabel("Birth")
        plt.ylabel("Death")
        plt.colorbar(label="Pixel Intensity")
        plt.show()

    return all_dimensions


# NOT USED
def generate_landscape_gtda(
        diagrams: np.ndarray, 
        n_layers=5, 
        n_bins=100, 
        plot=False, 
        plot_index=None
    ):
    transformer = gtd.PersistenceLandscape(n_layers=n_layers, n_bins=n_bins)
    transformed = transformer.fit_transform(diagrams)
    
    print("Homology Dimensions:", transformer.homology_dimensions_)
    print("Transformed Shape:", transformed.shape)
    print(transformed)

    if not plot:
        return transformed.flatten()
    
    # plotting
    if plot_index is None:
        fig = transformer.plot(transformed)
    else:
        fig = transformer.plot(transformed, sample=plot_index)
    fig.show()

    return transformed.flatten()


# NOT USED
def generate_image_gtda(
        diagrams: np.ndarray, 
        n_bins=100, 
        sigma=0.1, 
        plot=False, 
        plot_index=None,
        plot_dim=0,
    ):
    transformer = gtd.PersistenceImage(n_bins=n_bins, sigma=sigma)
    transformed = transformer.fit_transform(diagrams)

    print("Homology Dimensions:", transformer.homology_dimensions_)
    print("Transformed Shape:", transformed.shape)

    if not plot:
        return transformed.flatten()
    
    # plotting
    if plot_index is None:
        fig = transformer.plot(transformed)
    else:
        fig = transformer.plot(
            transformed, 
            sample=plot_index,
            homology_dimension_idx=plot_dim,
        )
    fig.show()

    return transformed.flatten()