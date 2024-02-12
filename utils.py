import numpy as np
import cv2 as cv

# funciones auxiliares
def get_neighbors(row, column, labels):
    neighbors = []
    if row > 0: neighbors.append(labels[row - 1][column])
    if column > 0: neighbors.append(labels[row][column - 1])
    return [n for n in neighbors if n > 0]

def update_linked_labels(linked):
    """ Actualizacion de etiquetas para que que 
    todas las etiquetas apunten a la etiqueta raiz """
    for key in linked.keys():
        while linked[key] != linked[linked[key]]:
            linked[key] = linked[linked[key]]


def two_pass_labeling(binary_image):

    # Inicializacion
    linked = {}
    next_label = 1
    rows_q, cols_q = binary_image.shape
    labels = np.zeros(binary_image.shape, dtype=np.uint32)

    # Primera pasada
    for row in range(rows_q):
        for col in range(cols_q):
            if binary_image[row][col] == 255:

                neighbors = get_neighbors(row, col, labels)
                if not neighbors:
                    linked[next_label] = next_label
                    labels[row][col] = next_label
                    next_label += 1
                else:
                    min_label = min(neighbors)
                    labels[row][col] = min_label
                    for n in neighbors:
                        if n != min_label: linked[n] = min_label

    update_linked_labels(linked)

    # Segunda pasada
    for row in range(rows_q):
        for col in range(cols_q):
            if labels[row][col] > 0: labels[row][col] = linked[labels[row][col]]

    return labels


def get_component_size(labeled_image, size):
    labels, labels_q = np.unique(labeled_image, return_counts=True)
    component_sizes = dict(zip(labels, labels_q))
    
    if 0 in component_sizes: del component_sizes[0]
    if size.lower() == 'small':
        min_size_label = min(component_sizes, key=component_sizes.get)
        min_size = component_sizes[min_size_label]
        image = np.where(labeled_image == min_size_label, 255, 0)
        info = f"La componente conexa de menor tamaño es la etiqueta {min_size_label} con {min_size} unidades."

    elif size.lower() == 'large':
        max_size_label = max(component_sizes, key=component_sizes.get)
        max_size = component_sizes[max_size_label]
        image = np.where(labeled_image == max_size_label, 255, 0)
        info = f"La componente conexa de mayor tamaño es la etiqueta {max_size_label} con {max_size} unidades."

    return info, image
