import numpy as np


def get_bbox_criteria(xmin, xmax, ymin, ymax):
    json_line = {
        "area": {
            "boundingBox": {
                "lowerCorner": {"lat": ymin, "lon": xmin},
                "upperCorner": {"lat": ymax, "lon": xmax},
            }
        }
    }
    return json_line


def divide_bbox(xmin, xmax, ymin, ymax, level=1):
    # order: upper left, lower left, upper right, lower right
    # xmins = np.zeros(level**level)
    # xmaxs = np.zeros(level**level)
    # ymins = np.zeros(level**level)
    # ymaxs = np.zeros(level**level)

    # xmin_coors = np.linspace(xmin, xmax, level + 1)[:level]
    # xmax_coors = np.linspace(xmax, xmin, level + 1)[:level][::-1]
    # ymin_coors = np.linspace(ymin, ymax, level + 1)[:level]
    # ymax_coors = np.linspace(ymax, ymin, level + 1)[:level][::-1]

    # for i in range(level):
    #     xmins[(i + 1) * level : (i + 1) * level + 1] = xmin_coors

    return xmin, xmax, ymin, ymax
