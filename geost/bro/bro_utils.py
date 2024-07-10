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
