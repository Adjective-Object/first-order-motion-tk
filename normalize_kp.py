from modules.util import get_inverse_kp_jacobian
import torch
from scipy.spatial import ConvexHull
import numpy as np


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], get_inverse_kp_jacobian(kp_driving_initial))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    # precalculate the inverse jacobian of driving features
    # to avoid redundant GPU syncs while running the network
    #
    # (this caches the inverse jacobian on the keypoint dict)
    get_inverse_kp_jacobian(kp_new)

    return kp_new
