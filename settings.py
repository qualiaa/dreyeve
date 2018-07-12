import re

import numpy as np

import metrics

gaze_radius = 32
attention_map_frames = 25
attention_map_aggregation_name = "max"
loss_function_name = "kl"
centre_attention_map_frames = False

loss_metrics = {
    "kl": metrics.kl_divergence,
    "cc": metrics.cross_correlation,
    "mse": "mse"
}

agg_functions = {
    "sum": np.sum,
    "max": np.max
}

def loss():
    return loss_metrics[loss_function_name]

def attention_map_aggregation():
    return agg_functions[attention_map_aggregation_name]

def run_name():
    format_str = "{agg}_{loss}_{radius:d}_{frames:d}"

    if centre_attention_map_frames:
        format_str = "centred_" + format_str

    return format_str.format(
            agg=attention_map_aggregation_name,
            loss=loss_function_name,
            radius=gaze_radius,
            frames=attention_map_frames)

def parse_run_name(run_name):
    match = re.fullmatch("(centred_)?([^_]*)_([^_]*)_(\d+)_(\d+)",run_name)

    centre_attention_map_frames = match.groups()[0] is not None

    attention_map_aggregation_name = match.groups()[1]
    loss_function_name = match.groups()[2]
    gaze_radius = match.groups()[3]
    attention_map_frames  = match.groups()[4]
