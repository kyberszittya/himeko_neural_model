import time
import os

import numpy as np

from torch.onnx.symbolic_opset9 import one_hot

from himeko_hypergraph.src.factories.creation_elements import FactoryHypergraphElements, create_vertex_by_labels
from himeko_hypergraph.src.mapping.surjective_mapping import surjective_connection, surjective_constructor_connection, \
    surjective_mapping, surjective_constructor_mapping
from himeko_neural_model.src.foundations.dataset.dataset_node import dataset_load, ClassLabelTransformer
from himeko_neural_model.src.foundations.dataset.text_dataset_node import CharDatasetNode


def create_dataset_node():
    v0: CharDatasetNode = FactoryHypergraphElements.create_vertex_constructor_default(
        CharDatasetNode, "dataset", time.time_ns())
    SEED = 242
    DATA_DIR = os.path.join("../../../text_datalist/class_data")
    df_raw = dataset_load(DATA_DIR)
    return df_raw


def test_char_dataset_class_mapping():
    df_raw = create_dataset_node()
    unique_labels = df_raw['label'].unique()
    assert 'city' in unique_labels
    assert 'degree' in unique_labels
    assert 'fname' in unique_labels
    assert 'lname' in unique_labels
    assert 'midschool' in unique_labels
    assert 'profession' in unique_labels
    assert 'softskills' in unique_labels
    assert 'university' in unique_labels
    assert 'university_abbreviation' in unique_labels
    assert 'lang' in unique_labels
    assert 'irsz' in unique_labels
    assert 'email' in unique_labels
    mapping_node = FactoryHypergraphElements.create_vertex_default("class_mapping", time.time_ns())
    raw_labels_vertices = create_vertex_by_labels(df_raw['label'].unique(), time.time_ns(), mapping_node)

    labels = dict()
    inv_labels = dict()
    with open('../../../text_datalist/class_mapping/class_mapping.txt') as f:
        for l in f:
            m, t = l.strip().split(',')
            m, t = m.strip(), t.strip()
            labels[m] = t
            if t not in inv_labels:
                inv_labels[t] = []
            inv_labels[t].append(m)
    edges = []
    # Create output tensor
    tensor_output = FactoryHypergraphElements.create_vertex_default(f"class_tensor_output", time.time_ns(), mapping_node)
    # Create maps
    mapped_labels = []
    for i,o in inv_labels.items():
        map_label = FactoryHypergraphElements.create_vertex_default(i, time.time_ns(), mapping_node)
        e = surjective_connection(
            f"conn_edge_{i}", time.time_ns(), o, i,
            mapping_node, lambda x1, x2: x1.name == x2)
        edges.append(e)
        mapped_labels.append(map_label)
    tensor_map = surjective_constructor_mapping(
        ClassLabelTransformer, f"class_tensor_transformer", time.time_ns(), mapped_labels, tensor_output, mapping_node)
    assert tensor_map('city') == 0
    assert tensor_map('email') == 2
    assert tensor_map('other') == 7
    assert tensor_map('education') == 1
    assert tensor_map.labels == ['city', 'education', 'email', 'gt', 'lang', 'name', 'number', 'other', 'profession', 'softskills']
    assert len(edges) == len(tensor_map.labels)
    class_one_hot_encoding = tensor_map('email', one_hot=True).numpy()
    assert class_one_hot_encoding.shape == (len(tensor_map.labels),)
    assert np.all(class_one_hot_encoding == np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
