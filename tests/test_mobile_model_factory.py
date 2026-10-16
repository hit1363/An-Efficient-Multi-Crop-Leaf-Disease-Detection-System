import pytest

tf = pytest.importorskip("tensorflow")

from training.model import get_model
from training.utils import get_preprocess_fn


@pytest.mark.parametrize(
    "architecture",
    [
        "mobilenet_v1",
        "mobilenetv2",
        "mobilenetv3_small",
        "mobilenet_v3_large",
        "nasnet_mobile",
        "efficientnet_b0",
        "efficientnet_v2_b0",
    ],
)
def test_keras_mobile_candidates_produce_72_logits_without_downloads(architecture):
    model, backbone = get_model(
        architecture=architecture,
        input_shape=(64, 64, 3),
        num_classes=72,
        dropout_rate=0.2,
        classifier_units=32,
        include_preprocessing=True,
        weights=None,
    )
    output = model(tf.zeros((1, 64, 64, 3)), training=False)
    assert tuple(output.shape) == (1, 72)
    assert backbone is not None
    tf.keras.backend.clear_session()


def test_each_application_input_contract_is_explicit():
    assert get_preprocess_fn("mobilenet_v1") is not None
    assert get_preprocess_fn("mobilenetv2") is not None
    assert get_preprocess_fn("nasnet_mobile") is not None
    assert get_preprocess_fn("mobilenetv3_small") is None
    assert get_preprocess_fn("mobilenet_v3_large") is None
    assert get_preprocess_fn("efficientnet_b0") is not None
    assert get_preprocess_fn("efficientnet_v2_b0") is None
