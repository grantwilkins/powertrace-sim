from model.utils.decode_time import derive_decode_time


def test_derive_decode_time_list_itl():
    decode_time, source = derive_decode_time([0.1, 0.2, 0.3], 4)
    assert source == "list"
    assert decode_time is not None
    assert abs(decode_time - 0.6) < 1e-12


def test_derive_decode_time_scalar_itl():
    decode_time, source = derive_decode_time(0.01, 100)
    assert source == "scalar"
    assert decode_time is not None
    assert abs(decode_time - 0.99) < 1e-12


def test_derive_decode_time_empty_list():
    decode_time, source = derive_decode_time([], 10)
    assert source == "list"
    assert decode_time is None


def test_derive_decode_time_none():
    decode_time, source = derive_decode_time(None, 10)
    assert source == "unknown"
    assert decode_time is None
