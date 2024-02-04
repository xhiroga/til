import pytest
from src.pipeline import generate_path

def test_generate_path():
    generated_path = generate_path('test_example_abcd1234.png', 'efgh5678', 'png')
    expected_path = 'test_example_efgh5678.png'
    assert generated_path == expected_path

    generated_path = generate_path('poketpair/paldeck_no_001/frame_1.png', 'defg4567', 'png')
    expected_path = 'poketpair/paldeck_no_001/frame_1_defg4567.png'
    assert generated_path == expected_path

    generated_path = generate_path('poketpair/paldeck_no_001/frame_1_ccfc389a.png', 'd5022320', 'png')
    expected_path = 'poketpair/paldeck_no_001/frame_1_d5022320.png'
    assert generated_path == expected_path

    # Test case: Normal operation
    original_path = "path_to_file_12345678.png"
    hash = "abcdefgh"
    ext = "png"
    index = "1"
    expected_output = "path_to_file_1_abcdefgh.png"
    assert generate_path(original_path, hash, ext, index) == expected_output
