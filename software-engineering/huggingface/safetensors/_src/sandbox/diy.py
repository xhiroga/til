import json
import struct
import sys


def parse_safetensors_header(file_path) -> dict:
    with open(file_path, 'rb') as f:
        header_len_bytes = f.read(8)
        print(f"{header_len_bytes=}")
        header_len = struct.unpack('<Q', header_len_bytes)[0]
        print(f"{header_len=}")

        header_json_bytes = f.read(header_len)
        header_json_str = header_json_bytes.decode('utf-8')
        header_json = json.loads(header_json_str)
        return header_json

def main():
    file_path = sys.argv[1]
    try:
        header_json = parse_safetensors_header(file_path)
        print(f"{header_json=}")
    except Exception as e:
        print(f"{e=}")


if __name__ == "__main__":
    main()
