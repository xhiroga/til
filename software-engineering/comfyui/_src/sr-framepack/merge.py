from safetensors.torch import load_file, save_file
import argparse

def merge_safetensors_files(file_paths, output_path):
    merged_state_dict = load_file(file_paths[0])
    for file_path in file_paths[1:]:
        state_dict = load_file(file_path)
        merged_state_dict.update(state_dict)
    
    save_file(merged_state_dict, output_path)

def main():
    parser = argparse.ArgumentParser(description='Safetensors files to merge')
    parser.add_argument('--input_files', nargs='+', required=True, help='Input files to merge (multiple files are possible)')
    parser.add_argument('--output', required=True, help='Output file path')
    
    args = parser.parse_args()
    
    print(f"Merging {len(args.input_files)} files to {args.output}")
    merge_safetensors_files(args.input_files, args.output)
    print(f"Merge completed: {args.output}")

if __name__ == '__main__':
    main()
