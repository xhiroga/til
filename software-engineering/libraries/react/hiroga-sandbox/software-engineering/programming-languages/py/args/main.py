import argparse

parser = argparse.ArgumentParser()
parser.add_argument('all', nargs='*')
args = parser.parse_args()
print(args.all)