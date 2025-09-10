from transformers

import pipeline
import sys

print(pipeline("video-classification", model="MCG-NJU/videomae-base-finetuned-kinetics")(sys.argv[1]))
