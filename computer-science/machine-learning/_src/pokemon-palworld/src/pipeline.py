import argparse
import cv2
import hashlib
import io
import logging
import numpy as np
import os
import re
import sqlite3
import time
import torch

from carvekit.api.high import HiInterface
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional
from PIL import Image


class Step(Enum):
    video = 'video'
    raw = 'raw'
    nobg = 'nobg'
    cropped = 'cropped'

class Label(Enum):
    pokemon = 'pokemon'
    pal = 'pal'

@dataclass
class Metadata:
    bucket: str
    path: str
    step: Step
    label: Label
    created_at: datetime

    @property
    def full_path(self) -> str:
        return os.path.join(self.bucket, self.path)


def root_dir(path: str) -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.normpath(os.path.join(project_root, path))


def data_dir(path: str):
    return os.path.normpath(os.path.join(root_dir('data'), path))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)

DB_NAME = root_dir('db/pipeline.db')


def calculate_hash(image: Image, ext: Optional[str]) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=ext or image.format)
    img_byte_arr = img_byte_arr.getvalue()
    hash_obj =  hashlib.sha256(img_byte_arr)
    return hash_obj.hexdigest()

def generate_path(original_path: str, hash: str, ext: str, index: Optional[str] = None):
    match = re.search(r'(.*?)(_\w{8})?\.(\w+)', original_path)
    path_head, _hashed, _ext = match.groups() if match else (None, None, None)
    return f"{path_head}{f'_{index}' if index is not None else ''}_{hash[:8]}.{ext}"

def create_tables_if_not_exists():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            bucket TEXT,
            path TEXT,
            step TEXT,
            label TEXT,
            created_at TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            step TEXT,
            executed_at TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def read_metadata_from(step: Step, last_executed_at: datetime | None):
    logging.debug(f"Reading metadata from step: {step}, last executed at: {last_executed_at}")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if last_executed_at:
        cursor.execute("SELECT * FROM metadata WHERE step = ? AND created_at > ?", (step.value, last_executed_at))
    else:
        cursor.execute("SELECT * FROM metadata WHERE step = ?", (step.value,))
    rows = cursor.fetchall()
    conn.close()
    logging.debug(f"Read metadata: {rows[:5]}...")
    return [Metadata(row[0], row[1], Step(row[2]), Label(row[3]), row[4]) for row in rows]

def read_metadata_by(step: Step, bucket: str, path: str):
    logging.debug(f"Reading metadata by step: {step}, bucket: {bucket}, path: {path}")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM metadata WHERE step = ? AND bucket = ? AND path = ?", (step.value, bucket, path))
    row = cursor.fetchone()
    conn.close()
    logging.debug(f"Read metadata: {row}")
    return Metadata(row[0], row[1], Step(row[2]), Label(row[3]), row[4]) if row else None

def create_metadata(metadata: Metadata):
    logging.debug(f"Creating metadata: {metadata}")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO metadata (bucket, path, step, label, created_at) VALUES (?, ?, ?, ?, ?)", 
    (metadata.bucket, metadata.path, metadata.step.value, metadata.label.value, metadata.created_at))
    conn.commit()
    conn.close()

def read_last_executed_at(step: Step) -> Optional[datetime]:
    logging.debug(f"Reading last executed at for step: {step}")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(executed_at) FROM jobs WHERE step = ?", (step.value,))
    result = cursor.fetchone()
    conn.close()
    logging.debug(f"Read last executed at: {result[0] if result else None}")
    return result[0] if result else None

def create_job(step: Step, executed_at: datetime):
    logging.debug(f"Creating job for step: {step}, executed at: {executed_at}")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO jobs (step, executed_at) VALUES (?, ?)", (step.value, executed_at))
    conn.commit()
    conn.close()

def get_object_bounding_boxes(image: Image):
        individual_channels = image.split()

        alpha_channel: np.array
        if len(individual_channels) == 4:
            alpha_channel = np.array(individual_channels[3])
        else:
            raise ValueError("Image does not have an alpha channel.")

        # cv2.threshold関数を使用して、アルファチャンネルの値が1以上のピクセルを255（白）に、それ以外を0（黒）に変換します。
        # これにより、画像のオブジェクト部分を白、背景部分を黒としたバイナリマスクが作成されます。
        _, binary_mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours or []

class Pipeline:
    def __init__(self):
        self.interface: Optional[HiInterface] = None

    def node(self, process_function, input_step: Step, output_step: Step, ):
        last_executed_at = read_last_executed_at(output_step)
        unprocessed_metadata = read_metadata_from(input_step, last_executed_at)
        logging.info(f"Job {output_step.value} started, processing {len(unprocessed_metadata)} items.")
        if len(unprocessed_metadata) == 0:
            return

        for unprocessed in unprocessed_metadata:
            if not os.path.exists(unprocessed.full_path):
                logging.warn(f"File {unprocessed.full_path} does not exist. Skipping.")
                continue

            processeds = process_function(unprocessed)
            for processed in processeds:
                create_metadata(processed)

        logging.info(f"Job {output_step.value} ended.")
        create_job(output_step, datetime.utcnow())

    def process_nobg(self, metadata: Metadata, bucket: str = data_dir(Step.nobg.value)) -> List[Metadata]:
        if self.interface is None:
            self.interface = HiInterface(
                object_type="object",  # Can be "object" or "hairs-like".
                batch_size_seg=5,
                batch_size_matting=1,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                matting_mask_size=2048,
                trimap_prob_threshold=231,
                trimap_dilation=30,
                trimap_erosion_iters=5,
                fp16=False
            )

        images_without_background = self.interface([metadata.full_path])
        image_wo_bg = images_without_background[0]

        ext = "png"
        nobg_hash = calculate_hash(image_wo_bg, ext)
        path = generate_path(metadata.path, nobg_hash, ext)
        nobg_metadata = Metadata(bucket=bucket, path=path, step=Step.nobg, label=metadata.label, created_at=datetime.utcnow())
        os.makedirs(os.path.dirname(image_wo_bg.full_path), exist_ok=True)
        image_wo_bg.save(nobg_metadata.full_path)
        
        return [nobg_metadata]
    
    def process_crop(self, metadata: Metadata, bucket: str = data_dir(Step.cropped.value)):
        min_height, min_width = 64, 64

        image = Image.open(metadata.full_path)
        contours = get_object_bounding_boxes(image)
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_height * min_width]
        cropped_metadatas = []
        
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cropped_image = image.crop((x, y, x + w, y + h))

            ext = "png"
            crop_hash = calculate_hash(cropped_image, ext)
            path = generate_path(metadata.path, crop_hash, ext)
            cropped_metadata = Metadata(bucket=bucket, path=path, step=Step.cropped, label=metadata.label, created_at=datetime.utcnow())
            os.makedirs(os.path.dirname(cropped_metadata.full_path), exist_ok=True)
            cropped_image.save(cropped_metadata.full_path)
            cropped_metadatas.append(cropped_metadata)
        
        return cropped_metadatas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging-level', default='INFO', help='Set logging level')
    args = parser.parse_args()
    logging.getLogger().setLevel(args.logging_level)

    pipeline = Pipeline()
    while True:
        pipeline.node(pipeline.process_nobg, Step.raw, Step.nobg)
        pipeline.node(pipeline.process_crop, Step.nobg, Step.cropped)
        time.sleep(10)
