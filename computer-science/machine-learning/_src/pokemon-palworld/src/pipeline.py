import hashlib
import io
import os
import re
import sqlite3
from typing import Optional
import torch

from carvekit.api.high import HiInterface
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from PIL import Image


class Step(Enum):
    video = 'video'
    raw = 'raw'
    nogb = 'nobg'
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


DB_NAME = root_dir('db/pipeline.db')


def calculate_hash(image: Image, ext: Optional[str]) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=ext or image.format)
    img_byte_arr = img_byte_arr.getvalue()
    hash_obj =  hashlib.sha256(img_byte_arr)
    return hash_obj.hexdigest()

def generate_path(original_path: str, hash: str, ext: str):
    match = re.search(r'(.*)(_\w{8})?\.(\w+)', original_path)
    path_head, _hashed, _ext = match.groups() if match else (None, None, None)
    return f"{path_head}_{hash[:8]}.{ext}"


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
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if last_executed_at:
        cursor.execute("SELECT * FROM metadata WHERE step = ? AND created_at > ?", (step.value, last_executed_at))
    else:
        cursor.execute("SELECT * FROM metadata WHERE step = ?", (step.value,))
    rows = cursor.fetchall()
    conn.close()
    return [Metadata(row[0], row[1], Step(row[2]), Label(row[3]), row[4]) for row in rows]

def read_metadata_by(step: Step, bucket: str, path: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM metadata WHERE step = ? AND bucket = ? AND path = ?", (step.value, bucket, path))
    row = cursor.fetchone()
    conn.close()
    return Metadata(row[0], row[1], Step(row[2]), Label(row[3]), row[4]) if row else None

def create_metadata(metadata: Metadata):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO metadata (bucket, path, step, label, created_at) VALUES (?, ?, ?, ?, ?)", 
    (metadata.bucket, metadata.path, metadata.step.value, metadata.label.value, metadata.created_at))
    conn.commit()
    conn.close()

def read_last_executed_at(step: Step) -> Optional[datetime]:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(executed_at) FROM jobs WHERE step = ?", (step.value,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def create_job(step: Step, executed_at: datetime):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO jobs (step, executed_at) VALUES (?, ?)", (step.value, executed_at))
    conn.commit()
    conn.close()


class Pipeline:
    def __init__(self):
        self.interface: Optional[HiInterface] = None

    def node_nobg(self):
        input_step = Step.raw
        last_executed_at = read_last_executed_at(input_step)
        non_processed_metadata = read_metadata_from(input_step, last_executed_at)
        print(f"Job nobg started at {datetime.now()}, processing {len(non_processed_metadata)} items.")
        for metadata in non_processed_metadata:
            self.process_nobg(metadata)
        print(f"Job nobg ended at {datetime.now()}.")
        create_job(Step.nogb, datetime.now())

    def process_nobg(self, metadata: Metadata, bucket: str = data_dir(Step.nogb.value)):
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
        nobg_metadata = Metadata(bucket=bucket, path=path, step=Step.nogb, label=metadata.label, created_at=datetime.now())
        os.makedirs(os.path.dirname(nobg_metadata.full_path), exist_ok=True)
        image_wo_bg.save(nobg_metadata.full_path)
        create_metadata(nobg_metadata)

    def node_crop(self):
        input_step = Step.nogb
        # TODO: last_executed_at
        # TODO: metadatas = get_new_data(input_step, last_executed_at)
        pass

    def process_crop(self, metadata: Metadata):
        output_step = Step.cropped
        # TODO: 
        pass


if __name__ == "__main__":
    pipeline = Pipeline()
    while True:
        pipeline.node_nobg()
        pipeline.node_crop()
