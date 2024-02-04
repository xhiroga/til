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


def root_dir(path: str) -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.normpath(os.path.join(project_root, path))


def data_dir(path: str):
    return os.path.normpath(os.path.join(root_dir('data'), path))


DB_NAME = root_dir('db/pipeline.db')
interface: Optional[HiInterface] = None


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
    return [Metadata(*row) for row in rows]


def read_metadata_by(step: Step, bucket: str, path: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM metadata WHERE step = ? AND bucket = ? AND path = ?", (step.value, bucket, path))
    row = cursor.fetchone()
    conn.close()
    return Metadata(*row) if row else None


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
    cursor.execute("INSERT INTO jobs (step, executed_at) VALUES (?, ?)", 
    (step.value, executed_at))
    conn.commit()
    conn.close()


def calculate_hash(image: Image) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_byte_arr = img_byte_arr.getvalue()
    hash_obj =  hashlib.sha256(img_byte_arr)
    return hash_obj.hexdigest()


def generate_path(original_path: str, hash: str, ext: str):
    match = re.search(r'(.*)_(\w+)\.(\w+)', original_path)
    path_head, _hash, _ext = match.groups() if match else (None, None, None)
    return f"{path_head}_{hash[:8]}.{ext}"


def node_nobg():
    input_step = Step.raw
    last_executed_at = read_last_executed_at(input_step)
    non_processed_metadata = read_metadata_from(input_step, last_executed_at)
    print(f"Job started at {datetime.now()}, processing {len(non_processed_metadata)} items.")
    for metadata in non_processed_metadata:
        process_nobg(metadata)
    print(f"Job ended at {datetime.now()}.")
    create_job(Step.nogb, datetime.now())


def process_nobg(metadata: Metadata, bucket: str = data_dir(Step.nogb.value)):
    if interface is None:
        interface = HiInterface(
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

    images_without_background = interface([metadata.path])
    image_wo_bg = images_without_background[0]

    ext = "png"
    nobg_hash = calculate_hash(image_wo_bg)
    path = generate_path(metadata.name, nobg_hash, ext)
    output_file_path = os.path.join(bucket, path)
    image_wo_bg.save(output_file_path)
    nobg_metadata = Metadata(bucket=bucket, path=path, step=Step.nogb, label=metadata.label, created_at=datetime.datetime.now())
    create_metadata(nobg_metadata)


def node_crop():
    input_step = Step.nogb
    # TODO: last_executed_at
    # TODO: metadatas = get_new_data(input_step, last_executed_at)
    pass


def process_crop(metadata: Metadata):
    output_step = Step.cropped
    # TODO: 
    pass


if __name__ == "__main__":
    while True:
        node_nobg()
        node_crop()
