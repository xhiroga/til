import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time

class ImageClassifier:
    def __init__(self):
        print("Loading pre-trained ResNet model...")
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def preprocess_image(self, image_array):
        """画像の前処理"""
        # NumPy配列をPIL Imageに変換
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        image = Image.fromarray(image_array)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 前処理を適用
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        
        return input_batch
    
    def classify(self, image_array):
        """画像分類を実行"""
        # 前処理
        input_batch = self.preprocess_image(image_array)
        
        # 推論
        with torch.no_grad():
            output = self.model(input_batch)
        
        # 後処理
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        return top5_prob.numpy(), top5_catid.numpy()
    
    def batch_classify(self, image_arrays, batch_size=32):
        """バッチ推論を実行"""
        results = []
        
        for i in range(0, len(image_arrays), batch_size):
            batch_images = image_arrays[i:i+batch_size]
            batch_tensors = []
            
            # バッチの前処理
            for img in batch_images:
                preprocessed = self.preprocess_image(img)
                batch_tensors.append(preprocessed)
            
            # バッチテンソルを作成
            batch = torch.cat(batch_tensors, dim=0)
            
            # バッチ推論
            with torch.no_grad():
                outputs = self.model(batch)
            
            # 各画像の結果を処理
            for output in outputs:
                probabilities = torch.nn.functional.softmax(output, dim=0)
                top5_prob, top5_catid = torch.topk(probabilities, 5)
                results.append((top5_prob.numpy(), top5_catid.numpy()))
        
        return results

def simulate_heavy_preprocessing(image):
    """重い前処理をシミュレート"""
    time.sleep(0.01)  # I/O待機をシミュレート
    
    # 複数の画像変換を実行
    for _ in range(5):
        image = np.flip(image, axis=0)
        image = np.flip(image, axis=1)
    
    return image

def main():
    print("PyTorch推論デモを開始します...")
    
    # モデルを初期化
    classifier = ImageClassifier()
    
    # ダミー画像を生成（RGB画像をシミュレート）
    num_images = 100
    print(f"\n{num_images}枚のダミー画像を生成中...")
    
    images = []
    for i in range(num_images):
        # ランダムなRGB画像を生成
        dummy_image = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
        
        # 重い前処理をシミュレート
        processed_image = simulate_heavy_preprocessing(dummy_image)
        images.append(processed_image)
    
    # 単一画像の推論
    print("\n単一画像の推論を実行中...")
    start_time = time.time()
    
    for i in range(10):
        result = classifier.classify(images[i])
        if i == 0:
            print(f"  Top-5予測確率: {result[0][:5]}")
    
    single_time = time.time() - start_time
    print(f"  単一画像推論時間（10枚）: {single_time:.2f}秒")
    
    # バッチ推論
    print("\nバッチ推論を実行中...")
    start_time = time.time()
    
    results = classifier.batch_classify(images, batch_size=16)
    
    batch_time = time.time() - start_time
    print(f"  バッチ推論時間（{num_images}枚）: {batch_time:.2f}秒")
    
    # 結果の統計を表示
    print(f"\n推論結果:")
    print(f"  処理した画像数: {len(results)}")
    print(f"  平均推論時間/画像: {batch_time/num_images:.4f}秒")
    
    # メモリ使用量の確認
    if torch.cuda.is_available():
        print(f"\nGPUメモリ使用量: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

if __name__ == "__main__":
    main()
