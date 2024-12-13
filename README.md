# Vision Transformer(Attention Mapの出力も含む)
## フォルダ構成
```
root/
│
├── imagenet/              # ImageNetデータセット用
│   ├── train.py           # モデルの学習スクリプト
│   ├── vis_attn.py        # 注目領域の可視化スクリプト
│   └── dataloaders/       # データローダー関連コード
│       ├── classes.py     # クラス情報の管理
│       ├── dataset.py     # データセットの定義と前処理
│       └── imagenet-1k.py # ImageNet-1k用の設定
│
├── mnist/                 # MNISTデータセット用
│   ├── train.py           # モデルの学習スクリプト
│   └── vis_attn.py        # 注目領域の可視化スクリプト
│
├── models/                # モデル定義
│   ├── attentionblock.py  # ViTのAttention Blockの定義
│   └── vit.py             # ViTモデルの定義
│
├── utils/                 # ユーティリティモジュール
│   ├── img_to_patch.py    # 画像をパッチに分割するモジュール
│   └── img2attn.py        # Attention Heatmapの抽出モジュール
│
└── README.md              # プロジェクト全体の説明ファイル
```

## 使い方
### 学習(MNIST)
1. パッケージのインストール
```
pip install torch torchvision
pip install -r requirements.txt
```
2. 訓練スクリプトの実行
```
python -m mnist.train
```
### Attention mapの可視化
1. `vis_attn.py`を編集
```
image_path = "<注目領域を可視化したい画像のパス>"
model_path = "<重みファイル>"
output_path = "<出力ファイルのパス>"
```
2. 可視化スクリプトの実行
```
python -m mnist.vis_attn.py
```
3. 実行確認

    <img src="imgs/mnist_attention.png" width="40%">

## Reference
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [ImageNet Large Scale Visual Recognition Challenge](https://arxiv.org/abs/1409.0575)