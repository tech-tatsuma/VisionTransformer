import torch
import torch.nn as nn

from models.attentionblock import AttentionBlock
from utils.img_to_patch import img_to_patch

class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout=0.0,
    ):
        """
        Inputs:
            embed_dim - Transformerに入力する特徴ベクトルの次元数
            hidden_dim - Transformer内のフィードフォワードネットワークの隠れ層の次元数
            num_channels - 入力画像のチャンネル数
            num_heads - マルチヘッドアテンションで使用するヘッドの数
            num_layers - Transformerの中のattention blockの数
            num_classes - 予測するクラスの数
            patch_size - パッチの各辺のピクセル数
            num_patches - 画像を分割して得られるパッチの最大数
            dropout - フィードフォワードネットワークと入力埋め込みに適用するドロップアウトの割合
        """
        super().__init__()

        self.patch_size = patch_size

        # 入力レイヤー: パッチ内のピクセル数 * チャンネル数を埋め込み次元に変換
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        # トランスフォーマーレイヤーの積み重ね（レイヤー数分）
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )

        # 出力層: 最終的な分類ヘッド（正規化 → 線形変換）
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        # ドロップアウト層
        self.dropout = nn.Dropout(dropout)

        # 埋め込み用のクラス識別トークン
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 位置埋め込み行列
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x):
        """
        前向き伝播の処理
        入力: 画像テンソル x (B, C, H, W)
        出力: クラススコア (B, num_classes)
        """
        # 入力画像をパッチに分割 (B, T, P^2*C)
        x = img_to_patch(x, self.patch_size)        # x.shape ---> batch, num_patches, (patch_size**2)
        # バッチサイズとパッチ数を取得
        B, T, _ = x.shape
        # パッチ特徴ベクトルを埋め込み次元に変換 (B, T, embed_dim)
        x = self.input_layer(x)                     # x.shape ---> batch, num_patches, embed_dim

        # クラス識別トークンをバッチサイズ分複製 (B, 1, embed_dim)
        cls_token = self.cls_token.repeat(B, 1, 1)
        # 入力パッチとクラス識別トークンを結合 (B, T+1, embed_dim)
        x = torch.cat([cls_token, x], dim=1)        # x.shape ---> batch, num_patches+1, embed_dim
        # 位置埋め込みを追加 (B, T+1, embed_dim)
        x = x + self.pos_embedding[:, : T + 1]      # x.shape ---> batch, num_patches+1, embed_dim

        # ドロップアウト適用
        x = self.dropout(x)
        # トランスフォーマー層に入力 (T+1, B, embed_dim)
        x = x.transpose(0, 1)                       # x.shape ---> num_patches+1, batch, embed_dim
        # トランスフォーマーを適用 (T+1, B, embed_dim)
        x = self.transformer(x)                     # x.shape ---> num_patches+1, batch, embed_dim

        # クラス識別トークンの出力を取得 (B, embed_dim)
        cls = x[0]
        # 最終出力層で分類スコアを計算 (B, num_classes)
        out = self.mlp_head(cls)
        return out