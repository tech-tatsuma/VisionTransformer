import torch
import torch.nn.functional as F
from utils.img_to_patch import img_to_patch

def img2attentionscores(img_tensor, model, device, image_size, patch_size, num_heads, num_patches):
    # 画像テンソルをGPUに転送
    img = img_tensor.to(device)

    # 入力画像をパッチに変換 (1, 1, image_size, image_size) -> (1, num_patches, patch_size*patch_size)
    patches = img_to_patch(img_tensor.unsqueeze(0), patch_size=patch_size)

    # パッチを入力層を通して埋め込みベクトルに変換
    patches = model.input_layer(patches.float())

    # クラストークンを結合し，位置埋め込みを追加
    transformer_input = torch.cat((model.cls_token, patches), dim=1) + model.pos_embedding

    # 最初のトランスフォーマーブロックの線型層を通して出力を計算
    transformer_input_expanded = model.transformer[0].linear[0](transformer_input).squeeze(0)

    # 出力を (num_patches+1, 3, num_heads, -1) の形状に変換
    qkv = transformer_input_expanded.reshape(num_patches+1, 3, num_heads, -1) # (num_patches+1, 3, num_heads, -1)

    # クエリ，キー，バリューを分割し次元を変更
    q = qkv[:, 0].permute(1, 0, 2) # (num_heads, num_patches+1, -1)
    k = qkv[:, 1].permute(1, 0, 2) # (num_heads, num_patches+1, -1)
    kT = k.permute(0, 2, 1)

    # クエリとキーの積を計算してattention matrixを取得
    attention_matrix = q @ kT # (num_heads, num_patches+1, num_patches+1)

    # 全てのヘッドでattention matrixを平均化
    attention_matrix_mean = torch.mean(attention_matrix, dim=0) # (num_patches+1, num_patches+1)

    # 残差接続を考慮してattention matrixに単位行列を追加
    residual_att = torch.eye(attention_matrix_mean.size(1)).to(device) # (num_patches+1, num_patches+1)
    attention_matrix_mean = attention_matrix_mean.to(device)
    aug_att_mat = attention_matrix_mean + residual_att # (num_patches+1, num_patches+1)

    # attention matrixを再正規化
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1) # (num_patches+1, num_patches+1)

    # クラストークンから各パッチへのアテンションスコアを抽出し，元の画像サイズにリサイズ
    attn_heatmap = aug_att_mat[0, 1:].reshape((int(image_size/patch_size), int(image_size/patch_size)))
    attn_heatmap_resized = F.interpolate(attn_heatmap.unsqueeze(0).unsqueeze(0), [image_size, image_size], mode='bilinear').view(image_size, image_size, 1)

    return attn_heatmap_resized