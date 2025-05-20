import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import logging

# ログの設定
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CNN_LSTM(nn.Module):
    def __init__(self, in_channels, cnn_channels, lstm_hidden, output_dim, embed_dim):
        super(CNN_LSTM, self).__init__()
        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=cnn_channels, 
                kernel_size=(1, 3),       # 時間方向1 → 時間長を変えない
                stride=(1, 1),            # 時間方向1 → ダウンサンプリングしない
                padding=(0, 1)            # 空間方向にだけパディング（周辺情報も見られるように）
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))  # 空間方向だけプーリング
        )
        # LSTM部分
        self.lstm_input_size = cnn_channels * 3 + embed_dim  # CNNの出力次元 + 言語特徴量の次元
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,  # 動的に計算された値を使用
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        # 出力層
        self.fc = nn.Linear(lstm_hidden * 2, output_dim)

    def forward(self, articulatory_features, phoneme_embeddings):
        # CNN部分
        x = self.cnn(articulatory_features)  # CNNの出力
        logging.info(f"CNN output shape: {x.shape}")  # CNNの出力形状をログに出力
        batch_size, cnn_channels, time_steps, feature_dim = x.size()
        x = x.permute(0, 2, 1, 3).reshape(batch_size, time_steps, cnn_channels * feature_dim)  # LSTM用に形状を変換

        # 言語特徴量をLSTMの入力に追加
        if phoneme_embeddings is not None:
            logging.info(f"Phoneme embeddings shape: {phoneme_embeddings.shape}")  # 言語特徴量の形状をログに出力
            x = torch.cat((x, phoneme_embeddings), dim=-1)  # 最後の次元で結合
            logging.info(f"Combined input shape for LSTM: {x.shape}")  # LSTMに渡すテンソルの形状をログに出力

        # LSTM部分
        x, _ = self.lstm(x)
        # 出力層
        x = self.fc(x)
        return x