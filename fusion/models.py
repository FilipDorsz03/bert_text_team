import torch
from torch import nn
import numpy as np
from transformers import BertModel, Wav2Vec2Model
import timm

# --- Helper Functions ---

def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False
    module.eval()
    return module 

def partial_freeze_module(module, freeze_until_layer=8):
    for name, param in module.named_parameters():
        if "encoder.layer" in name:
            try:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_num >= freeze_until_layer: 
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            except:
                param.requires_grad = False
        elif "pooler" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return module 

def apply_mask(x, mask_ratio):
    B, S, D = x.shape
    # If mask_ratio is 0, return no mask
    if mask_ratio <= 0.0:
        return x, torch.zeros(B, S, device=x.device, dtype=torch.bool)
        
    mask = torch.rand(B, S, device=x.device) < mask_ratio
    x_masked = x.clone()
    x_masked[mask] = 0.0 
    return x_masked, mask

def interpolate(x, target_len):
    B, S, D = x.shape
    if S == target_len:
        return x
        
    # Interpolate expects [Batch, Channels, SeqLen]
    x = x.permute(0, 2, 1) 
    x = torch.nn.functional.interpolate(x, size=target_len, mode='linear', align_corners=False)
    x = x.permute(0, 2, 1)
    return x

def interpolate_masks(mask, target_len):
    B, S = mask.shape
    if S == target_len:
        return mask
    mask = mask.unsqueeze(1).float() 
    mask = torch.nn.functional.interpolate(mask, size=target_len, mode='nearest')
    return mask.squeeze(1).bool()

# --- Main Architecture ---

class TextAlignedMaskedAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, text_inputs, audio_inputs, video_inputs, mask_ratios=None):
        """
        mask_ratios: Optional tuple (text_ratio, audio_ratio, video_ratio).
                     If None, defaults to encoder's self.training status.
        """
        fused_feats, masks = self.encoder(
            text_inputs, audio_inputs, video_inputs, mask_ratios=mask_ratios
        )
        mask_t, mask_a, mask_v = masks

        (recon_t, recon_a, recon_v), _ = self.decoder(
            fused_feats, mask_t, mask_a, mask_v
        )

        return (recon_t, recon_a, recon_v), masks


class TextAlignedClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.dim, num_classes)
        
    def forward(self, text_inputs, audio_inputs, video_inputs):
        # Force 0.0 masking for classification tasks to ensure full context
        multimodal_feats, _ = self.encoder(
            text_inputs, audio_inputs, video_inputs, 
            mask_ratios=(0.0, 0.0, 0.0)
        )
        cls = multimodal_feats[:, 0, :] 
        logits = self.classifier(cls)
        return logits

    @classmethod
    def from_pretrained_mae(cls, mae_model, num_classes):
        """
        Utility to easily transfer the encoder from a pre-trained MAE 
        to a classifier instance.
        """
        return cls(mae_model.encoder, num_classes)


class TextAlignedEncoderModel(nn.Module):
    def __init__(self, text_encoder, audio_encoder, video_encoder, dim, T, n_tals=3, freeze_until_layer=8, mask_ratios=(0.3, 0.6, 0.6)):
        super().__init__()
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.mask_ratios = mask_ratios
        
        self.text_encoder = partial_freeze_module(self.text_encoder, freeze_until_layer=freeze_until_layer)
        self.audio_encoder = freeze_module(self.audio_encoder)
        self.video_encoder = freeze_module(self.video_encoder)
        
        self.lin_text = nn.Linear(self.text_encoder.config.hidden_size, dim)
        self.lin_audio = nn.Linear(self.audio_encoder.config.hidden_size, dim)
        self.lin_video = nn.Linear(self.video_encoder.output_dim, dim)
        
        assert n_tals > 1, "Number of Text-Guided Alignment Layers must be > 1"
        self.tals = nn.ModuleList([TextGuidedAlignmentLayer(dim) for _ in range(n_tals)])
        self.text_transformers = nn.ModuleList([
            nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True, dropout=0.1), 
            num_layers=1) for _ in range(n_tals)
        ])
        self.cross_modal_transformer = CrossModalAttentionTransformer(dim)
        
        self.T = T
        self.dim = dim
        
    def forward(self, text_inputs, audio_inputs, video_inputs, mask_ratios=None):
        
        # Determine masking ratios based on training mode if not provided
        if mask_ratios is None:
            if self.training:
                current_mask_ratios = self.mask_ratios
            else:
                current_mask_ratios = (0.0, 0.0, 0.0) # No masking during inference
        else:
            current_mask_ratios = mask_ratios

        # Extract features
        text_feats = self.lin_text(self.text_encoder(**text_inputs).last_hidden_state)
        audio_feats = self.lin_audio(self.audio_encoder(**audio_inputs).last_hidden_state)
        video_feats = self.lin_video(self.video_encoder(video_inputs))
        
        # Interpolate to common T
        text_feats = interpolate(text_feats, self.T)
        audio_feats = interpolate(audio_feats, self.T)
        video_feats = interpolate(video_feats, self.T)
        
        # Apply Masking
        text_feats, mask_t = apply_mask(text_feats, current_mask_ratios[0])
        audio_feats, mask_a = apply_mask(audio_feats, current_mask_ratios[1])
        video_feats, mask_v = apply_mask(video_feats, current_mask_ratios[2])
        
        # Begin Alignment
        multimodal_feats = torch.zeros_like(text_feats)
        
        # Iterative Alignment (TALs)
        for tal, text_transformer in zip(self.tals, self.text_transformers):
            text_feats = text_transformer(text_feats)
            multimodal_feats = tal(multimodal_feats, audio_feats, text_feats, video_feats)
        
        # Final Fusion
        fused_feats = self.cross_modal_transformer(text_feats, multimodal_feats, multimodal_feats)
        
        return fused_feats, (mask_t, mask_a, mask_v)
        
    
class TextAlignedDecoderModel(nn.Module):
    def __init__(
        self, 
        dim, 
        dim_text, 
        dim_audio, 
        dim_video, 
        text_length, 
        audio_length, 
        video_length, 
        mask_token_id=None,
        num_layers=2
    ):
        super().__init__()
        text_transformer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True)
        self.text_transformer = nn.TransformerEncoder(text_transformer, num_layers=num_layers)
        
        audio_transformer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True)
        self.audio_transformer = nn.TransformerEncoder(audio_transformer, num_layers=num_layers)
        
        video_transformer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True)
        self.video_transformer = nn.TransformerEncoder(video_transformer, num_layers=num_layers)
        
        self.lin_text = nn.Linear(dim,  dim_text)
        self.lin_audio = nn.Linear(dim, dim_audio)
        self.lin_video = nn.Linear(dim, dim_video)
        
        self.text_length = text_length
        self.audio_length = audio_length
        self.video_length = video_length
        self.mask_token_id = mask_token_id

    def forward(self, multimodal_feats, mask_text, mask_audio, mask_video):
        # Latent representations
        text_feats = self.text_transformer(multimodal_feats)
        audio_feats = self.audio_transformer(multimodal_feats)
        video_feats = self.video_transformer(multimodal_feats)

        # Interpolate back to original lengths
        text_feats = interpolate(text_feats, self.text_length)
        audio_feats = interpolate(audio_feats, self.audio_length)
        video_feats = interpolate(video_feats, self.video_length)
        
        # Interpolate masks
        mask_text = interpolate_masks(mask_text, self.text_length)
        mask_audio = interpolate_masks(mask_audio, self.audio_length)
        mask_video = interpolate_masks(mask_video, self.video_length)
        
        # Project back to original dimensions
        text_feats = self.lin_text(text_feats)
        audio_feats = self.lin_audio(audio_feats)
        video_feats = self.lin_video(video_feats)

        return (text_feats, audio_feats, video_feats), (mask_text, mask_audio, mask_video)


# --- Sub-Modules ---

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, 
                                               batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        query = self.norm(query)
        attn_output, _ = self.attention(query, key, value)
        out = query + self.dropout(attn_output)
        return out


class TextGuidedAlignmentLayer(nn.Module): 
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        mlp_hidden_dim = dim
        
        self.cross_attn_a = CrossModalAttention(dim, num_heads=num_heads, dropout=dropout)
        self.cross_attn_v = CrossModalAttention(dim, num_heads=num_heads, dropout=dropout)
        
        self.ffn_v = nn.Linear(dim, dim)
        self.ffn_a = nn.Linear(dim, dim)
        nn.init.constant_(self.ffn_a.bias, 2.0)
        nn.init.constant_(self.ffn_v.bias, 2.0)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, previous_output, audio_feats, text_feats, video_feats):
        
        # Use Text as Query, Audio/Video as Key/Value
        aligned_audio = self.cross_attn_a(query=text_feats, key=audio_feats, value=audio_feats)
        aligned_video = self.cross_attn_v(query=text_feats, key=video_feats, value=video_feats)
        
        # Gating
        audio_out = self.ffn_a(audio_feats)
        video_out = self.ffn_v(video_feats)
        
        gated_audio = aligned_audio * self.sigmoid(audio_out)
        gated_video = aligned_video * self.sigmoid(video_out)
        
        return previous_output + gated_audio + gated_video
    

class CrossModalAttentionTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, 
                                                     batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True, dropout=dropout),
            num_layers=1
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        query = self.norm(query)
        attn_output, _ = self.cross_attention(query, key, value)
        query = query + self.dropout(attn_output)
        output = self.transformer_encoder(query)
        return output

# --- Placeholder Backbones ---

class Bert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained('google-bert/bert-base-uncased', config=config)
    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask)
    
class Wav2Vec2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wav2vec2 = Wav2Vec2Model.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition', config=config)
    def forward(self, input_values, attention_mask=None):
        return self.wav2vec2(input_values=input_values, attention_mask=attention_mask)

class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=0)
        self.output_dim = 1280 
        
    def forward(self, x):
        # x: [Batch, Time, Channels, Height, Width]
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w) 
        features = self.cnn(x) # (B*T, 1280)
        video_features = features.view(b, t, -1) # (B, T, 1280)
        return video_features

# --- Usage Example (Run this file directly) ---
if __name__ == "__main__":
    print("Initializing Dummy Models for Test...")
    
    # 1. Dummy Data
    B, T = 2, 50
    dim = 128
    
    # Mock implementations to avoid loading heavy weights for this quick test
    class MockTextEncoder(nn.Module):
        def __init__(self): 
            super().__init__()
            self.config = type('obj', (object,), {'hidden_size': 768})
        def forward(self, **kwargs): 
            return type('obj', (object,), {'last_hidden_state': torch.randn(B, 20, 768)})
            
    class MockAudioEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('obj', (object,), {'hidden_size': 768})
        def forward(self, **kwargs):
            return type('obj', (object,), {'last_hidden_state': torch.randn(B, 100, 768)})

    class MockVideoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_dim = 1280
        def forward(self, x):
            return torch.randn(B, T, 1280)

    # 2. Initialize Models
    encoder = TextAlignedEncoderModel(
        MockTextEncoder(), MockAudioEncoder(), MockVideoEncoder(),
        dim=dim, T=T
    )
    
    decoder = TextAlignedDecoderModel(
        dim=dim, dim_text=768, dim_audio=768, dim_video=1280,
        text_length=20, audio_length=100, video_length=T
    )
    
    mae = TextAlignedMaskedAutoencoder(encoder, decoder)
    
    # 3. Forward Pass
    txt_input = {'input_ids': torch.zeros(B, 20), 'attention_mask': torch.ones(B, 20)}
    aud_input = {'input_values': torch.zeros(B, 16000)}
    vid_input = torch.zeros(B, T, 3, 224, 224)
    
    print("\n--- Running MAE Forward Pass ---")
    (rec_t, rec_a, rec_v), masks = mae(txt_input, aud_input, vid_input)
    print(f"Reconstructed Text Shape: {rec_t.shape}")
    print(f"Reconstructed Audio Shape: {rec_a.shape}")
    print(f"Reconstructed Video Shape: {rec_v.shape}")
    
    # 4. Convert to Classifier
    print("\n--- Converting to Classifier ---")
    classifier = TextAlignedClassifier.from_pretrained_mae(mae, num_classes=3)
    logits = classifier(txt_input, aud_input, vid_input)
    print(f"Classifier Logits: {logits}")
    print("\nSuccess! Code is operational.")