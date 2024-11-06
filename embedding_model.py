import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

print('[start] embedd model loading ...')
tokenizer_embed = AutoTokenizer.from_pretrained('bert-base-uncased')
model_embed = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, safe_serialization=True)
model_embed.eval()
print('embedd model loaded ...')

def embedd(text: str) -> np.array:
    """
    Embeds a text string into a numerical vector using a pre-trained language model.

    Args:
        text (str): The input text string to be embedded.

    Returns:
        np.array: A numerical vector representation of the input text.

    Notes:
        This function uses a pre-trained language model to generate embeddings for the input text.
        The embeddings are computed using mean pooling over the token embeddings and attention mask.
        The resulting vector is normalized to have a length of 1 using L2 normalization.
    """
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    encoded_input = tokenizer_embed(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model_embed(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return np.array(embeddings)[0]
