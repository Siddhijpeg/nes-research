MODEL_REGISTRY = {
'llama': { 'root': 'model.layers', 'mlp': '.mlp', 'attn': '.self_attn' },
'mistral': { 'root': 'model.layers', 'mlp': '.mlp', 'attn': '.self_attn' },
'gemma': { 'root': 'model.layers', 'mlp': '.mlp', 'attn': '.self_attn' },
'qwen': { 'root': 'model.layers', 'mlp': '.mlp', 'attn': '.self_attn' },
'phi': { 'root': 'model.layers', 'mlp': '.mlp', 'attn': '.self_attn' },
'deepseek': { 'root': 'model.layers', 'mlp': '.mlp', 'attn': '.self_attn' },
'olmo': { 'root': 'model.layers', 'mlp': '.mlp', 'attn': '.self_attn' },
'falcon': { 'root': 'transformer.h', 'mlp': '.mlp', 'attn': '.self_attention' },
'mixtral': { 'root': 'model.layers', 'mlp': '.block_sparse_moe', 'attn': '.self_attn' },
}
def get_layer_module(model, family: str, layer_id: int, component: str = 'mlp'):
    reg = MODEL_REGISTRY[family]
    root = model
    for part in reg['root'].split('.'):
        root = getattr(root, part)
        layer = root[layer_id]
        suffix = reg[component].lstrip('.')
        return getattr(layer, suffix)

def get_num_layers(model) -> int:
    cfg = model.config
    for attr in ('num_hidden_layers', 'n_layer', 'num_layers'):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
        raise ValueError('Cannot detect num_layers from model.config')