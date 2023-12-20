from transformers import AutoTokenizer, BlenderbotForConditionalGeneration

def load_model(model_name_or_path, cache_dir='cache'):
    
    #Load model and corresponding tokenizer
    
    model_name = model_name_or_path 
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
    return tokenizer, model