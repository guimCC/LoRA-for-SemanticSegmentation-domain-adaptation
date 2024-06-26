from peft import PeftConfig, PeftModel

model_id = "segformer-gta-cityscapes-lora"

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

checkpoint_dir = "guimCC/segformer-v0-gta"

image_processor = SegformerImageProcessor()

model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_dir)

config = PeftConfig.from_pretrained(model_id)

# Load the Lora model
inference_model = PeftModel.from_pretrained(model, model_id)