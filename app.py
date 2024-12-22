from fastapi import FastAPI, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms
from mangum import Mangum
from typing import List, Union
import timm
import torch
import torch.nn as nn
import uuid
import numpy as np
import torch.nn.functional as F
import uvicorn

app = FastAPI()
handler=Mangum(app)

image_results = {}

class SwinWithTwoOutputs(nn.Module):

    # Caricamento del modello
    def __init__(self, model_path, num_classes_multiclass, num_classes_binary):
        super(SwinWithTwoOutputs, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model for device {self.device}")

        self.base = timm.create_model('swin_large_patch4_window7_224', num_classes=num_classes_multiclass)
        self.fc_2cls = nn.Linear(num_classes_multiclass, num_classes_binary, bias=True)

        # Separare i componenti del blocco head
        self.global_pool = self.base.head.global_pool
        self.dropout = self.base.head.drop
        self.fc = self.base.head.fc

        # Rimuovere l'head originale del modello
        self.base.head = nn.Identity()

        for param in self.base.parameters():
            param.requires_grad = True

        self.state_dict = torch.load(model_path, map_location=self.device)

        self.load_state_dict(self.state_dict)
        self.to(self.device)
        self.eval()


    # Preprocessing dell'immagine
    def preprocess(self, image_data):
        image = Image.fromarray(image_data)
        if image.mode != "RGB":
            image = image.convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((256)),
            transforms.CenterCrop((224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(image)
        return torch.unsqueeze(tensor, dim=0)


    # Predizione per un'immagine che restituisce l'output multiclasse, la sua percentuale e l'output binario.
    def predict(self, image_data):
        data = image_data.to(self.device)

        features = self.base(data)
        pooled_features = self.global_pool(features)
        pooled_features = self.dropout(pooled_features)
        pooled_features = torch.flatten(pooled_features, 1)

        multiclass_output = self.fc(pooled_features)
        probabilities = F.softmax(multiclass_output, dim=1)
        pred_multiclass = probabilities.argmax(dim=1, keepdim=True).item()
        probabilities_formatted = [format(prob, '.4f') for prob in probabilities.squeeze().tolist()]

        binary_output = self.fc_2cls(multiclass_output)
        probabilities_binary = F.softmax(binary_output, dim=1)  
        binary_prediction = probabilities_binary.argmax(dim=1).item()
        probabilities_binary_formatted = [format(prob, '.4f') for prob in probabilities_binary.squeeze().tolist()]

        return pred_multiclass, probabilities_formatted, binary_prediction, probabilities_binary_formatted
        


    # Inferenza dell'immagine
    def infer(self, image_data):
        preprocessed_image_data = self.preprocess(image_data)
        prediction = self.predict(preprocessed_image_data)
        return prediction


inference_model = SwinWithTwoOutputs('model/model_best_test_nd3+multi.pt', num_classes_multiclass=8, num_classes_binary=2)

class PendingClassificationDTO(BaseModel):
    inference_id: str

class ClassificationResultDTO(BaseModel):
    predicted_multiclass: int
    probabilities_multi: List[float]
    predicted_binary: int
    probabilities_binary: List[float]


class ImageData(BaseModel):
    image_data: bytes


@app.get('/', tags=["Root"])
async def hello():
    return {"Hello": "welcome to Dermchecker backend!"}


@app.post('/analyze', response_model=PendingClassificationDTO, status_code=200, tags=["Inference"])
async def analyze(file:UploadFile):
    im = Image.open(file.file)
    image_data = np.array(im)
    inference_id = str(uuid.uuid4())
    image_results[inference_id] = image_data

    print(f"Inference ID: {inference_id}, ImageShape: {image_data.shape}")
    return PendingClassificationDTO(inference_id=inference_id)


@app.get('/result/{inference_id}', status_code=200, response_model=Union[ClassificationResultDTO, PendingClassificationDTO], tags=["Inference"])
async def classification_result(inference_id:str):
    image_data= image_results.get(inference_id)
    if image_data is None:
        print("No image data")
        return PendingClassificationDTO(inference_id=inference_id)
    else:
        result_multiclass, result_probabilities_multi, result_binary, result_probabilities_binary = inference_model.infer(image_data)
        print(f"Multiclass: {result_multiclass}, Probabilities multiclass: {result_probabilities_multi}, Binary: {result_binary}, Probabilities binary: {result_probabilities_binary}")
        return ClassificationResultDTO(predicted_multiclass=result_multiclass, probabilities_multi=result_probabilities_multi, 
                                       predicted_binary=result_binary, probabilities_binary=result_probabilities_binary)