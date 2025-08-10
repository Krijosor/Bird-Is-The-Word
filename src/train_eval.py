import torch
from torchmetrics import FBetaScore, Accuracy

from src import BWDatasets

def prediction(dataloader, model):
    preds = []
    targets = []
    count = 0
    for data in dataloader:
        count +=1

        # Print count every 10 batches
        if count % 10 == 0:
            print(f'Batch {count} of {len(dataloader)}')
        
        images = BWDatasets.tensor_to_numpy(data["ocr_image"])
        labels = data["label"][0]
        for i, image in enumerate(images):
            
            result = model.predict(image)

            pred = result[0]["rec_texts"]

            if not pred:
                pred = ['Not Found']

            preds.append(pred[0])

            label = labels[i]
            
            targets.append(label)

    return preds, targets

def evaluate_model(preds, labels):
    # Boolean mask for the predictions and the labels
    # Prediction = 1 if it is correct, 0 otherwise
    # Labels are always = to 1
    bin_preds = [int(pred == label) for pred, label in zip(preds, labels)]
    bin_labels = [1]*len(labels)

    preds_tensor, labels_tensor = torch.tensor(bin_preds), torch.tensor(bin_labels)
    fbeta = FBetaScore(task='binary', beta=0.5)
    acc = Accuracy(task='binary')

    preds_not_found = sum([int(pred == "Not Found") for pred in preds])
    preds_correct = sum(bin_preds)
    preds_wrong = len(preds) - preds_correct - preds_not_found

    return {"F1 Score": fbeta(preds_tensor, labels_tensor), 
            "Accuracy": acc(preds_tensor, labels_tensor),
            "Not Found": preds_not_found,
            "Wrong": preds_wrong,
            "Correct":preds_correct}





