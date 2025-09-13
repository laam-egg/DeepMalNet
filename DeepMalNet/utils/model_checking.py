def model_sanity_check(model, loader, device):
    import torch
    from sklearn.metrics import confusion_matrix, classification_report
    from tqdm import tqdm

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in tqdm(loader):
            X = X.to(device)
            logits = model(X)
            preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy().squeeze()
            labels = y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
