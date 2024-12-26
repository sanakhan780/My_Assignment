import torch.nn.functional as F

def calculate_loss(predictions, targets):
    return F.cross_entropy(predictions, targets)

def calculate_metrics(predictions, targets):
    correct = (predictions.argmax(1) == targets).sum().item()
    total = targets.numel()
    return correct / total

def calculate_iou(predictions, targets, num_classes):
    iou_list = []
    for cls in range(num_classes):
        intersection = ((predictions == cls) & (targets == cls)).sum().item()
        union = ((predictions == cls) | (targets == cls)).sum().item()
        iou_list.append(intersection / union if union > 0 else float('nan'))
    
    return sum(iou_list) / len(iou_list)  # Average IoU across classes.
