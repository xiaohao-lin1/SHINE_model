import torch
from torch import nn
# from new_utils.ContrastiveLoss import ContrastiveLoss

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label):
        # Calculate the contrastive loss as per the formula given in the Keras code
        square_pred = output ** 2
        margin_square = torch.clamp(self.margin - output, min=0.0) ** 2
        loss = torch.mean((1 - label) * square_pred + label * margin_square)
        return loss

import torch
import torch.nn.functional as F

class CosineSimilarity_With_Cross_Entropy(torch.nn.Module):
    def __init__(self, lambda_value=0.1):
        super(CosineSimilarity_With_Cross_Entropy, self).__init__()
        self.lambda_value = lambda_value

    def forward(self, outputs, labels, left_brain_features, right_brain_features):
        # Calculate the cosine similarity between left and right brain features
        cosine_similarity = F.cosine_similarity(left_brain_features.flatten(1), right_brain_features.flatten(1)).mean()

        # Calculate the cross-entropy loss between outputs and labels
        cross_entropy_loss = F.cross_entropy(outputs, labels)

        # Calculate the final loss, minimising the cosine similarity
        final_loss = self.lambda_value * cosine_similarity + (1 - self.lambda_value) * cross_entropy_loss

        return cosine_similarity, cross_entropy_loss, final_loss

class NegativeCosineSimilarity_With_Cross_Entropy(torch.nn.Module):
    def __init__(self, lambda_value=0.1):
        super(NegativeCosineSimilarity_With_Cross_Entropy, self).__init__()
        self.lambda_value = lambda_value

    def forward(self, outputs, labels, left_brain_features, right_brain_features):
        # Calculate the cosine similarity between left and right brain features
        cosine_similarity = F.cosine_similarity(left_brain_features.flatten(1), right_brain_features.flatten(1)).mean()
        cosine_similarity *= -1

        # Calculate the cross-entropy loss between outputs and labels
        cross_entropy_loss = F.cross_entropy(outputs, labels)

        # Calculate the final loss, minimising the cosine similarity
        final_loss = self.lambda_value * cosine_similarity + (1 - self.lambda_value) * cross_entropy_loss

        return cosine_similarity, cross_entropy_loss, final_loss

class CooperativeLearningLoss(nn.Module):
    def __init__(self, alpha):
        super(CooperativeLearningLoss, self).__init__()
        self.alpha = alpha  # Weighting factor for cooperative reward
        self.classification_loss = nn.functional.cross_entropy

    def forward(self, combined_output, targets, left_output, right_output):
        # (outputs, labels, left_brain_features, right_brain_features
        # Standard classification loss for combined features
        loss_combined = self.classification_loss(combined_output, targets)

        # Classification loss for independent features
        loss_left = self.classification_loss(left_output, targets)
        loss_right = self.classification_loss(right_output, targets)

        # Cooperative reward: improvement in loss when using combined features
        cooperative_reward = (loss_left + loss_right) / 2 - loss_combined

        # Final loss: classification loss - reward (ensure it doesn't become negative)
        final_loss = loss_combined - self.alpha * torch.clamp(cooperative_reward, min=0)
        # print('loss_left', loss_left)
        # print('loss_right', loss_right)
        # print('loss_combined', loss_combined)
        # print('cooperative_reward', cooperative_reward)
        # print('final_loss', final_loss)
        # print('Alpha is ', self.alpha)
        # print('Final loss == loss_combined', final_loss == loss_combined)

        return final_loss, loss_left, loss_right, loss_combined

# import torch.nn.functional as

# def get_criterion(criterion, **kwargs):
#     criterion_dict = {
#         'nll': nn.NLLLoss,
#         'nll_F': nn.functional.nll_loss,
#         'ce': nn.CrossEntropyLoss,
#         'ce_F': nn.functional.cross_entropy,
#         'ContrastiveLoss': ContrastiveLoss,
#         'CosineSimilarity_With_Cross_Entropy': CosineSimilarity_With_Cross_Entropy,
#         'NegativeCosineSimilarity_With_Cross_Entropy': NegativeCosineSimilarity_With_Cross_Entropy,
#         'CooperativeLearningLoss': CooperativeLearningLoss,
#     }
#
#     # Only pass kwargs to the relevant loss function
#     if criterion in criterion_dict:
#         return criterion_dict[criterion](**kwargs)
#     else:
#         return criterion_dict[criterion]

import torch.nn as nn

def get_criterion(criterion, **kwargs):
    criterion_dict = {
        'nll': nn.NLLLoss,
        'nll_F': nn.functional.nll_loss,
        'ce': nn.CrossEntropyLoss,
        'ce_F': nn.functional.cross_entropy,
        'ContrastiveLoss': ContrastiveLoss,
        'CosineSimilarity_With_Cross_Entropy': CosineSimilarity_With_Cross_Entropy,
        'NegativeCosineSimilarity_With_Cross_Entropy': NegativeCosineSimilarity_With_Cross_Entropy,
        'CooperativeLearningLoss': CooperativeLearningLoss,
    }

    if criterion in criterion_dict:
        if 'F' in criterion:  # For functional forms
            return lambda *args, **kwargs: criterion_dict[criterion](*args, **kwargs)
        else:  # For class-based forms
            return criterion_dict[criterion](**kwargs)
    else:
        raise ValueError(f"Criterion '{criterion}' not recognized")


if __name__ == '__main__':
    loss = get_criterion('ce_F')
    print(loss)
    # from model_configs.DCN_split_spatial_kernel_3_layers_cross_attention import Configs
    # args = Configs()
    #
    # import torch
    # # Assuming a binary classification task
    num_samples = 10
    num_classes = 2
    #
    # # Synthetic outputs from the model
    combined_output = torch.randn(num_samples, num_classes)
    # left_output = torch.randn(num_samples, num_classes)
    # right_output = torch.randn(num_samples, num_classes)
    #
    # # Random target labels
    targets = torch.randint(0, num_classes, (num_samples,))
    # loss_function = CooperativeLearningLoss(alpha=0)
    # loss = loss_function(combined_output, targets, left_output, right_output)
    # print(f"Calculated Loss: {loss.item()}")
