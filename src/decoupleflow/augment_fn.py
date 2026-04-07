"""Data augmentation helpers for vision and NLP tasks.

Public APIs for users:
- Vision_augment: build a dataloader with two image augmentations per sample.
- NLP_augment: build augmented sentence/label pairs for text classification.
"""

import torch
from torch.utils.data import Dataset
import random
import nltk
from torchvision import transforms
from nltk.corpus import wordnet


class _AugmentedDataset(Dataset):
    """Dataset wrapper that materializes two augmented samples per item."""

    def __init__(self, original_dataset, **kwargs):
        self.original_dataset = original_dataset
        self.augmented_dataset = []
        self._make_augmentation(kwargs)
        
    def _make_augmentation(self, kwargs): 
        """Create and store augmented samples.

        Args:
            kwargs: Parameters for augmentation transform construction.
        """
        transform_function = _vision_transform(**kwargs)            
        for data, label in self.original_dataset:
            augmented_data_1, augmented_data_2 = transform_function.apply(data)
            self.augmented_dataset.append((augmented_data_1, label))
            self.augmented_dataset.append((augmented_data_2, label))
    
    def __len__(self):
        return len(self.augmented_dataset)
    
    def __getitem__(self, idx):
        data, label = self.augmented_dataset[idx]
        return data, label

class _vision_transform:
    """Strong/weak image augmentation pair used in contrastive pipelines."""

    def __init__(self, **kwargs):
        self.image_size = kwargs["image_size"]
        self.mean = kwargs["mean"]
        self.std = kwargs["std"]
    
        self.strong_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
        ])
        self.weak_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
        ])
        
    def apply(self, image):
        """Create paired image augmentations.

        Args:
            image: Input PIL image or image-like object.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Strong and weak augmented images.
        """
        strong_image = self.strong_transform(image)
        weak_image = self.weak_transform(image)
        return strong_image, weak_image

def _nlp_transform(
    original_sentences,
    original_labels,
    synonyms,
    probability,
    num_augments_per_sentence,
    include_original,
):
        """Apply random text augmentations per sentence.

        Args:
            original_sentences: Input sentence list.
            original_labels: Label list aligned with sentences.
            synonyms: Word-to-synonyms dictionary.
            probability: Augmentation strength factor.
            num_augments_per_sentence: Number of augmented outputs per sentence.
            include_original: Whether to keep original sentence in outputs.

        Returns:
            Tuple[list, list]: Augmented sentences and corresponding labels.
        """
        AUGMENTATION_METHODS = {
            'synonym_replacement': {'fn': _synonym_replacement, 'params': {'probability': probability, 'synonyms': synonyms}},
            'random_insertion': {'fn': _random_insertion, 'params': {'probability': probability, 'synonyms': synonyms}},
            'random_swap': {'fn': _random_swap, 'params': {'probability': probability}},
            'random_deletion': {'fn': _random_deletion, 'params': {'probability': probability}}
        }

        augmented_sentences = []
        augmented_labels = []
        for sentence, label in zip(original_sentences, original_labels):
            if include_original:
                augmented_sentences.append(sentence)
                augmented_labels.append(label)

            methods = list(AUGMENTATION_METHODS.values())
            if num_augments_per_sentence <= len(methods):
                chosen_augmentations = random.sample(methods, num_augments_per_sentence)
            else:
                chosen_augmentations = random.choices(methods, k=num_augments_per_sentence)

            for i in range(len(chosen_augmentations)):
                augment_fn = chosen_augmentations[i]['fn']
                augment_params = chosen_augmentations[i]['params']
                augmented_sentences.append(augment_fn(sentence, **augment_params))
                augmented_labels.append(label)
            
        return augmented_sentences, augmented_labels
        
def _extract_unique_words(dataset):
    """Collect unique tokens from sentence list.

    Args:
        dataset: Iterable sentence collection.

    Returns:
        set: Unique token set.
    """
    unique_words = set()
    for sentence in dataset:
        words = sentence.split()
        unique_words.update(words)
    return unique_words

def _ensure_wordnet_available(auto_download=False):
    """Ensure NLTK WordNet corpus is available before NLP augmentation.

    Args:
        auto_download: If True, try downloading missing corpus automatically.

    Raises:
        RuntimeError: If corpus is missing and cannot be downloaded.
    """
    try:
        wordnet.synsets("test")
    except LookupError as exc:
        if auto_download:
            download_ok = nltk.download("wordnet", quiet=True)
            if download_ok:
                return
            raise RuntimeError(
                "Failed to auto-download NLTK 'wordnet'. "
                "Please run: import nltk; nltk.download('wordnet')"
            ) from exc
        raise RuntimeError(
            "NLTK 'wordnet' corpus is required for NLP_augment but is not installed. "
            "Run: import nltk; nltk.download('wordnet') "
            "or call NLP_augment(..., auto_download_wordnet=True)."
        ) from exc
    
def _get_synonyms(dataset):
    """Build token-to-synonyms mapping from WordNet.

    Args:
        dataset: Iterable sentence collection.

    Returns:
        dict: Mapping from token to synonym list.
    """
    unique_words = _extract_unique_words(dataset)
    synonym_dict = {}
    for word in unique_words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym_list = []
            for syn in synonyms:
                for lemma in syn.lemmas():
                    synonym_list.append(lemma.name())
            synonym_dict[word] = list(set(synonym_list))
    return synonym_dict

def _synonym_replacement(sentence, probability, synonyms):
    """Randomly replace eligible words with synonyms.

    Args:
        sentence: Input sentence.
        probability: Replacement ratio.
        synonyms: Word-to-synonyms dictionary.

    Returns:
        str: Augmented sentence.
    """
    words = sentence.split()
    num_words = max(1, int(probability * len(words)))
    eligible_words = [word for word in words if word in synonyms]

    if len(eligible_words) == 0:
        return sentence

    words_to_replace = random.sample(eligible_words, min(num_words, len(eligible_words)))
       
    new_words = []
    for word in words:
        if word in words_to_replace:
            synonym_list = synonyms[word]
            new_word = random.choice(synonym_list)
            new_words.append(new_word)
        else:
            new_words.append(word)

    return ' '.join(new_words)

def _random_insertion(sentence, probability, synonyms):
    """Insert random synonyms at random positions.

    Args:
        sentence: Input sentence.
        probability: Insertion ratio.
        synonyms: Word-to-synonyms dictionary.

    Returns:
        str: Augmented sentence.
    """
    words = sentence.split()
    num_words = max(1, int(probability * len(words)))
    eligible_words = [word for word in words if word in synonyms]

    if len(eligible_words) == 0:
        return sentence

    words_to_insert = random.sample(eligible_words, min(num_words, len(eligible_words)))

    for word in words_to_insert:
        synonym_list = synonyms[word]
        random_synonym = random.choice(synonym_list)
        random_idx = random.randint(0, len(words))
        words.insert(random_idx, random_synonym)
    return ' '.join(words)

def _random_swap(sentence, probability):
    """Randomly swap word positions multiple times.

    Args:
        sentence: Input sentence.
        probability: Swap ratio.

    Returns:
        str: Augmented sentence.
    """
    words = sentence.split()
    if len(words) <= 1:
        return sentence
    num_words = max(1, int(probability * len(words)))
    for _ in range(num_words):
        idx1 = random.randint(0, len(words)-1)
        idx2 = random.randint(0, len(words)-1)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

def _random_deletion(sentence, probability):
    """Drop words with given probability while keeping at least one token.

    Args:
        sentence: Input sentence.
        probability: Deletion probability per token.

    Returns:
        str: Augmented sentence.
    """
    words = sentence.split()
    if len(words) <= 1:
        return sentence

    new_words = []
    for word in words:
        if random.uniform(0, 1) > probability:
            new_words.append(word)

    if len(new_words) == 0:
        return random.choice(words)
    return ' '.join(new_words)
        
            
def Vision_augment(dataset, image_size=32, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """Create vision-augmented dataset.

    Args:
        dataset: Source dataset. It should return (image, label), where image is
            PIL image or image-like input accepted by torchvision.
        image_size: Resize/crop image size.
        mean: Normalize mean.
        std: Normalize std.

    Returns:
        Dataset: Augmented dataset with 2x samples (strong + weak view per input).
    """
    augmented_dataset = _AugmentedDataset(dataset, image_size=image_size, mean=mean, std=std)
    return augmented_dataset

def NLP_augment(
    original_sentences,
    original_labels,
    probability=0.1,
    auto_download_wordnet=False,
    num_augments_per_sentence=2,
    include_original=False,
):
    """Create NLP-augmented sentences and labels.

    Args:
        original_sentences: Input sentence list.
        original_labels: Label list aligned with sentences (same length as
            original_sentences).
        probability: Augmentation strength factor.
        auto_download_wordnet: Whether to auto-download missing WordNet corpus.
        num_augments_per_sentence: Number of augmented outputs generated per
            input sentence.
        include_original: Whether to include the original sentence in outputs.

    Returns:
        Tuple[list, list]: Augmented sentences and labels.
    """
    if len(original_sentences) != len(original_labels):
        raise ValueError("original_sentences and original_labels must have same length.")
    if num_augments_per_sentence < 1:
        raise ValueError("num_augments_per_sentence must be >= 1.")

    _ensure_wordnet_available(auto_download=auto_download_wordnet)
    synonyms = _get_synonyms(original_sentences)
    augmented_sentences, augmented_labels = _nlp_transform(
        original_sentences,
        original_labels,
        synonyms,
        probability,
        num_augments_per_sentence,
        include_original,
    )
    return augmented_sentences, augmented_labels