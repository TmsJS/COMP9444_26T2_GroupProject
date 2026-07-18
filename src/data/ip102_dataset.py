from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class IP102Dataset(Dataset):
    """
    PyTorch Dataset for the IP102 insect classification dataset.
    Each line in an annotation file has the following format:

        00002.jpg 0

    The first value is the image filename.
    The second value is the class label, ranging from 0 to 101.
    """

    def __init__(self, images_dir, annotation_file, transform=None):
        """
        Initialize the IP102 dataset.
        Args:
            images_dir:
                Path to the folder containing all IP102 images.

            annotation_file:
                Path to train.txt, val.txt, or test.txt.

            transform:
                Optional image preprocessing and data augmentation.
                Different models can provide different transforms.
        """

        # Convert the image directory into a Path object.
        # Path makes it easier to combine directories and filenames.
        self.images_dir = Path(images_dir)

        # Save the image transform.
        # It will be applied when an image is loaded.
        self.transform = transform

        # Store all (image_name, label) pairs in this list.
        self.samples = []

        # Open the annotation file.
        with open(annotation_file, "r") as file:
            # Read the annotation file one line at a time.
            for line in file:
                # Remove spaces and the newline character from both ends.
                line = line.strip()
                # Skip empty lines if the annotation file contains any.
                if not line:
                    continue
                # Example:
                # "00002.jpg 0"
                # becomes:
                # image_name = "00002.jpg"
                # label = "0"
                image_name, label = line.split()

                # Convert the label from a string to an integer.
                label = int(label)

                # Save the image filename and its label.
                self.samples.append((image_name, label))

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        PyTorch uses this method when calling:
            len(dataset)
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Load and return one image and its label.
        PyTorch uses this method when calling:
            dataset[index]

        Args:
            index:
                The position of the requested sample.

        Returns:
            image:
                The loaded and transformed image.

            label:
                The integer class label from 0 to 101.
        """

        # Get the image filename and label at the requested position.
        image_name, label = self.samples[index]

        # Build the complete image path.
        #
        # Example:
        # images_dir = ".../ip102_v1.1/images"
        # image_name = "00002.jpg"
        #
        # Result:
        # ".../ip102_v1.1/images/00002.jpg"
        image_path = self.images_dir / image_name

        # Open the image.
        #
        # convert("RGB") ensures every image has exactly three channels,
        # even if the original image is grayscale or uses another format.
        image = Image.open(image_path).convert("RGB")

        # Apply preprocessing or data augmentation if it was provided.
        #
        # For example:
        # Resize, RandomHorizontalFlip, ToTensor and Normalize.
        if self.transform is not None:
            image = self.transform(image)

        # Return one image and its corresponding class label.
        return image, label