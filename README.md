# AI-pizza-identifier
An AI model to identify toppings on uncooked pizzas.

## Useful commands

- `source ~/.venvs/AI-pizza-identifier/bin/activate`
- `python3 train_pizza.py --epochs 30`
- `python3 predict_one.py --image pizza_dataset/other/T1.png`

---

## Project overview

This project trains a lightweight image classification model to distinguish between three classes: **Cheese**, **Pepperoni**, and **Hamburger**. The dataset is read directly from the `pizza_dataset/` folder, with labels inferred from the first character of each filename.  

The outputs are stored in the `outputs/` folder:
- `final_model.keras` – the trained model
- `class_index.json` – mapping of class names to indices

---

## Code structure

### `train_pizza.py`
- Splits the dataset into training (80%) and validation (20%) sets, and prints which images are in each split.
- Uses a MobileNetV2 backbone with ImageNet weights, frozen to avoid overfitting on the very small dataset.
- Adds a simple dense classification head to adapt the model to the three pizza classes.
- Applies basic augmentation (horizontal flip) and balances classes through weighted loss.
- Saves the trained model and class mapping for later use.

### `predict_one.py`
- Loads the trained model and class mapping.
- Takes a single input image, applies the same preprocessing as training, and outputs class probabilities along with the predicted label.

---

## Design choices

- **MobileNetV2 backbone**: chosen for efficiency and suitability for small-scale datasets.
- **Frozen feature extractor**: prevents overfitting given the very limited number of training images.
- **Simple augmentation**: light flipping increases variability without distorting the dataset.
- **Class weighting**: addresses imbalance by ensuring minority classes contribute proportionally to training.
- **Fixed preprocessing**: consistent input resizing and normalization ensures reproducibility across training and inference.

---

## Current status

This project is currently **on hold** due to the lack of a sufficient training dataset. It has been tested with a very small set of images (around 10), where it demonstrated that the pipeline works end-to-end, though the results are limited by the data size.
